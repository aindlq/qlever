// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorDistanceExpression.h"

#include <absl/strings/charconv.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "backports/StartsWithAndEndsWith.h"
#include "engine/sparqlExpressions/LiteralExpression.h"
#include "engine/sparqlExpressions/SparqlExpressionGenerators.h"
#include "index/ExportIds.h"
#include "index/Index.h"
#include "index/IndexImpl.h"
#include "parser/NormalizedString.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "util/Exception.h"
#include "util/Log.h"
#include "util/Timer.h"

namespace sparqlExpression {

namespace {
using qlever::vector::VectorIndex;

// How often the per-row loop polls the cancellation handle -- frequent enough
// for prompt cancellation, rare enough to stay out of the hot path.
constexpr size_t CHECK_INTERRUPT_PERIOD = 65536;

// Below this many rows the fork/join overhead of the parallel scan outweighs
// the win, so the per-row distance loops stay on the serial path.
constexpr size_t VEC_DISTANCE_PARALLEL_THRESHOLD = 2048;

// Rows per work item handed to a worker. The per-row cost is uniform (one
// rowmap binary search + one SIMD kernel call), so a coarse fixed chunk keeps
// scheduling overhead negligible; it is also the cancellation-poll granularity.
constexpr size_t VEC_DISTANCE_PARALLEL_CHUNK = 1024;

#ifdef _OPENMP
// Worker cap for the parallel scan. QLever answers many queries concurrently on
// a thread pool already sized to the hardware, so letting a single
// `vec:distance` grab every core would oversubscribe when several queries run
// at once. Cap at the hardware concurrency AND OpenMP's configured maximum (so
// `OMP_NUM_THREADS` can still lower it). The scan is memory-bandwidth bound, so
// there is nothing to gain past the physical core count anyway. (No per-query
// thread-count config is reachable from the expression's `EvaluationContext`;
// if one is exposed here later, prefer it over the hardware concurrency.)
int vectorDistanceThreadCap() {
  unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  return std::max(1, std::min(omp_get_max_threads(), static_cast<int>(hw)));
}
#endif

// Parse a comma-separated list of finite floats, e.g. "0.1,-0.2,0.3" (the
// string form of an explicit query vector, and the output format of
// `vec:embed`). Returns `nullopt` (instead of throwing) on anything that is
// not a non-empty list of finite numbers, because per-row values that are not
// float lists simply resolve to `UNDEF`.
std::optional<std::vector<float>> tryParseFloatList(std::string_view s) {
  std::vector<float> out;
  for (std::string_view token :
       absl::StrSplit(s, ',', absl::SkipWhitespace{})) {
    while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) {
      token.remove_prefix(1);
    }
    while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) {
      token.remove_suffix(1);
    }
    float value{};
    auto [ptr, ec] =
        absl::from_chars(token.data(), token.data() + token.size(), value);
    if (ec != std::errc{} || ptr != token.data() + token.size() ||
        !std::isfinite(value)) {
      return std::nullopt;
    }
    out.push_back(value);
  }
  if (out.empty()) {
    return std::nullopt;
  }
  return out;
}

// A resolved vector source (one side of the distance):
//  * `monostate`     -> no vector (the row evaluates to `UNDEF`);
//  * `Id`            -> an entity with a stored vector in the index;
//  * `vector<float>` -> an explicit (parsed) query vector.
using ResolvedSource = std::variant<std::monostate, Id, std::vector<float>>;

// Whether a source is being resolved ONCE (a constant sub-expression) or per
// row. A constant that is a malformed or dimension-mismatched float list
// throws a loud, user-facing error (it is certainly a query mistake); per-row
// values never throw and resolve to `monostate` (-> `UNDEF`) instead, so a
// mixed column cannot abort a scan.
enum class SourceMode { Constant, PerRow };

// Resolve one evaluated source element (a `ValueId` or an
// `IdOrLocalVocabEntry`) to a `ResolvedSource`: the entity path first (a
// cheap rowmap lookup), then the float-list path for literal content.
template <typename E>
ResolvedSource resolveSource(const E& element, const VectorIndex& vidx,
                             const std::string& indexName,
                             const EvaluationContext* context,
                             SourceMode mode) {
  using T = std::decay_t<E>;
  static_assert(std::is_same_v<T, ValueId> ||
                std::is_same_v<T, IdOrLocalVocabEntry>);

  // The entity path: a value id whose entity has a (live) stored vector.
  const ValueId* id = nullptr;
  if constexpr (std::is_same_v<T, ValueId>) {
    id = &element;
  } else {
    id = std::get_if<ValueId>(&element);
  }
  if (id != nullptr) {
    if (id->isUndefined()) {
      return std::monostate{};
    }
    if (vidx.hasVector(*id)) {
      return *id;
    }
  }

  // The float-list path: literal content that parses as floats (an inline
  // "0.1,0.2,..." or the output of `vec:embed`). An IRI/entity WITHOUT a
  // stored vector also ends up here and resolves to `monostate`, because an
  // IRI never parses as a float list.
  std::optional<ad_utility::triple_component::LiteralOrIri> term;
  if (id != nullptr) {
    term = ql::exportIds::idToLiteralOrIri(context->_qec.getIndex().getImpl(),
                                           *id, context->_localVocab);
  } else if constexpr (std::is_same_v<T, IdOrLocalVocabEntry>) {
    term = static_cast<const ad_utility::triple_component::LiteralOrIri&>(
        std::get<LocalVocabEntry>(element));
  }
  if (!term.has_value() || !term->isLiteral()) {
    return std::monostate{};
  }
  const auto& literal = term->getLiteral();
  // A TYPED query-vector literal `"f0,f1,..."^^<.../vec/MODEL/PRECISION>` (the
  // output of `vec:embed`/`vec:vector` -- possibly of a DIFFERENT index)
  // carries its embedding space in the datatype. VALIDATE it against THIS
  // index before even parsing the floats: same-dimension vectors from
  // different models are numerically computable but semantically garbage, so
  // a space mismatch makes the row UNDEF (or throws for a constant source,
  // like the dimension error below). The model check is skipped when EITHER
  // side declares no model, so vector-only indices (no `embeddingModel`) stay
  // usable. A PLAIN untyped float-list string (an inline query vector) skips
  // all of this and is only dimension-checked below.
  if (literal.hasDatatype()) {
    std::string_view datatypeIri = asStringViewUnsafe(literal.getDatatype());
    if (ql::starts_with(datatypeIri,
                        qlever::vector::VEC_QUERY_DATATYPE_PREFIX)) {
      std::optional<std::pair<std::string, std::string>> space =
          qlever::vector::parseVectorQueryDatatypeIri(datatypeIri);
      const auto& config = vidx.metadata().config_;
      const std::string precision = qlever::vector::toString(config.scalar_);
      const bool modelComparable =
          space.has_value() &&
          (space->first.empty() || config.embeddingModel_.empty() ||
           space->first == config.embeddingModel_);
      if (!space.has_value() || space->second != precision ||
          !modelComparable) {
        if (mode == SourceMode::Constant) {
          AD_THROW(absl::StrCat(
              "vec:distance: the typed query vector's datatype <", datatypeIri,
              "> does not match vector index \"", indexName, "\" (model \"",
              config.embeddingModel_.empty() ? "<none>"
                                             : config.embeddingModel_,
              "\", precision \"", precision,
              "\"): vectors are only comparable within one embedding space "
              "(same model, same precision, same dimension)."));
        }
        return std::monostate{};
      }
    }
  }
  std::string_view content = asStringViewUnsafe(term->getContent());
  std::optional<std::vector<float>> floats = tryParseFloatList(content);
  if (!floats.has_value()) {
    if (mode == SourceMode::Constant) {
      AD_THROW(absl::StrCat(
          "vec:distance: the constant string \"", content,
          "\" is not a comma-separated list of finite numbers (a query "
          "vector), and no entity with a vector in index \"",
          indexName, "\" matches it."));
    }
    return std::monostate{};
  }
  if (floats->size() != vidx.dimensions()) {
    if (mode == SourceMode::Constant) {
      AD_THROW(absl::StrCat("vec:distance: the query vector has dimension ",
                            floats->size(), ", but vector index \"", indexName,
                            "\" has dimension ", vidx.dimensions(), "."));
    }
    return std::monostate{};
  }
  return std::move(floats).value();
}

// Convert a raw distance (`NaN` = no result) to the result `Id`.
Id toDistanceId(float distance) {
  return std::isnan(distance)
             ? Id::makeUndefined()
             : Id::makeFromDouble(static_cast<double>(distance));
}

// The one-shot distance between two resolved sources (used when neither side
// is a fixed query point that would pay for a reusable `DistanceComputer`).
Id pairDistance(const ResolvedSource& a, const ResolvedSource& b,
                const VectorIndex& vidx) {
  if (std::holds_alternative<std::monostate>(a) ||
      std::holds_alternative<std::monostate>(b)) {
    return Id::makeUndefined();
  }
  const Id* idA = std::get_if<Id>(&a);
  const Id* idB = std::get_if<Id>(&b);
  if (idA != nullptr && idB != nullptr) {
    return toDistanceId(vidx.distance(*idA, *idB));
  }
  if (idA != nullptr) {
    return toDistanceId(vidx.distance(*idA, std::get<std::vector<float>>(b)));
  }
  if (idB != nullptr) {
    return toDistanceId(vidx.distance(*idB, std::get<std::vector<float>>(a)));
  }
  return toDistanceId(vidx.distance(std::get<std::vector<float>>(a),
                                    std::get<std::vector<float>>(b)));
}
}  // namespace

// _____________________________________________________________________________
VectorDistanceExpression::VectorDistanceExpression(std::string indexName,
                                                   Ptr source1, Ptr source2)
    : indexName_{std::move(indexName)},
      sources_{std::move(source1), std::move(source2)} {}

// _____________________________________________________________________________
ExpressionResult VectorDistanceExpression::evaluate(
    EvaluationContext* context) const {
  const Index& index = context->_qec.getIndex();
  auto vectorIndex = qlever::vector::getVectorIndex(index, indexName_);
  if (!vectorIndex) {
    AD_THROW(absl::StrCat("vec:distance: no vector index named \"", indexName_,
                          "\" is loaded."));
  }
  const VectorIndex& vidx = *vectorIndex;
  const size_t resultSize = context->size();

  // How many rows of this `evaluate` block actually resolved to a distance
  // (UNDEF rows are not counted); drives the one INFO timing line below.
  size_t numDistances = 0;

  // Fill `out[0..resultSize)` with `perRowDistance(i)` for every row, running
  // the per-row work in parallel once the block is large enough. Correctness
  // rests on `perRowDistance` being a PURE function of `i`: it only reads const
  // index state (rowmap binary search + mmap SIMD) plus the serially
  // materialized per-row buffer and writes nothing shared, so every `out[i]` is
  // an independent, disjoint write and the parallel result is bit-identical to
  // (and as deterministic as) the serial one. `numDistances` is combined via a
  // reduction, so its total is order-independent and matches the serial count
  // exactly. Cancellation follows the standard OpenMP pattern: check before the
  // region, poll a non-throwing `isCancelled()` per chunk into a shared atomic
  // flag inside the region (never unwinding an exception out of it), then throw
  // once after the region.
  auto fillDistances = [&](VectorWithMemoryLimit<Id>& out,
                           const auto& perRowDistance) {
    // Pre-size (all UNDEF) so workers write DISJOINT slots with zero allocator
    // contention -- never `reserve` + `push_back` from multiple threads.
    out.resize(resultSize, Id::makeUndefined());
    context->cancellationHandle_->throwIfCancelled();
    size_t produced = 0;
#ifdef _OPENMP
    if (resultSize >= VEC_DISTANCE_PARALLEL_THRESHOLD) {
      std::atomic<bool> cancelled{false};
      const int numThreads = vectorDistanceThreadCap();
#pragma omp parallel for schedule(dynamic, VEC_DISTANCE_PARALLEL_CHUNK) \
    num_threads(numThreads) reduction(+ : produced)
      for (size_t i = 0; i < resultSize; ++i) {
        // An exception must never unwind out of the OpenMP region: on
        // cancellation drain the remaining iterations cheaply and throw
        // afterwards. (`perRowDistance` itself never throws -- `resolveSource`
        // in `PerRow` mode and the `DistanceComputer`/`pairDistance` calls are
        // all no-throw.)
        if (cancelled.load(std::memory_order_relaxed)) {
          continue;
        }
        if (i % VEC_DISTANCE_PARALLEL_CHUNK == 0 &&
            context->cancellationHandle_->isCancelled()) {
          cancelled.store(true, std::memory_order_relaxed);
          continue;
        }
        Id result = perRowDistance(i);
        out[i] = result;
        if (!result.isUndefined()) {
          ++produced;
        }
      }
      if (cancelled.load(std::memory_order_relaxed)) {
        // Re-check through the handle so the proper timeout/manual-cancel
        // exception is raised with the right message.
        context->cancellationHandle_->throwIfCancelled();
      }
      numDistances += produced;
      return;
    }
#endif
    // Serial fallback: a small block (parallel overhead not worth it) or a
    // build without OpenMP.
    size_t sinceCheck = 0;
    for (size_t i = 0; i < resultSize; ++i) {
      Id result = perRowDistance(i);
      out[i] = result;
      if (!result.isUndefined()) {
        ++produced;
      }
      if (++sinceCheck == CHECK_INTERRUPT_PERIOD) {
        sinceCheck = 0;
        context->cancellationHandle_->throwIfCancelled();
      }
    }
    numDistances += produced;
  };

  // Per-row source against a FIXED source: encode the fixed query point ONCE
  // into a reusable `DistanceComputer` (one rowmap lookup + one SIMD kernel
  // call per row). Both `resolveSource` (const index/vocab reads + local float
  // parsing, no shared mutation) and `DistanceComputer::operator()` (rowmap
  // binary search + mmap SIMD into a LOCAL buffer, no shared mutable state) are
  // pure const reads, so the per-row closure is safe to run concurrently.
  auto perRowAgainstConstant = [&](const ResolvedSource& constant,
                                   auto&& perRow) -> ExpressionResult {
    if (std::holds_alternative<std::monostate>(constant)) {
      // The fixed side resolves to no vector -> every row is UNDEF. (A
      // constant is a valid constant result, it broadcasts to all rows.)
      return Id::makeUndefined();
    }
    std::optional<VectorIndex::DistanceComputer> computer =
        std::holds_alternative<Id>(constant)
            ? vidx.makeDistanceComputerByEntity(std::get<Id>(constant))
            : std::optional{vidx.makeDistanceComputer(
                  std::get<std::vector<float>>(constant))};
    // `resolveSource` only returns an entity after checking `hasVector`.
    AD_CORRECTNESS_CHECK(computer.has_value());

    // Materialize the per-row source into a random-access buffer FIRST (cheap:
    // copies ValueIds/handles, no SIMD, single-threaded) so the expensive
    // resolve + SIMD distance can then run over disjoint `out[i]` in parallel,
    // keeping an exact row->index correspondence.
    auto gen = detail::makeGenerator(AD_FWD(perRow), resultSize, context);
    using Element = ql::ranges::range_value_t<decltype(gen)>;
    std::vector<Element> elements;
    elements.reserve(resultSize);
    for (auto&& element : gen) {
      elements.emplace_back(AD_FWD(element));
    }
    AD_CORRECTNESS_CHECK(elements.size() == resultSize);

    VectorWithMemoryLimit<Id> out{context->_allocator};
    fillDistances(out, [&](size_t i) -> Id {
      ResolvedSource row = resolveSource(elements[i], vidx, indexName_, context,
                                         SourceMode::PerRow);
      if (const Id* entity = std::get_if<Id>(&row)) {
        return toDistanceId((*computer)(*entity));
      }
      if (const auto* vec = std::get_if<std::vector<float>>(&row)) {
        return toDistanceId((*computer)(*vec));
      }
      return Id::makeUndefined();
    });
    return out;
  };

  // Both sources per-row: materialize both columns, then compute each pair via
  // `fillDistances` (parallel for a large block). `pairDistance` dispatches to
  // the const `VectorIndex::distance` overloads (rowmap lookups + local encode
  // buffers + mmap SIMD, entity<->entity uses the stored bytes directly), so
  // the per-row closure is likewise a pure concurrent read.
  auto perRowPair = [&](auto&& perRow1, auto&& perRow2) -> ExpressionResult {
    auto gen1 = detail::makeGenerator(AD_FWD(perRow1), resultSize, context);
    auto gen2 = detail::makeGenerator(AD_FWD(perRow2), resultSize, context);
    using Element1 = ql::ranges::range_value_t<decltype(gen1)>;
    using Element2 = ql::ranges::range_value_t<decltype(gen2)>;
    std::vector<Element1> elements1;
    std::vector<Element2> elements2;
    elements1.reserve(resultSize);
    elements2.reserve(resultSize);
    for (auto&& element : gen1) {
      elements1.emplace_back(AD_FWD(element));
    }
    for (auto&& element : gen2) {
      elements2.emplace_back(AD_FWD(element));
    }
    AD_CORRECTNESS_CHECK(elements1.size() == resultSize &&
                         elements2.size() == resultSize);

    VectorWithMemoryLimit<Id> out{context->_allocator};
    fillDistances(out, [&](size_t i) -> Id {
      ResolvedSource a = resolveSource(elements1[i], vidx, indexName_, context,
                                       SourceMode::PerRow);
      ResolvedSource b = resolveSource(elements2[i], vidx, indexName_, context,
                                       SourceMode::PerRow);
      return pairDistance(a, b, vidx);
    });
    return out;
  };

  ExpressionResult result1 = sources_[0]->evaluate(context);
  ExpressionResult result2 = sources_[1]->evaluate(context);

  // Time ONLY the distance computation below (row resolution + the
  // `DistanceComputer`/SIMD kernel calls) -- the child evaluation above (e.g.
  // a `vec:embed`) is timed and logged separately.
  ad_utility::Timer distanceTimer{ad_utility::Timer::Started};

  // Branch on constness: the expression framework yields a CONSTANT result
  // for a constant sub-expression, so a constant source (an inline float
  // list, a constant entity IRI, a `vec:embed` of constants, ...) is
  // parsed/looked up/encoded exactly once, never per row.
  ExpressionResult result = std::visit(
      [&](auto&& in1, auto&& in2) -> ExpressionResult {
        using T1 = std::decay_t<decltype(in1)>;
        using T2 = std::decay_t<decltype(in2)>;
        if constexpr (isConstantResult<T1> && isConstantResult<T2>) {
          // Both constant -> a single constant distance.
          ResolvedSource a = resolveSource(in1, vidx, indexName_, context,
                                           SourceMode::Constant);
          ResolvedSource b = resolveSource(in2, vidx, indexName_, context,
                                           SourceMode::Constant);
          Id distance = pairDistance(a, b, vidx);
          if (!distance.isUndefined()) {
            ++numDistances;
          }
          return distance;
        } else if constexpr (isConstantResult<T1>) {
          return perRowAgainstConstant(
              resolveSource(in1, vidx, indexName_, context,
                            SourceMode::Constant),
              AD_FWD(in2));
        } else if constexpr (isConstantResult<T2>) {
          // All supported metrics are symmetric, so the sides may swap.
          return perRowAgainstConstant(
              resolveSource(in2, vidx, indexName_, context,
                            SourceMode::Constant),
              AD_FWD(in1));
        } else {
          return perRowPair(AD_FWD(in1), AD_FWD(in2));
        }
      },
      std::move(result1), std::move(result2));
  // A block that computed no distance at all (everything UNDEF) stays silent.
  if (numDistances > 0) {
    AD_LOG_INFO << "vec:distance[" << indexName_ << "]: " << numDistances
                << " distances, dim " << vidx.dimensions() << " "
                << qlever::vector::toString(vidx.metadata().config_.scalar_)
                << ", in " << distanceTimer.msecs().count() << " ms"
                << std::endl;
  }
  return result;
}

// _____________________________________________________________________________
std::string VectorDistanceExpression::getCacheKey(
    const VariableToColumnMap& varColMap) const {
  return absl::StrCat("VEC_DISTANCE(", indexName_, "|",
                      sources_[0]->getCacheKey(varColMap), "|",
                      sources_[1]->getCacheKey(varColMap), ")");
}

// _____________________________________________________________________________
ql::span<SparqlExpression::Ptr> VectorDistanceExpression::childrenImpl() {
  return {sources_.data(), sources_.size()};
}

// _____________________________________________________________________________
std::string parseVectorIndexIriArgument(const SparqlExpression* arg,
                                        std::string_view functionName) {
  if (const auto* iri = dynamic_cast<const IriExpression*>(arg)) {
    std::optional<std::string> name = qlever::vector::indexNameFromMetadataIri(
        iri->value().toStringRepresentation());
    if (name.has_value()) {
      return std::move(name).value();
    }
  }
  AD_THROW(absl::StrCat(
      functionName,
      ": the first argument must be the index IRI -- the same resource that "
      "carries the index's metadata triples -- of the form ",
      qlever::vector::VECTOR_METADATA_SUBJECT_PREFIX, "NAME>, e.g. ",
      functionName, "(", qlever::vector::VECTOR_METADATA_SUBJECT_PREFIX,
      "clip>, ...)."));
}

// _____________________________________________________________________________
SparqlExpression::Ptr makeVectorDistanceExpression(
    std::vector<SparqlExpression::Ptr> args) {
  if (args.size() != 3) {
    AD_THROW(absl::StrCat(
        "vec:distance takes three arguments: the index IRI (",
        qlever::vector::VECTOR_METADATA_SUBJECT_PREFIX,
        "NAME>) and two vector sources, each of which is an entity with a "
        "stored vector or a comma-separated float-list string; got ",
        args.size(), "."));
  }
  std::string indexName =
      parseVectorIndexIriArgument(args[0].get(), "vec:distance");
  return std::make_unique<VectorDistanceExpression>(
      std::move(indexName), std::move(args[1]), std::move(args[2]));
}

}  // namespace sparqlExpression
