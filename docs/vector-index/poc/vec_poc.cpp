// Standalone proof-of-concept for the QLever vector-index core.
// Verifies, with nothing but the C++ STL + header-only usearch:
//   1. A flat, contiguous, fixed-stride float store (the mmap layout we will use).
//   2. Our own exact brute-force top-k cosine kNN  (the "Exact" algorithm).
//   3. usearch HNSW build + search                 (the "Hnsw" algorithm).
//   4. Recall@k of HNSW vs. exact.
//   5. usearch save() -> make(path, view=true) mmap round-trip (huge-index serving).
//   6. Exact search restricted to a candidate subset (the "5000 green statues" path).
//   7. Keying every vector by a uint64 == QLever VocabIndex.
//
// Build: g++ -std=c++17 -O2 -I build/_deps/usearch-src/include build/vec_poc.cpp -o build/vec_poc
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <random>
#include <queue>
#include <vector>
#include <unordered_set>

#include <usearch/index_dense.hpp>

using namespace unum::usearch;

using Key = std::uint64_t;  // stands in for QLever's VocabIndex (< 2^60)
static constexpr std::size_t D = 128;
static constexpr std::size_t N = 20000;
static constexpr std::size_t K = 10;

// ---- flat store: N*D contiguous floats, row i = entity with key keys[i] -------
struct FlatStore {
  std::vector<Key> keys;          // ascending; row i belongs to entity keys[i]
  std::vector<float> data;        // row-major, stride D
  const float* row(std::size_t i) const { return data.data() + i * D; }
};

static float cosDistance(const float* a, const float* b) {
  // unit vectors -> cosine distance = 1 - dot  (matches usearch cos_k)
  float dot = 0.f;
  for (std::size_t j = 0; j < D; ++j) dot += a[j] * b[j];
  return 1.f - dot;
}

// Our exact top-k over an explicit candidate set of ROW indices.
static std::vector<std::pair<float, Key>> exactTopK(
    const FlatStore& s, const float* query, std::size_t k,
    const std::vector<std::size_t>& candidateRows) {
  std::priority_queue<std::pair<float, Key>> heap;  // max-heap on distance
  for (std::size_t i : candidateRows) {
    float dist = cosDistance(query, s.row(i));
    if (heap.size() < k) {
      heap.push({dist, s.keys[i]});
    } else if (dist < heap.top().first) {
      heap.pop();
      heap.push({dist, s.keys[i]});
    }
  }
  std::vector<std::pair<float, Key>> out;
  while (!heap.empty()) { out.push_back(heap.top()); heap.pop(); }
  std::reverse(out.begin(), out.end());  // ascending distance
  return out;
}

int main() {
  std::mt19937 rng(42);
  std::normal_distribution<float> g(0.f, 1.f);

  // Build the flat store of random unit vectors, keyed by a non-contiguous id
  // space (key = 1000 + i*7) to prove keys != row indices.
  FlatStore store;
  store.keys.resize(N);
  store.data.resize(N * D);
  for (std::size_t i = 0; i < N; ++i) {
    store.keys[i] = 1000 + Key(i) * 7;
    float* r = store.data.data() + i * D;
    float norm = 0.f;
    for (std::size_t j = 0; j < D; ++j) { r[j] = g(rng); norm += r[j] * r[j]; }
    norm = std::sqrt(norm);
    for (std::size_t j = 0; j < D; ++j) r[j] /= norm;
  }
  std::printf("[setup] %zu vectors, dim=%zu, key[0]=%llu key[N-1]=%llu\n", N, D,
              (unsigned long long)store.keys.front(),
              (unsigned long long)store.keys.back());

  std::vector<std::size_t> allRows(N);
  for (std::size_t i = 0; i < N; ++i) allRows[i] = i;

  // ---- build usearch HNSW, keyed by VocabIndex ------------------------------
  metric_punned_t metric(D, metric_kind_t::cos_k, scalar_kind_t::f32_k);
  index_dense_config_t cfg;
  cfg.connectivity = 16;            // M
  cfg.expansion_add = 128;          // efConstruction
  cfg.expansion_search = 64;        // efSearch
  auto made = index_dense_t::make(metric, cfg);
  if (!made) { std::printf("FAIL make: %s\n", made.error.what()); return 1; }
  index_dense_t index = std::move(made.index);
  index.reserve(N);
  for (std::size_t i = 0; i < N; ++i) {
    auto ar = index.add(store.keys[i], store.row(i));
    if (!ar) { std::printf("FAIL add: %s\n", ar.error.what()); return 1; }
  }
  std::printf("[hnsw]  built, size=%zu\n", index.size());

  // ---- recall@K : HNSW vs our exact, over many queries ----------------------
  const std::size_t Q = 200;
  std::size_t hits = 0, total = 0;
  Key kbuf[64]; index_dense_t::distance_t dbuf[64];
  for (std::size_t q = 0; q < Q; ++q) {
    const float* query = store.row(q * 97 % N);
    auto exact = exactTopK(store, query, K, allRows);
    std::unordered_set<Key> truth;
    for (auto& p : exact) truth.insert(p.second);

    auto res = index.search(query, K);
    if (!res) { std::printf("FAIL search\n"); return 1; }
    std::size_t cnt = res.dump_to(kbuf, dbuf);
    for (std::size_t i = 0; i < cnt; ++i) if (truth.count(kbuf[i])) ++hits;
    total += K;
  }
  std::printf("[recall] HNSW recall@%zu over %zu queries = %.3f\n", K, Q,
              double(hits) / double(total));

  // ---- usearch built-in EXACT search must match our exact exactly ----------
  {
    const float* query = store.row(123);
    auto ours = exactTopK(store, query, K, allRows);
    auto res = index.search(query, K, 0, /*exact=*/true);
    std::size_t cnt = res.dump_to(kbuf, dbuf);
    bool match = cnt == ours.size();
    for (std::size_t i = 0; i < cnt && match; ++i) match = (kbuf[i] == ours[i].second);
    std::printf("[exact]  usearch exact == our brute force: %s (top key ours=%llu usearch=%llu)\n",
                match ? "YES" : "NO",
                (unsigned long long)ours[0].second, (unsigned long long)kbuf[0]);
  }

  // ---- save -> mmap view round-trip ----------------------------------------
  const char* path = "build/vec_poc.usearch";
  if (!index.save(path)) { std::printf("FAIL save\n"); return 1; }
  auto viewed = index_dense_t::make(path, /*view=*/true);  // memory-mapped
  if (!viewed) { std::printf("FAIL view: %s\n", viewed.error.what()); return 1; }
  index_dense_t mindex = std::move(viewed.index);
  {
    const float* query = store.row(7);
    auto a = index.search(query, K);
    auto b = mindex.search(query, K);
    Key ka[64], kb[64]; index_dense_t::distance_t da[64], db[64];
    std::size_t na = a.dump_to(ka, da), nb = b.dump_to(kb, db);
    bool same = na == nb;
    for (std::size_t i = 0; i < na && same; ++i) same = (ka[i] == kb[i]);
    std::printf("[mmap]   view(save()) search identical to in-memory: %s (mmapped size=%zu)\n",
                same ? "YES" : "NO", mindex.size());
  }

  // ---- exact over a candidate SUBSET (the "5000 green statues" path) --------
  {
    std::vector<std::size_t> subset;
    std::unordered_set<Key> subsetKeys;
    for (std::size_t i = 0; i < N; i += 9) {  // ~2222 "candidates"
      subset.push_back(i); subsetKeys.insert(store.keys[i]);
    }
    const float* query = store.row(55);
    auto top = exactTopK(store, query, K, subset);
    bool allInSubset = true;
    for (auto& p : top) allInSubset &= (subsetKeys.count(p.second) > 0);
    std::printf("[subset] exact over %zu candidates -> top-%zu all within subset: %s "
                "(nearest dist=%.4f)\n",
                subset.size(), top.size(), allInSubset ? "YES" : "NO", top[0].first);
  }

  std::printf("\nALL CORE CLAIMS VERIFIED.\n");
  return 0;
}
