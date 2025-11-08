# Analysis of Concurrent Cache Race Condition in QLever

## 1. The Problem

When multiple SPARQL queries sharing a common sub-operation are executed concurrently, a race condition can occur in the `ConcurrentCache`, leading to a `std::runtime_error` with the message "Trying to insert a cache key which was already present." This is often followed by a `WaitedForResultWhichThenFailedException` in other threads that were waiting for the result.

## 2. The Root Cause: Conflicting Caching Strategies

The bug is caused by the interaction of two different caching mechanisms that can be triggered for the *same cache key* by different types of queries.

- **Path A: Full Materialization (`computeOnce`)**: Queries like `COUNT` or any query that is explicitly pinned must compute their entire result set upfront. This process uses the `ConcurrentCache::computeOnce` method, which:
    1.  Checks if a result is in the cache.
    2.  If not, it adds a placeholder to an internal `_inProgress` map.
    3.  **Crucially, it releases the main lock** to perform the long-running computation.
    4.  After computation, it re-acquires the lock and attempts to insert the result.

- **Path B: Lazy Caching (`tryInsertIfNotPresent`)**: Queries like `ASK` or simple `SELECT` statements can be processed in a lazy, streaming fashion. This is handled by the `Result::cacheDuringConsumption` mechanism, which:
    1.  Aggregates the streamed chunks of a result in the background.
    2.  When the stream is fully consumed, it calls `ConcurrentCache::tryInsertIfNotPresent`.
    3.  This function acquires the main lock but **does not check the `_inProgress` map**. It only checks if a final result is in the cache.

## 3. The Race Condition Scenario

The race condition is triggered as follows:

1.  A slow, fully-materialized query (e.g., `COUNT`) starts, taking **Path A**. It marks its cache key as "in progress" and begins computing, releasing the lock.
2.  A fast, lazy query (e.g., `ASK`) for the same underlying operation starts concurrently, taking **Path B**.
3.  The lazy query finishes first. Its `cacheDuringConsumption` logic successfully inserts the final result into the cache via `tryInsertIfNotPresent` because the lock is free and the final result slot is empty.
4.  The slow query finally finishes its computation. It re-acquires the lock and attempts to insert its result, but the cache key is now already present.
5.  **CRASH**: The cache throws the "Trying to insert a cache key which was already present" exception.

## 4. The Proposed Fix

The fix must be implemented inside `ConcurrentCache` to ensure atomicity and encapsulation. The solution involves:

1.  **Preventing the Crash**: Modify `moveFromInProgressToCache` to re-check if the key exists in the cache *after* acquiring the lock and *before* attempting to insert. If the key is present, the new result is discarded.
2.  **Ensuring Consistency**: The `moveFromInProgressToCache` function must return the value that is ultimately present in the cache (either the one it just inserted or the one that was already there). This "winning" result is then propagated to all waiting threads, ensuring all clients have a consistent view.

## 5. Detailed Code Path Analysis

The two conflicting caching paths originate in `Operation.cpp` and `Result.cpp`.

### Path A: Full Materialization in `Operation.cpp`

The `Operation::getResult` method is the entry point for executing a query operation. For queries that require full materialization (like `COUNT`), it calls `computeOnce` on the `ConcurrentCache`.

```cpp
// From src/engine/Operation.cpp
std::shared_ptr<const Result> Operation::getResult(
    bool isRoot, ComputationMode computationMode) {
    // ...
    auto result = [&]() {
      auto compute = [&](auto&&... args) {
        if (!canResultBeCached()) {
          return cache.computeButDontStore(AD_FWD(args)...);
        }
        return pinResult ? cache.computeOncePinned(AD_FWD(args)...)
                         : cache.computeOnce(AD_FWD(args)...);
      };
      return compute(cacheKey, cacheSetup, onlyReadFromCache, suitedForCache);
    }();
    // ...
}
```

This path marks the cache key as "in progress" and then releases the lock to perform the computation, creating the window for the race condition.

### Path B: Lazy Caching in `Result.cpp`

For lazy, streaming results (like `ASK` queries), the `Operation::runComputationAndPrepareForCache` method sets up a callback that uses `Result::cacheDuringConsumption`.

```cpp
// From src/engine/Operation.cpp
CacheValue Operation::runComputationAndPrepareForCache(
    const ad_utility::Timer& timer, ComputationMode computationMode,
    const QueryCacheKey& cacheKey, bool pinned, bool isRoot) {
    // ...
    result.cacheDuringConsumption(
        // ...
        [&cache, cacheKey](Result aggregatedResult) {
          // ...
          cache.tryInsertIfNotPresent(
              false, cacheKey,
              std::make_shared<CacheValue>(std::move(aggregatedResult),
                                           std::move(copy)));
        });
    // ...
}
```

The `cacheDuringConsumption` method in `Result.cpp` then calls `tryInsertIfNotPresent` on the cache when the lazy stream is fully consumed.

```cpp
// From src/engine/Result.cpp
void Result::cacheDuringConsumption(
    // ...
    std::function<void(Result)> storeInCache) {
  AD_CONTRACT_CHECK(!isFullyMaterialized());
  data_.emplace<GenContainer>(ad_utility::wrapGeneratorWithCache(
      idTables(),
      // ...
      [storeInCache = std::move(storeInCache),
       sortedBy = sortedBy_](IdTableVocabPair pair) mutable {
        storeInCache(
            Result{std::move(pair.idTable_), std::move(sortedBy),
                   SharedLocalVocabWrapper{std::move(pair.localVocab_)}});
      }));
}
```

This `tryInsertIfNotPresent` call is what "wins" the race, as it doesn't check for "in progress" keys.

## 5. Code Proof of Conflicting Strategies

The distinction between lazy and fully materialized results is confirmed by the source code.

### `ASK` Queries: Lazy Evaluation

The implementation for `ASK` queries in `src/engine/ExportQueryExecutionTrees.cpp` demonstrates lazy evaluation. The `getResultForAsk` function iterates through result chunks and stops as soon as it finds a single result, avoiding full materialization.

```cpp
// From src/engine/ExportQueryExecutionTrees.cpp
bool getResultForAsk(const std::shared_ptr<const Result>& result) {
  if (result->isFullyMaterialized()) {
    return !result->idTable().empty();
  } else {
    return ql::ranges::any_of(result->idTables(), [](const auto& pair) {
      return !pair.idTable_.empty();
    });
  }
}
```

### `COUNT` Queries: Full Materialization

Conversely, `COUNT` operations, as seen in `src/engine/CountAvailablePredicates.cpp`, always compute their full result set into an `IdTable` before returning. The `requestLaziness` parameter in its `computeResult` method is explicitly ignored.

```cpp
// From src/engine/CountAvailablePredicates.cpp
Result CountAvailablePredicates::computeResult(
    [[maybe_unused]] bool requestLaziness) {
  AD_LOG_DEBUG << "CountAvailablePredicates result computation..." << std::endl;
  IdTable idTable{getExecutionContext()->getAllocator()};
  idTable.setNumColumns(2);

  // ... logic to fully populate idTable ...

  return {std::move(idTable), resultSortedOn(), LocalVocab{}};
}
```
