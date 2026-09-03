## 2024-05-18 - [Optimize Cosine Similarity]
**Learning:** Cloudflare BGE-M3 vector embeddings in this project are already L2-normalized.
**Action:** When calculating similarity between two normalized vectors, skip magnitude calculation and just use dot product. This yields a ~2.5x speedup for this function.

## 2024-05-18 - [Edge Isolate Memory Caching]
**Learning:** Next.js Edge Runtimes (like Cloudflare Workers) maintain global module-level variables across requests hitting the same isolate instance. We don't necessarily need Redis/KV for simple identical-request deduplication if instances handle multiple requests.
**Action:** Use a simple bounded `Map` for high-latency/costly API responses keyed by input to provide a "free" caching layer that dramatically reduces repeated costs and latency for identical queries without adding external dependencies.

## 2024-08-09 - [O(N) Top-K Search]
**Learning:** For small or medium datasets in memory where only the top K (e.g. 2) results are needed, mapping all results to a new array and then sorting is $O(N \log N)$ and allocates many temporary objects.
**Action:** Use a single-pass loop that maintains the top K elements, bringing the time complexity to $O(N)$ and reducing memory pressure significantly.

## 2024-11-20 - [Optimize Cosine Similarity with Loop Unrolling]
**Learning:** In Node.js/Edge environments, heavy numeric operations like dot products on large arrays (e.g., 1024-dimension embeddings) can be slightly bottlenecked by loop overhead and branching.
**Action:** Unroll tight inner loops (e.g., by 4x) for operations on large arrays to reduce conditional jumps. This provides a ~20% speedup in V8/JS engines without resorting to native modules.

## 2024-11-20 - [Float32Array vs Normal Array in Unrolled Loops]
**Learning:** In V8/Node.js, converting standard arrays to `Float32Array` for an unrolled cosine similarity loop surprisingly *degrades* performance by almost 2x. Standard arrays perform much faster with loop unrolling for dot products of this dimension (1024), likely due to V8's internal optimizations (e.g. PACKED_DOUBLE_ELEMENTS vs TypedArray bounds checking and cast overhead).
**Action:** Do not blindly cast JSON parsed arrays to Typed Arrays (like `Float32Array`) for math operations without profiling first; in this codebase's specific unrolled loop, stick to native arrays.

## 2024-11-20 - [Loop Unrolling Calibration for 1024-dim vectors]
**Learning:** For extremely tight and repetitive CPU loops, like dot product similarity over 1024-dimension vectors in JavaScript Edge Runtimes or Node.js, there is a sweet spot for loop unrolling. Unrolling 4x provided a big jump over no unrolling, but unrolling 8x yields maximum performance (~15-20% faster than 4x). However, pushing to 16x *decreases* performance slightly due to instruction cache limits or register pressure.
**Action:** When manually unrolling numerical loops on array iterations (like 1024 dimensions), 8x unrolling is generally the optimal balance point in V8 for maximizing CPU throughput without exhausting the instruction cache.

## 2024-11-20 - [Map LRU Caching]
**Learning:** In JavaScript, the `Map` object preserves key insertion order. A simple FIFO bounded cache using `Map.keys().next().value` drops the oldest inserted item when full, regardless of how often it is accessed.
**Action:** Always delete and re-insert items into the `Map` when a cache hit occurs. This simple $O(1)$ trick pushes the item to the end of the insertion order, instantly converting a naive FIFO cache into a true Least Recently Used (LRU) cache, significantly improving cache hit rates for hot keys with zero extra memory overhead.

## 2024-11-20 - [Module-Level JSON Array Caching]
**Learning:** Next.js Edge APIs parsing/resolving large imported JSON arrays (like a 2.7MB RAG corpus) inside the request handler incurs per-request execution overhead. Because Edge environments (like Cloudflare Workers) maintain global module-level variables across requests hitting the same isolate instance, this work only needs to be done once per instance.
**Action:** Always resolve and cache large, static JSON datasets at the module level (outside the `POST`/`GET` handlers) to prevent repetitive evaluation and object allocations, drastically speeding up initialization per request.

## 2026-07-28 - [Request Coalescing in Edge Runtimes]
**Learning:** High-latency API endpoints (like RAG translations involving external embeddings and LLMs) can easily suffer from cache stampedes if multiple identical requests arrive simultaneously before the first one completes and populates the cache.
**Action:** Use a module-level Map of pending Promises (Request Coalescing) in Edge Runtimes to serve identical concurrent requests with a single downstream API call, drastically reducing cost and rate-limit risks without adding complex lock mechanisms.

## 2023-10-27 - [RAG Exact Match Short-Circuit]
**Learning:** RAG systems usually always incur a vector embedding and LLM generation cost, but when a predefined dataset (corpus) is small enough to fit in edge memory, exact matches can bypass the entire pipeline (network embedding, dense vector math, LLM API call).
**Action:** When a static baseline corpus exists in a RAG system and strings are short (like translation sentences), always implement a hash map (Map or Record) to check for O(1) exact matches before doing O(N) cosine similarities or calling external embedding APIs.
## 2024-07-30 - [Optimize Top-K Search with Index Tracking]
**Learning:** Even within an $O(N)$ single-pass loop designed to find top K elements, creating new object literals (e.g., `{ input, output, similarity }`) on every swap or matching assignment increases garbage collection pressure, especially when iterating over large arrays in JavaScript.
**Action:** When tracking top elements in hot loops, use primitive variables to track indices (`topIdx`) and comparison values (`topSim`), and only allocate the final object literals after the loop finishes. This approach minimizes memory allocation overhead without sacrificing readability.
## 2025-01-20 - [Cache Pollution by Static Maps]
**Learning:** In Next.js Edge APIs, populating an LRU cache with results that are already resolvable via an O(1) static Map causes severe cache pollution. This prematurely evicts expensive, dynamically computed entries (like LLM generations) when high-volume static exact matches occur.
**Action:** Always check O(1) static lookup maps *before* the LRU cache, and never insert static map hits into the LRU cache.
## 2024-11-21 - [Map Double Lookup Elimination]
**Learning:** Using `Map.has(key)` followed by `Map.get(key)` in hot paths or API handlers executes the hash function and table traversal twice.
**Action:** Always use `const val = Map.get(key); if (val !== undefined)` (or truthiness check if applicable) to halve the lookup overhead for caches and static dictionaries.
## 2024-05-19 - [Avoid Object Property Lookups in Hot Math Loops]
**Learning:** In Edge environments and V8, accessing object properties (e.g. `item.embedding`) inside hot, tight mathematical loops (like iterating over thousands of vectors for cosine similarity) adds significant overhead compared to direct array access.
**Action:** Extract nested arrays/properties from objects into flat, parallel arrays at module initialization time. Use these parallel arrays in hot math loops to bypass object property lookups and improve performance.
## 2026-08-05 - [Module Initialization Loop Fusion]
**Learning:** Iterating over a very large JSON array (like a 2.7MB RAG corpus) multiple times during module initialization increases cold-start latency and overhead.
**Action:** Merge multiple initialization steps (like building parallel arrays and O(1) exact match maps) into a single O(N) loop pass to minimize traversal overhead.

## 2026-08-05 - [Failed Optimization: Top-K Vector Search Early Termination]
**Learning:** Attempting to short-circuit a Top-K search loop (where K > 1) early when a single perfect match (similarity > 0.99) is found is fundamentally flawed. It breaks the Top-K constraint because the second-best result (`top2Idx`) might be left completely uninitialized or assigned a wildly incorrect local maximum, leading to crashes or severely degraded RAG output.
**Action:** Do not blindly apply early loop termination in search functions that need to return more than one top result. Ensure that all required Top-K elements meet early-termination criteria before breaking.

## 2026-08-06 - [Top-K Branch Check Reduction]
**Learning:** In a Top-K search loop (like iterating through thousands of vectors to find the top 2 matches), checking the highest rank condition (`similarity > top1Sim`) first is inefficient because >99% of elements won't even qualify for the lowest rank. If the first check falls through, the engine must evaluate the second branch (`similarity > top2Sim`) as well.
**Action:** Always check the lowest threshold (`similarity > topKSim`) first. If it fails, you can skip all other checks, effectively halving the number of conditional branches evaluated in hot O(N) loops.
## 2026-08-07 - [Module-Level Array Pre-allocation]
**Learning:** In JavaScript/V8, initializing an empty array (`[]`) and dynamically `.push()`ing thousands of elements into it forces the engine to repeatedly reallocate memory and resize the underlying array structure.
**Action:** When the final size of an array is known in advance (like iterating over a static dataset during module initialization), always pre-allocate the array using `new Array(size)` and assign values by index (`arr[i] = val`). This avoids dynamic resizing overhead and reduces module cold-start latency.
## 2026-08-08 - [Loop Invariant Code Motion]
**Learning:** In tight, unrolled loops (like cosine similarity calculations), placing arithmetic expressions like `len - 8` in the loop condition causes the JS engine to re-evaluate the subtraction on every single iteration.
**Action:** Always hoist static math calculations out of loop conditions (`const limit = len - 8;`) into variables. This eliminates redundant operations and maximizes throughput in hot math paths.
## 2026-08-11 - [Instruction-Level Parallelism in Unrolled Loops]
**Learning:** In tight, unrolled mathematical loops (like cosine similarity), using a single accumulator (e.g., `sum += ...`) forces the CPU to wait for the previous addition to complete before starting the next one, creating a data dependency chain.
**Action:** Use multiple independent accumulators (e.g., `sum0`, `sum1`, `sum2`, `sum3`) in unrolled loops. This breaks the data dependency chain, allowing V8 and the CPU's superscalar architecture to execute the floating-point additions concurrently, yielding a measurable speedup in hot O(N) paths.
## 2026-08-12 - [Eager Cancellation of Zombie Network Requests]
**Learning:** In Edge environments and API routes, merely rejecting a local Promise on timeout leaves the underlying network request (`fetch` or SDK calls like Gemini) running in the background. These "zombie" requests silently consume memory, CPU, bandwidth, and importantly, third-party API rate limits and costs.
**Action:** Use an `AbortController` in timeout wrappers to eagerly cancel underlying network requests (by passing the `signal`) when a timeout occurs, rather than just rejecting the wrapping promise.

## 2026-08-13 - [Generous Timeouts for AI Endpoints]
**Learning:** Setting overly aggressive timeouts (like 5 seconds) for external 3rd-party AI endpoints (e.g., Cloudflare embedding APIs) leads to premature network abortion failures under normal AI processing variance and load.
**Action:** When setting request timeouts for external 3rd-party AI endpoints, use generous timeouts (like 10 seconds) to accommodate processing spikes while still providing an upper bound for zombie connection cancellation.
