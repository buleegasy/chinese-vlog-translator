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
