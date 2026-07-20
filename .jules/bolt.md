## 2024-05-18 - [Optimize Cosine Similarity]
**Learning:** Cloudflare BGE-M3 vector embeddings in this project are already L2-normalized.
**Action:** When calculating similarity between two normalized vectors, skip magnitude calculation and just use dot product. This yields a ~2.5x speedup for this function.

## 2024-05-18 - [Edge Isolate Memory Caching]
**Learning:** Next.js Edge Runtimes (like Cloudflare Workers) maintain global module-level variables across requests hitting the same isolate instance. We don't necessarily need Redis/KV for simple identical-request deduplication if instances handle multiple requests.
**Action:** Use a simple bounded `Map` for high-latency/costly API responses keyed by input to provide a "free" caching layer that dramatically reduces repeated costs and latency for identical queries without adding external dependencies.
