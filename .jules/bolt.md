## 2024-07-18 - Cloudflare BGE-M3 Vector Normalization
**Learning:** Cloudflare BGE-M3 vector embeddings in this codebase are already L2-normalized (length/norm is 1). Therefore, cosine similarity calculations don't need to compute the norms of the vectors, saving significant math overhead.
**Action:** Replace full cosine similarity implementations (which calculate dot product divided by norms) with a simple dot product for these vectors.
