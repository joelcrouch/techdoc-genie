TechDoc-Genie: Gemini Embedding & Multi-DB Experiment To-Do List
1️⃣ Gemini Setup

 - [✅ ] Get a Gemini API key (store securely in .env or secret manager).

 - [✅ ] Confirm which Gemini embedding model you’ll use (gemini-embedding-001) and its token limits.

 - [✅ ] Write a small test script to embed a single piece of text to confirm your API key works.

**NOTE:  CHANGE TO HUGGING FACE SENTENCE TRANSFORMER, BECAUSE ITS FREE.**

<!-- 2️⃣ Implement Gemini Embedder

 - [ ] Create a GeminiEmbeddingProvider class in src/ingestion/providers/.

 - [ ] Follow the same interface as your BaseEmbedder.

 - [ ] Implement embed_documents(texts: List[str]) -> List[List[float]] and embed_query(query: str) -> List[float].

 - [ ] Implement a mock embedder for local testing without API costs.

 - [ ] Write pytest tests for both real and mock Gemini embedder. -->

2️⃣ Implement huggingface Embedder

 - [ ✅] Create a Ghuggingface class in src/ingestion/providers/.

 - [✅ ] Follow the same interface as your BaseEmbedder.

 - [✅ ] Implement embed_documents(texts: List[str]) -> List[List[float]] and embed_query(query: str) -> List[float].

 - [✅ ] Implement a mock embedder for local testing without API costs.

 - [✅ ] Write pytest tests for both real and mock Gemini embedder.

3️⃣ Vector Store Setup

 - [✅ ] Choose your vector store backend (FAISS, Chroma, etc.).

 - [ ✅] Implement a VectorStore class if not already done.

 - [✅ ] Make it configurable so you can point to multiple DB instances.
=> could be better with a vecotrSTore mngr. currently make a new instance like this: ```
vs1 = VectorStore(persist_path="/path/to/db1", embedder=embedder)
  vs2 = VectorStore(persist_path="/path/to/db2", embedder=embedder)```
 - [✅ ] Ensure saving/loading works locally.

4️⃣ Multiple Vector Store Instances

 - [ ] Dockerize your vector store so you can spin up multiple isolated instances.

 - [ ] Implement a naming scheme for multiple instances (e.g., vectorstore_chunk512_overlap50).

 - [ ] Add a helper script to create and populate new DB instances with specific chunking strategies.

5️⃣ Long Ingestion Jobs

 - [ ] Chunk your documents with different strategies (size, overlap).

 - [ ] Embed each chunk using Gemini, respecting free-tier limits.

 - [ ] Populate each vector store DB instance.

 - [ ] Monitor rate limits / API costs during ingestion.

6️⃣ Query Experiment Script

 Implement a script to:

- [ ] Take a list of test queries.

- [ ] Embed each query using Gemini.

- [ ] Run similarity search across all DB instances.

- [ ] Collect results and compute metrics (# of chunks retrieved, relevance, etc.).

7️⃣ Collate & Analyze

 - [ ] Programmatically compare results across different DB instances / chunking strategies.

 - [ ] Identify the best-performing strategy for current query set.

 - [ ] Optionally, store old DBs for future testing as queries grow.

8️⃣ Optional / Future

 - [ ] Automate DB cleanup / archiving for old strategies.

 - [ ] Add monitoring/logging for embedding costs, query latency, and DB usage.

 - [ ] Integrate your best DB strategy into your main app pipeline.