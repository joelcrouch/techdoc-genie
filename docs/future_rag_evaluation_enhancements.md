# Future Enhancements: Extending RAG Evaluation for Multiple Chunking Strategies

This document outlines a plan to enhance the `evals/rag_evaluation.py` script to allow for systematic comparison of the RAG pipeline's performance across various document chunking strategies. This will move beyond the current single-strategy evaluation to provide a more comprehensive view of how chunking parameters impact answer quality.

## Objective

To enable `evals/rag_evaluation.py` to:
1.  Read a declarative configuration of multiple chunking strategies (e.g., different chunk sizes, overlaps, and methods like recursive or semantic).
2.  Execute the RAG evaluation process (vector store creation, LLM querying, semantic similarity scoring) for each defined strategy.
3.  Aggregate and present the evaluation results, allowing for easy comparison between chunking approaches.

## Proposed Approach: Declarative Configuration

Instead of hardcoding chunking parameters, a separate configuration file (e.g., `evals/chunking_configs.json` or `evals/chunking_configs.yaml`) will define the strategies to be evaluated. This makes the evaluation flexible and easy to extend without modifying the core script logic.

### 1. Define the Declarative Configuration File Structure

The configuration file (e.g., `evals/evaluation_configs.json`) could look like this:

```json
{
  "document_paths": {
    "postgresql": "/home/dell-linux-dev3/Projects/techdoc-genie/data/raw/postgresql/postgresql-16-A4.pdf",
    "ubuntu": "/path/to/ubuntu/docs.pdf"
  },
  "test_queries_path": "evals/test_queries.json",
  "ollama_config": {
    "llm_provider_type": "ollama",
    "model_id": "phi3:mini"
  },
  "chunking_strategies": [
    {
      "name": "Recursive_256_Overlap_30",
      "chunk_size": 256,
      "chunk_overlap": 30,
      "method": "chunk_recursive"
    },
    {
      "name": "Recursive_512_Overlap_50",
      "chunk_size": 512,
      "chunk_overlap": 50,
      "method": "chunk_recursive"
    },
    {
      "name": "Semantic_512_Overlap_50",
      "chunk_size": 512,
      "chunk_overlap": 50,
      "method": "chunk_semantic"
    },
    {
      "name": "Recursive_1024_Overlap_100",
      "chunk_size": 1024,
      "chunk_overlap": 100,
      "method": "chunk_recursive"
    }
  ]
}
```

### 2. Steps for Modification in `evals/rag_evaluation.py`

1.  **Load Configuration**:
    *   Add logic to load the `evaluation_configs.json` file at the beginning of `run_rag_evaluation`.
    *   Replace hardcoded values for `PDF_PATH` and `CHUNKING_CONFIG` with values from the loaded configuration.
    *   Update `TEST_QUERIES_PATH` and `llm_provider_type`/`model_id` if they become configurable.

2.  **Iterate Through Document Paths**:
    *   The `rag_evaluation.py` script currently processes a single PDF. It should be modified to iterate through the `document_paths` defined in the config. For each document:
        *   Load the source document(s).
        *   Initialize the `HuggingFaceEmbeddingProvider` for document embedding.

3.  **Iterate Through Chunking Strategies**:
    *   Introduce a loop that iterates through each `strategy` defined in `chunking_strategies` from the configuration file.
    *   Inside this loop, for each strategy:
        *   Create a `DocumentChunker` instance with the specified `chunk_size` and `chunk_overlap`.
        *   Call the appropriate chunking `method` (e.g., `chunk_recursive`, `chunk_semantic`) on the loaded `source_documents`.
        *   Create a new in-memory `VectorStore` using the generated chunks.
        *   Initialize the `RAGChain` with this new `VectorStore` and the LLM configuration (e.g., `ollama_config`).

4.  **Execute Query Evaluation**:
    *   The existing logic for running test queries through the `RAGChain`, calculating semantic similarity, and storing results will be encapsulated within these loops.
    *   Ensure that the `evaluation_results` list collects data tagged with the current `document` being evaluated and the `chunking_strategy` used.

5.  **Enhance `display_and_save_results`**:
    *   Modify the `display_and_save_results` function to group and present results in a clear comparative manner, possibly using nested tables or a more elaborate summary that highlights performance across different strategies and documents.
    *   The saved JSON output should clearly distinguish results by document and chunking strategy.

### 3. Considerations

*   **Error Handling**: Robust error handling should be in place for file loading, LLM calls, and embedding generation.
*   **Performance**: Evaluating many strategies across multiple documents will be computationally intensive and time-consuming. Considerations for parallelization or running subsets might be necessary.
*   **Ground Truth**: Ensure `test_queries.json` contains relevant `ground_truth` answers for the documents being evaluated.
*   **Logging**: Enhance logging to provide clear progress updates and debug information during long evaluation runs.
*   **Command-line Arguments**: Consider adding optional command-line arguments to `rag_evaluation.py` to specify the path to the configuration file or to filter which documents/strategies to run.
