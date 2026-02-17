Hey there, TechDoc enthusiasts!


  It's been an incredibly productive few weeks for the TechDoc Genie team. We've just wrapped up Sprints 0 and 1,
  laying down the fundamental architecture and building out the core of our Retrieval-Augmented Generation (RAG)
  system. Let's dive into what we've achieved!

  ---

  Sprint 0: Laying the Groundwork – Building a Solid Foundation


  Our initial sprint was all about getting the basics right. We focused on establishing a robust development
  environment and setting up our data ingestion pipeline.

  What We Accomplished:


   * Project Setup: We kicked things off by structuring our GitHub repository, ensuring a clean and organized
     codebase from day one.
   * Dockerized Environment: A Docker development environment was established, making our setup consistent and
     reproducible for everyone on the team.
   * PostgreSQL Documentation Ingestion: The core task was to ingest and chunk PostgreSQL documentation. This
     involved developing a robust document loading mechanism capable of handling various formats.
   * Initial Vector Database: We successfully created an initial vector database, populated with over 500
     document chunks, ready for efficient retrieval.
   * Baseline Retrieval: We implemented a basic retrieval script that can now fetch semantically relevant chunks
     based on a given query.

  Key Metrics & Outcomes from Sprint 0:


   * Dependencies: All project dependencies are installed and functioning correctly.
   * Documentation Processed: Successfully processed 100+ pages of documentation, validating our ingestion
     pipeline.
   * Retrieval Relevance: Initial tests showed our retrieval mechanism returning relevant chunks for 5 key test
     queries.

  ---


  Sprint 1: Igniting the Core RAG Pipeline – From Retrieval to Generation

  With our foundation solid, Sprint 1 was dedicated to bringing our RAG system to life. This is where the magic
  of combining powerful LLMs with our specialized documentation really began to shine.

  What We Accomplished:


   * Full RAG Pipeline Implementation: We successfully integrated LangChain, enabling a seamless flow from user
     query -> document retrieval -> LLM generation -> coherent answer.
   * Citation System: A crucial component was the development of a citation manager and response formatter. Our
     RAG system now links generated answers directly to their source documents, enhancing trustworthiness and
     allowing users to dive deeper.
   * Prompt Engineering: We created multiple prompt templates (base, detailed, troubleshooting, code) to guide
     the LLM's responses, making them more grounded and relevant to technical queries.
   * Initial Evaluation Dataset: A 20-query evaluation dataset with ground truth answers was established,
     providing a quantifiable way to measure our system's performance.
   * Optimization Iteration: Through analysis of initial results, we began iterating on retrieval strategies and
     prompt engineering, documenting our experiments.
   * Robustness Improvements: We significantly enhanced the robustness of our document loading, ensuring proper
     HTML parsing, metadata handling, and clear logging for edge cases (like no files found).
   * Expanded Test Coverage: Implemented extensive unit tests for core modules, particularly DocumentLoader and
     query_interface.py, and introduced dedicated test files for various LLM providers (Claude, Gemini, Ollama,
     OpenAI). This boosted our overall code coverage to a healthy 69%.

  Key Metrics & Outcomes from Sprint 1:


   * Answer Coherence: Generated coherent answers with citations for all 20 test queries.
   * Answer Relevance (Manual): Achieved over 80% manual evaluation score for answer relevance in initial checks.
   * Response Latency: Initial cold query response latency is under 5 seconds.
   * Hallucination Rate: Focused on minimizing hallucinations through prompt engineering and source grounding.
   * Test Coverage: Increased overall code coverage to 69%, with all tests currently passing.

  ---

  Example Outputs & How We Measure Success

  We've been actively running the platform with different models and databases to observe real-world performance.

  Running the Platform with Different Models & Databases:


  We've successfully run the query_interface.py to interact with our RAG system. For instance:

   * Default Setup (PostgreSQL Vector Store, `phi3:mini` model):


    1     Your question: how do i make a table
    2     ANSWER:
    3     ────────────────────────────────────────────────────────────────────────────────
    4     To create a table in PostgreSQL, you can use the `CREATE TABLE` statement...
    5     ────────────────────────────────────────────────────────────────────────────────
    6     SOURCES (5):
    7     [1] [1] postgresql-16-A4.pdf, Page N/A
    8         # CREATE TABLE
    9         Section: CREATE TABLE
   10     ...
      This demonstrates the system's ability to fetch relevant information from the PostgreSQL PDF documentation
  and provide a structured SQL example.

   * Using `ubuntu_docs_pdf` Vector Store with `qwen2.5:1.5b` model:


    1     Your question: how do i contribute to ubuntu
    2     ANSWER:
    3     ────────────────────────────────────────────────────────────────────────────────
    4     To contribute to Ubuntu, you can start by following these steps:
    5     1. Pick up an existing GitHub Issue...
    6     ────────────────────────────────────────────────────────────────────────────────
    7     SOURCES (5):
    8     [1] [1] ubuntu.pdf, Page N/A
    9         We believe that everyone has something valuable to contribute...
   10     ...
      This highlights the flexibility to switch between different documentation sets and utilize alternative LLM
  models, all through simple command-line arguments.


   * Gemini Model Integration: We also tested integration with Gemini models:


    1     python src/agent/query_interface.py --provider gemini --model gemini-2.5-flash-lite
    2     Your question: difference between a left join and middle join
    3     ANSWER:
    4     ────────────────────────────────────────────────────────────────────────────────
    5     I don't have enough information
    6     ────────────────────────────────────────────────────────────────────────────────
    7     SOURCES (5):
    8     [1] [1] postgresql-16-A4.pdf, Page N/A
    9         of the joined table by inserting null values for the right-hand columns.
   10     ...
      This shows our system gracefully handling queries where the retrieved context might not contain enough
  information, as per our prompt engineering instructions.

  Metrics for Measuring Outcomes:


  Our "Success Metrics" are our guiding stars for evaluating the system's performance. For Sprint 1, we
  established baselines and criteria that will drive future optimization:


   * Answer Relevance: Manually scored (1-5 scale) on how well the answer addresses the user's question.
   * Groundedness: Manually scored (1-5 scale) on whether all claims in the answer are supported by provided
     sources.
   * Completeness: Manually scored (1-5 scale) on if the answer provides sufficient detail and fully addresses
     the query.
   * Citation Quality: Manually scored (1-5 scale) on the accuracy and appropriateness of the cited sources.
   * Response Latency: Measured to ensure a responsive user experience.
   * Hallucination Rate: A critical metric, aiming for zero hallucinations, where the LLM invents information not
     present in the context.
   * Code Coverage: Tracking the percentage of our codebase covered by automated tests to ensure reliability and
     maintainability.

  ---

  Looking Ahead to Sprint 2!


  With a solid RAG pipeline and an initial evaluation framework in place, we're excited to move into Sprint 2.
  Our focus will shift towards implementing the Model Context Protocol (MCP) server and integrating sophisticated
  tool-calling capabilities, paving the way for even more dynamic and powerful interactions with our technical
  documentation.

  Stay tuned for more updates on TechDoc Genie's journey!
