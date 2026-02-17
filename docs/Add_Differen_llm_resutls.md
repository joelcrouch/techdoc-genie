
## ph3:mini

## qwen2.5:1.1.5b

## Add Gemini

I added an API key from Google for their service.   I am using gemini-2.5-flash-lite.  It has a 1000 requerst per day limit.  BLARRRGH! I neglected to add in rate limiting logic (request per minute)  to the calling logic.  It bound to fail. Anyways, Ill see hwo far it gets.  Maybe the latency will keep it from crashing spectacularly.  (the latency form making an aswer from the request/(query+vectordb answers)).  For some reason I have not figured out yet there is a very dramatic delay between submitting a request and getting an answer. Oddly enough, there almost appears to be a warm-up period, b/c I can watch the latency decrease the further along into the 45 question test bed the LLM gets.  And there are clear differences in latency between qwen and ph3:mini. Whoops. Im getting away from the task.  I ran one query here:

```
ython src/agent/query_interface.py --provider gemini --model gemini-2.5-flash-lite
2026-02-15 11:59:06,048 - __main__ - INFO - Loading vector store...
2026-02-15 11:59:10,994 - src.ingestion.embedder - INFO - Using HuggingFaceEmbedder with model all-MiniLM-L6-v2
2026-02-15 11:59:11,346 - src.retrieval.vector_store - INFO - Vector store loaded from data/vector_store/vectorstore_chunk512_overlap50/vectorstore_chunk512_overlap50
2026-02-15 11:59:11,347 - __main__ - INFO - Vector store 'vectorstore_chunk512_overlap50' loaded successfully from data/vector_store/vectorstore_chunk512_overlap50.
2026-02-15 11:59:11,347 - __main__ - INFO - Initializing RAG chain with provider 'gemini' and model 'gemini-2.5-flash-lite'...
2026-02-15 11:59:11,348 - src.agent.rag_chain - INFO - Initialized RAG chain with Gemini model: gemini-2.5-flash-lite
2026-02-15 11:59:11,348 - src.agent.rag_chain - INFO - RAG chain fully initialized with provider 'gemini' and retriever.

================================================================================
Technical Documentation Assistant
Provider: gemini, Model: gemini-2.5-flash-lite, Prompt: base
Vector Store: vectorstore_chunk512_overlap50
Type 'exit' or 'quit' to end.
================================================================================


Your question: what is the difference between a left join and a right join?

🔍 Searching documentation and generating answer...
2026-02-15 11:59:37,989 - src.agent.rag_chain - INFO - Processing query: what is the difference between a left join and a right join?
2026-02-15 11:59:40,788 - src.agent.rag_chain - INFO - Generated response with 5 sources.

================================================================================
ANSWER:
────────────────────────────────────────────────────────────────────────────────
LEFT OUTER JOIN returns all rows in the qualified Cartesian product that pass its join condition, plus one copy of each row in the left-hand table for which there was no right-hand row that passed the join condition. This left-hand row is extended to the full width of the joined table by inserting null values for the right-hand columns.

RIGHT OUTER JOIN returns all the joined rows, plus one row for each unmatched right-hand row (extended with nulls on the left). This is a notational convenience, as it can be converted to a LEFT OUTER JOIN by switching the left and right tables.

The words INNER and OUTER are optional in all forms. INNER is the default; LEFT, RIGHT, and FULL imply an outer join. The join condition is specified in the ON or USING clause, or implicitly by the word NATURAL.

================================================================================
SOURCES (5):
────────────────────────────────────────────────────────────────────────────────

[1] [1] postgresql-16-A4.pdf, Page N/A
    REFERENCES right);
It is typically used like this:
SELECT right.* from right JOIN one_to_many ON (right.id =
one_to_many.right)
WHERE one_to_many.left = item;
This will return all the items in the rig...

[2] [2] postgresql-16-A4.pdf, Page N/A
    A joined table is a table derived from two other (real or derived) tables according to the rules of the
particular join type. Inner, outer, and cross-joins are available. The general syntax of a joine...

[3] [3] postgresql-16-A4.pdf, Page N/A
    of the joined table by inserting null values for the right-hand columns. Note that only the JOIN
clause's own condition is considered while deciding which rows have matches. Outer conditions
are appli...

[4] [4] postgresql-16-A4.pdf, Page N/A
    binds more tightly than the commas separating FROM-list items. All the JOIN options are just a
notational convenience, since they do nothing you couldn't do with plain FROM and WHERE.
LEFT OUTER JOIN ...

[5] [5] postgresql-16-A4.pdf, Page N/A
    The words INNER and OUTER are optional in all forms. INNER is the default; LEFT, RIGHT,
and FULL imply an outer join.
The join condition is specified in the ON or USING clause, or implicitly by the wo...

================================================================================


Your question: 

```

## Claude

## openAi


## Results


## Conclusion

Hypothesis:  I think there will be some chunking strategy (strategy + size + overlap)  that gives the best performance compared to memory footprint.  If we can get the same performcance out of 1024  size as we can with 512, perhaps going with 1024 is optimal for memory usage. In this case we are really talking about RAM and Vram, once the db is written to disk, who cares?  The rate limiting step will probably be I/O, and all the different sizes overlaps experimentewd with show similar performance.
Optimzing size of k might allow for a quicker knn search and having the citation manager return k citations might also save some time.  Maybe trun off the citation manager, once the use is statisfied with the results-ie for the end user. (not the ML practitioner using this)

### Next steps  
Use a different embedder, maybe test with different gpus (rent one on the cloud or somehthing)  See if there are measurable differneces that are meaningful with a 'better' gpu.
