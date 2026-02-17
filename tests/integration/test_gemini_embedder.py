# import os
# import pytest
# from dotenv import load_dotenv
# from google import genai
# from google.genai import types
# # from google.api_core.exceptions import InvalidArgument # Or other specific exceptions

# def test_gemini_embedder_api_failure():
#     load_dotenv()
#     api_key = os.getenv("GEMINI_EMBEDDING_KEY")

#     if not api_key: # This means it's None or an empty string
#         # If the API key is truly missing, the client might not even be instantiable
#         # or the embed_content call will fail.
#         # We can simulate the expected behavior of a missing key causing an error.
#         expected_exception = Exception # Use a broad exception, can be refined to google.api_core.exceptions.GoogleAPIError
#         with pytest.raises(expected_exception):
#             client = genai.Client(api_key=api_key)
#             text = "Test text."
#             client.models.embed_content(
#                 model="gemini-embedding-001",
#                 contents=text,
#                 config=types.EmbedContentConfig(task_type="retrieval_document"),
#             )
#     else: # If an API key string (like "DELETED") is present, expect API call to fail
#         # This branch handles the case where an invalid key is provided (e.g., "DELETED")
#         # We expect the actual API call to raise an exception.
#         expected_exception = Exception # Refine this to a more specific API error later
#         with pytest.raises(expected_exception):
#             client = genai.Client(api_key=api_key)
#             text = "Test text."
#             client.models.embed_content(
#                 model="gemini-embedding-001",
#                 contents=text,
#                 config=types.EmbedContentConfig(task_type="retrieval_document"),
#             )

