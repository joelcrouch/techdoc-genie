# MCP Protocol Reference — TechDoc Genie

## Overview

The Model Context Protocol (MCP) is a standardized JSON-RPC 2.0 protocol that
allows LLMs to call external tools and access resources through a well-defined
interface. TechDoc Genie exposes its document retrieval capabilities as an MCP
server so that any MCP-aware LLM can search technical documentation without
embedding retrieval logic inside the model itself.

**Server base URL (development)**: `http://localhost:8001`
**Protocol version**: MCP v1.0
**Transport**: HTTP/1.1, `POST /mcp`
**Serialization**: JSON (UTF-8)

---

## Protocol Fundamentals

### JSON-RPC 2.0 Envelope

Every request and response uses the JSON-RPC 2.0 envelope.

#### Request

```json
{
  "jsonrpc": "2.0",
  "method": "<namespace>/<action>",
  "params": { ... },
  "id": "caller-chosen-string-or-integer"
}
```

| Field      | Type             | Required | Description                                      |
|------------|------------------|----------|--------------------------------------------------|
| `jsonrpc`  | `"2.0"` (string) | Yes      | Protocol version — must be exactly `"2.0"`       |
| `method`   | string           | Yes      | Namespaced method name (see [Methods](#methods)) |
| `params`   | object           | No       | Method-specific parameters                       |
| `id`       | string \| int    | No       | Caller-supplied correlation ID; echoed in reply  |

#### Success Response

```json
{
  "jsonrpc": "2.0",
  "result": { ... },
  "id": "caller-chosen-string-or-integer"
}
```

#### Error Response

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32600,
    "message": "Invalid Request"
  },
  "id": "caller-chosen-string-or-integer"
}
```

Standard JSON-RPC error codes used by this server:

| Code    | Meaning          | When used                              |
|---------|------------------|----------------------------------------|
| -32600  | Invalid Request  | Malformed envelope or missing fields   |
| -32601  | Method Not Found | Unknown `method` value                 |
| -32602  | Invalid Params   | Bad or missing tool arguments          |
| -32603  | Internal Error   | Unhandled server-side exception        |

---

## Transport & Endpoint

```
POST /mcp
Content-Type: application/json
Authorization: Bearer <api_key>   (only required when MCP_AUTH_REQUIRED=true)
```

A health-check endpoint is also available:

```
GET /
```

Returns:

```json
{
  "status": "healthy",
  "service": "TechDoc Genie MCP Server",
  "version": "0.1.0",
  "protocol": "MCP v1.0"
}
```

---

## Methods

### `tools/list`

Return the catalogue of callable tools that this server exposes.

**Request**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": "list-1"
}
```

No `params` field is required.

**Response**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "search_documentation",
        "description": "Search technical documentation using semantic similarity. Returns relevant document chunks.",
        "input_schema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search query or question"
            },
            "max_results": {
              "type": "integer",
              "description": "Maximum number of results to return (default: 5)",
              "default": 5
            },
            "doc_type": {
              "type": "string",
              "description": "Optional filter for document type",
              "enum": ["postgresql", "ubuntu", "all"]
            }
          },
          "required": ["query"]
        }
      },
      {
        "name": "get_document",
        "description": "Retrieve a specific document by ID or path.",
        "input_schema": {
          "type": "object",
          "properties": {
            "document_id": {
              "type": "string",
              "description": "The unique identifier or path of the document"
            }
          },
          "required": ["document_id"]
        }
      }
    ]
  },
  "id": "list-1"
}
```

---

### `tools/call`

Invoke a tool by name with the supplied arguments.

**Request**

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search_documentation",
    "arguments": {
      "query": "How do I create an index in PostgreSQL?",
      "max_results": 5,
      "doc_type": "postgresql"
    }
  },
  "id": "call-1"
}
```

| `params` field | Type   | Required | Description                  |
|----------------|--------|----------|------------------------------|
| `name`         | string | Yes      | Tool name from `tools/list`  |
| `arguments`    | object | Yes      | Tool-specific arguments      |

**Success Response**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "CREATE INDEX creates an index on the specified column(s)...",
        "metadata": {
          "source": "postgresql/indexes.html",
          "page": 42,
          "section": "11.2. Index Types"
        },
        "score": 0.9231
      }
    ]
  },
  "id": "call-1"
}
```

**Error Response** (tool-level error, not protocol-level)

```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [],
    "is_error": true,
    "error": "Document not found"
  },
  "id": "call-1"
}
```

> **Note**: Tool execution errors are returned in `result.is_error` rather than
> the top-level `error` field. Top-level `error` is reserved for protocol and
> transport failures.

---

### `resources/list`

Return the document collections (resources) that this server can search.

**Request**

```json
{
  "jsonrpc": "2.0",
  "method": "resources/list",
  "id": "res-1"
}
```

**Response**

```json
{
  "jsonrpc": "2.0",
  "result": {
    "resources": [
      {
        "uri": "techdoc://postgresql/docs",
        "name": "PostgreSQL Documentation",
        "description": "PostgreSQL 16 official documentation",
        "mime_type": "application/x-documentation"
      }
    ]
  },
  "id": "res-1"
}
```

Resource URIs use the `techdoc://` scheme: `techdoc://<doc_type>/<collection>`.

---

## Tool Reference

### `search_documentation`

Performs semantic similarity search across the loaded document vector store and
returns the most relevant chunks.

| Argument      | Type    | Required | Default | Constraints              |
|---------------|---------|----------|---------|--------------------------|
| `query`       | string  | Yes      | —       | 1–1000 characters        |
| `max_results` | integer | No       | `5`     | 1–20                     |
| `doc_type`    | string  | No       | `"all"` | `"all"`, `"postgresql"`, `"ubuntu"` |

**Response content item fields**

| Field      | Type   | Description                                  |
|------------|--------|----------------------------------------------|
| `type`     | string | Always `"text"` for this tool                |
| `text`     | string | The document chunk content                   |
| `metadata` | object | Source file, page number, section heading    |
| `score`    | float  | Similarity score (higher = more relevant)    |

---

### `get_document`

Retrieves a single document by its unique identifier.

| Argument      | Type   | Required | Description                        |
|---------------|--------|----------|------------------------------------|
| `document_id` | string | Yes      | Document path or unique identifier |

**Response content item fields**

| Field      | Type   | Description           |
|------------|--------|-----------------------|
| `type`     | string | Always `"text"`       |
| `text`     | string | Full document content |
| `metadata` | object | Document metadata     |

---

## Authentication

Authentication is controlled by the `MCP_AUTH_REQUIRED` environment variable.

### Development (default — auth disabled)

`MCP_AUTH_REQUIRED=false`. All requests are accepted without credentials.

### Production (auth enabled)

`MCP_AUTH_REQUIRED=true`. Every request must include a Bearer token in the
`Authorization` header:

```
Authorization: Bearer <your-api-key>
```

Requests without a valid key receive HTTP 401:

```json
{
  "detail": "Invalid API key"
}
```

API keys are provisioned via the `MCP_API_KEY` environment variable. In a
production deployment, keys should be stored in a secrets manager rather than
plain environment variables.

---

## Input Validation

The server enforces the following validation rules before executing any tool:

| Parameter     | Rule                                  | Error code |
|---------------|---------------------------------------|------------|
| `query`       | Required, string, 1–1000 chars        | -32602     |
| `max_results` | Integer, 1–20                         | -32602     |
| `doc_type`    | One of `"all"`, `"postgresql"`, `"ubuntu"` | -32602 |
| `document_id` | Required, non-empty string            | -32602     |

Requests that fail validation receive an error response with `code: -32602`
and a human-readable `message`.

---

## Configuration Reference

Set these in your `.env` file or as environment variables:

```bash
# MCP Server
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8001
MCP_AUTH_REQUIRED=false
MCP_API_KEY=your-api-key-here
MCP_ALLOWED_ORIGINS=*
```

| Variable             | Default       | Description                                        |
|----------------------|---------------|----------------------------------------------------|
| `MCP_SERVER_HOST`    | `0.0.0.0`     | Bind address                                       |
| `MCP_SERVER_PORT`    | `8001`        | Listen port                                        |
| `MCP_AUTH_REQUIRED`  | `false`       | Require Bearer token for all requests              |
| `MCP_API_KEY`        | *(unset)*     | Valid API key when auth is enabled                 |
| `MCP_ALLOWED_ORIGINS`| `*`           | CORS allowed origins (comma-separated)             |

---

## Complete Request/Response Examples

### Example 1 — List Tools

```bash
curl -s -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": "ex-1"
  }' | python -m json.tool
```

### Example 2 — Search Documentation

```bash
curl -s -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search_documentation",
      "arguments": {
        "query": "How do I configure max_connections?",
        "max_results": 3,
        "doc_type": "postgresql"
      }
    },
    "id": "ex-2"
  }' | python -m json.tool
```

### Example 3 — Search with Authentication

```bash
curl -s -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search_documentation",
      "arguments": {
        "query": "VACUUM ANALYZE explained",
        "max_results": 5
      }
    },
    "id": "ex-3"
  }'
```

### Example 4 — Retrieve a Specific Document

```bash
curl -s -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "get_document",
      "arguments": {
        "document_id": "postgresql/indexes.html"
      }
    },
    "id": "ex-4"
  }'
```

---

## How an LLM Uses This Server

A typical MCP-enabled interaction follows this sequence:

```
1. LLM receives user question
2. LLM calls  POST /mcp  →  tools/list
   └─ learns which tools are available and their schemas
3. LLM decides to call  search_documentation
   └─ POST /mcp  →  tools/call  { name: "search_documentation", arguments: {...} }
4. Server returns relevant document chunks as content items
5. LLM synthesizes an answer grounded in the retrieved content
6. LLM returns answer + citations to the user
```

The `scripts/mcp_llm_integration_demo.py` script simulates this flow end-to-end.

---

## Running the Server

### Development

```bash
python src/mcp_server/server.py
```

### Production (uvicorn directly)

```bash
uvicorn src.mcp_server.server:app --host 0.0.0.0 --port 8001 --workers 2
```

### Docker Compose

```bash
docker-compose up mcp-server
```

---

## Relation to the Official MCP Specification

This implementation follows the [Model Context Protocol specification](https://modelcontextprotocol.io/).
Key design choices made for this project:

- **Transport**: HTTP/JSON-RPC (not stdio) — easier to test with `curl` and
  Postman during development.
- **Authentication**: Optional Bearer token rather than the spec's OAuth flow —
  appropriate for a single-tenant demo system.
- **Tool-level errors**: Returned in `result.is_error` per spec recommendation
  so the LLM can read and handle them gracefully.
- **Resources**: Minimal `resources/list` implementation; full resource-read
  support is deferred to a later sprint.

---

## Related Documents

- `docs/sprint2.md` — day-by-day implementation plan for the MCP server
- `docs/design-decisions.md` — architectural rationale
- `src/mcp_server/server.py` — FastAPI server implementation
- `src/mcp_server/tools.py` — tool implementations wired to the vector store
- `src/mcp_server/auth.py` — authentication logic
- `src/mcp_server/validators.py` — input validation
- `scripts/test_mcp_client.py` — manual test client
- `scripts/mcp_llm_integration_demo.py` — end-to-end LLM demo
