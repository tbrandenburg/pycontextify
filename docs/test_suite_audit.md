# Python Test Suite Audit

## Classification overview

### Unit suite (`tests/unit`)
| Module | Notes |
| --- | --- |
| `tests/unit/test_chunker.py` | Exercises chunker abstractions entirely in memory with temporary configs and mocks, keeping the scope to single components.【F:tests/unit/test_chunker.py†L1-L79】 |
| `tests/unit/test_cli_args.py` | Validates CLI parsing and configuration override helpers purely via argparse without touching the runtime.【F:tests/unit/test_cli_args.py†L1-L77】 |
| `tests/unit/test_config.py` | Manipulates environment variables and patches dotenv loading to verify configuration precedence and validation logic in isolation.【F:tests/unit/test_config.py†L1-L80】 |
| `tests/unit/test_embeddings.py` | Uses mock embedder implementations and factory shims to test error handling and metadata without loading real models.【F:tests/unit/test_embeddings.py†L1-L80】 |
| `tests/unit/test_hybrid_search.py` | Focuses on the hybrid search engine’s bookkeeping with mocked metadata stores and vector scores for deterministic assertions.【F:tests/unit/test_hybrid_search.py†L1-L105】 |
| `tests/unit/test_loaders.py` | Covers loader helpers with patched Crawl4AI responses and temporary files, avoiding any real crawling or filesystem traversal beyond fixtures.【F:tests/unit/test_loaders.py†L1-L160】 |
| `tests/unit/test_metadata.py` | Confirms dataclass defaults, relationship helpers, and serialization behaviour in memory-only scenarios.【F:tests/unit/test_metadata.py†L1-L120】 |
| `tests/unit/test_models.py` | Verifies response builders and enum utilities for search results without hitting the indexing stack.【F:tests/unit/test_models.py†L1-L80】 |
| `tests/unit/test_pdf_loader.py` | Exercises PDF loader validation paths with extensive mocking of engine availability and file IO.【F:tests/unit/test_pdf_loader.py†L1-L80】 |
| `tests/unit/test_vector_store.py` | Mocks FAISS bindings to test vector addition, persistence hooks, and error handling without a real index backend.【F:tests/unit/test_vector_store.py†L1-L76】 |

### Integration suite (`tests/integration`)
| Module | Notes |
| --- | --- |
| `tests/integration/test_bootstrap_integration.py` | Spins up local HTTP servers, archives, and checksum validation to cover index bootstrap workflows end to end.【F:tests/integration/test_bootstrap_integration.py†L1-L118】 |
| `tests/integration/test_hybrid_search.py` | Stores real metadata and runs vector-weighted queries through the hybrid search engine.【F:tests/integration/test_hybrid_search.py†L1-L27】 |
| `tests/integration/test_integration.py` | Drives the indexing pipeline via `IndexManager`, chunking sample content and validating retrieval results.【F:tests/integration/test_integration.py†L1-L70】 |
| `tests/integration/test_loaders.py` | Exercises CodeLoader against actual directories and stitches Crawl4AI results through the WebpageLoader with patched execution.【F:tests/integration/test_loaders.py†L39-L70】 |
| `tests/integration/test_mcp_server.py` | Contains comprehensive tool-level tests using FastMCP plus full workflow scenarios that index documents and query results.【F:tests/integration/test_mcp_server.py†L195-L470】 |
| `tests/integration/test_persistence.py` | Validates auto-persist and auto-load flows by writing indexes to disk, rehydrating them, and executing searches.【F:tests/integration/test_persistence.py†L1-L99】 |
| `tests/integration/test_recursive_crawling.py` | Performs live recursive crawling against python.org to verify real-world loader behaviour (marked slow).【F:tests/integration/test_recursive_crawling.py†L1-L78】 |

## MCP server function coverage
| MCP function | Unit test coverage | Real integration coverage | Notes |
| --- | --- | --- | --- |
| `index_code` | Mocked success/validation paths exercise FastMCP wiring through patched managers.【F:tests/integration/test_mcp_server.py†L195-L238】 | ❌ None | No scenario indexes a real codebase; integration fixtures only hit error paths for nonexistent directories.【F:tests/integration/test_mcp_server.py†L462-L470】 |
| `index_document` | Mocked manager confirms validation and response shaping for supported and unsupported extensions.【F:tests/integration/test_mcp_server.py†L239-L270】 | ✅ Full workflow indexes a temporary markdown file through the MCP interface.【F:tests/integration/test_mcp_server.py†L406-L456】 |
| `index_webpage` | ❌ No direct unit coverage. | ❌ None | The suite still lacks an MCP-level invocation of the webpage indexing tool; only loader integrations cover crawling.【F:tests/integration/test_recursive_crawling.py†L1-L78】 |
| `search` | Mocked `IndexManager.search` responses verify result formatting and validation.【F:tests/integration/test_mcp_server.py†L272-L305】 | ✅ Integration workflow asserts real search results after ingesting content.【F:tests/integration/test_mcp_server.py†L439-L454】 |
| `reset_index` | Unit tests confirm confirmation semantics and manager interactions via mocks.【F:tests/integration/test_mcp_server.py†L332-L357】 | ❌ None | No integration test performs a reset against the temporary manager fixture. |
| `status` | Unit test checks status payload structure and metadata enrichment.【F:tests/integration/test_mcp_server.py†L306-L330】 | ✅ Integration workflow verifies initial status from the isolated MCP environment.【F:tests/integration/test_mcp_server.py†L410-L413】 |

**Legend:** ✅ = present, ❌ = missing.

## Observations
- The test tree now clearly separates fast component checks under `tests/unit` from scenario-based integrations in `tests/integration`, with directory-level markers to simplify selective runs.【F:tests/README.md†L1-L32】【F:tests/unit/conftest.py†L1-L3】【F:tests/integration/conftest.py†L1-L3】
- Integration coverage clusters around document ingestion, persistence, and MCP workflows, while unit modules focus on loader utilities, metadata stores, and embedders with heavy mocking.【F:tests/integration/test_integration.py†L1-L70】【F:tests/unit/test_loaders.py†L1-L160】
- Real-world HTTP crawling remains limited to the slow recursive crawling suite; adding a non-network smoke test for the MCP webpage tool would help close the remaining coverage gap.【F:tests/integration/test_recursive_crawling.py†L1-L78】【F:tests/integration/test_mcp_server.py†L239-L470】
