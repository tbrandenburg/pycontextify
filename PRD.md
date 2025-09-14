I have created the following plan after thorough exploration and analysis of the codebase. Follow the below plan verbatim. Trust the files and references. Do not re-verify what's written in the plan. Explore only when absolutely necessary. First implement all the proposed file changes and then I'll review all the changes together at the end.

### Observations

The user wants to build a Python-based MCP server for semantic search over codebases, documents, and webpages. The server should use FastMCP for the MCP protocol, FAISS for vector similarity search, and support in-memory operation with auto-persistence configured via environment variables. The workspace is currently empty, so this is a greenfield project requiring a complete implementation from scratch. The user has specified that persistence should occur after each new resource is indexed, wants the code to be prepared for switching between different embedding providers (sentence-transformers vs Ollama) in the future, wants the project to be set up as a UV project, wants to include the lightweight knowledge graph preparation with enhanced MCP functions for relationship-aware search, wants to add a WebPageChunker for web-specific content processing, and now wants to remove the find_related and get_relationships MCP functions to maintain simplicity while keeping the search_with_context function.

### Approach

I'll create a modular architecture with clear separation of concerns, an extensible embedding system, and lightweight knowledge graph capabilities, all structured as a UV project:
- **UV Project Structure**: Use pyproject.toml for dependency management and project configuration
- **Core Components**: IndexManager (orchestrates everything), VectorStore (FAISS wrapper), Embedder (abstract interface with sentence transformers implementation), Chunker (text splitting)
- **Extensible Embedding Architecture**: Abstract base class for embedders with concrete implementations for different providers
- **Lightweight Knowledge Graph**: Enhanced metadata with relationship hints, simple relationship storage, and relationship-aware search
- **Content Loaders**: Separate loaders for code files, documents (PDF/MD/TXT), and webpages with relationship extraction
- **Enhanced Chunking**: Specialized chunkers for different content types including WebPageChunker for web-specific processing
- **Simplified MCP Interface**: FastMCP server exposing the 5 original tools plus only the search_with_context function for relationship-aware search
- **Auto-Persistence**: Immediate save after each indexing operation plus environment variable configuration

The design prioritizes simplicity by limiting the MCP interface to essential functions while maintaining extensibility for future embedding providers and knowledge graph capabilities, all within a modern UV-based Python project structure.

### Reasoning

I explored the empty workspace directory and researched the required technologies. I studied FastMCP's decorator-based API for building MCP servers, FAISS for vector similarity search, and various Python libraries for document processing (PyPDF2, pdfplumber) and web crawling (requests, BeautifulSoup). I then analyzed the requirements and designed a modular architecture that separates concerns while maintaining simplicity. The user confirmed they want auto-persistence with environment variable configuration, specified that persistence should occur after each new resource is indexed, wants the embedding system to be extensible for future providers like Ollama, wants the project to be structured as a UV project, wants to include the lightweight knowledge graph preparation with enhanced MCP functions for relationship-aware search, wants to add a WebPageChunker for specialized web content processing, and now wants to remove the find_related and get_relationships MCP functions to maintain simplicity.

## Mermaid Diagram

sequenceDiagram
    participant Client as MCP Client
    participant Server as FastMCP Server (Simplified)
    participant Manager as IndexManager
    participant Factory as ChunkerFactory
    participant WebChunker as WebPageChunker
    participant DocChunker as DocumentChunker
    participant CodeChunker as CodeChunker
    participant Embedder as BaseEmbedder
    participant VectorStore as FAISS VectorStore
    participant MetaStore as MetadataStore
    participant RelStore as RelationshipStore
    participant FS as File System

    Note over Client,FS: Simplified MCP Interface (6 Functions Only)
    Client->>Server: Available MCP Functions
    Note over Server: index_code, index_document, index_webpage, search, status, search_with_context
    Note over Server: Removed: find_related, get_relationships (for simplicity)

    Note over Client,FS: Enhanced Search with Simplified Interface
    Client->>Server: search_with_context(query="navigation menu", include_related=true)
    Server->>Manager: search_with_context(query, top_k, include_related)
    Manager->>Embedder: embed_single(query)
    Embedder-->>Manager: query_vector
    Manager->>VectorStore: search(query_vector, top_k)
    VectorStore-->>Manager: (distances, faiss_ids)
    Manager->>MetaStore: get_chunk(faiss_id)
    MetaStore-->>Manager: chunk_metadata (with relationships)
    
    alt include_related=true
        Manager->>RelStore: get_related_chunks(entities_from_results)
        RelStore-->>Manager: related_chunk_ids
        Manager->>MetaStore: get_chunk(related_chunk_id)
        MetaStore-->>Manager: related_chunk_metadata
        Manager->>Manager: re_rank_with_relationships(results, related_chunks)
    end
    
    Manager-->>Server: enhanced_results_with_context
    Server-->>Client: formatted_relationship_aware_results

    Note over Client,FS: Content-Specific Processing Maintained
    Client->>Server: index_webpage(url, recursive=true, max_depth=2)
    Server->>Manager: index_webpage(url, recursive, max_depth)
    Manager->>Manager: Load webpage content with WebpageLoader
    Manager->>Factory: get_chunker(SourceType.WEBPAGE, config)
    Factory-->>Manager: WebPageChunker instance
    Manager->>WebChunker: chunk_text(html_content, url, provider, model)
    
    Note over WebChunker: Web-Specific Processing
    WebChunker->>WebChunker: Parse HTML structure
    WebChunker->>WebChunker: Extract navigation hierarchy
    WebChunker->>WebChunker: Filter out boilerplate content
    WebChunker->>WebChunker: Extract internal/external links
    WebChunker->>WebChunker: Identify semantic sections
    WebChunker-->>Manager: [ChunkMetadata with web-specific relationships]
    
    Manager->>Embedder: embed_texts([chunk.text, ...])
    Embedder-->>Manager: embeddings_array
    Manager->>VectorStore: add_vectors(embeddings)
    VectorStore-->>Manager: faiss_ids
    Manager->>MetaStore: add_chunk(metadata with web relationships)
    Manager->>RelStore: extract_web_relationships(chunks)
    
    Note over Manager,FS: Auto-Save After Indexing
    Manager->>VectorStore: save_to_file(index_path)
    VectorStore->>FS: Write index.faiss
    Manager->>MetaStore: save_to_file(metadata_path)
    MetaStore->>FS: Write metadata.pkl
    Manager->>RelStore: save_to_file(relationships_path)
    RelStore->>FS: Write relationships.pkl
    
    Manager-->>Server: indexing_stats + relationship_stats
    Server-->>Client: success_response

    Note over Client,FS: Basic Search Still Available
    Client->>Server: search(query="function definition", top_k=10)
    Server->>Manager: search(query, top_k)
    Manager->>Embedder: embed_single(query)
    Embedder-->>Manager: query_vector
    Manager->>VectorStore: search(query_vector, top_k)
    VectorStore-->>Manager: (distances, faiss_ids)
    Manager->>MetaStore: get_chunk(faiss_id)
    MetaStore-->>Manager: chunk_metadata
    Manager-->>Server: basic_search_results
    Server-->>Client: formatted_results

## Proposed File Changes

### pyproject.toml(NEW)

Create the UV project configuration file with all dependencies and project metadata:

```toml
[project]
name = "pycontextify"
version = "0.1.0"
description = "A Python-based MCP server for semantic search over codebases, documents, and webpages with lightweight knowledge graph capabilities"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
keywords = ["mcp", "semantic-search", "embeddings", "faiss", "rag", "knowledge-graph"]

dependencies = [
    "fastmcp>=1.0.0",
    "faiss-cpu>=1.7.0",
    "sentence-transformers>=2.2.0",
    "PyPDF2>=3.0.0",
    "pdfplumber>=0.9.0",
    "beautifulsoup4>=4.12.0",
    "requests>=2.31.0",
    "psutil>=5.9.0",
    "numpy>=1.24.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
# Future embedding providers
ollama = ["ollama>=0.1.0"]
openai = ["openai>=1.0.0"]
# Development dependencies
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]
# GPU support
gpu = ["faiss-gpu>=1.7.0"]

[project.scripts]
pycontextify = "pycontextify.mcp_server:main"

[project.urls]
Homepage = "https://github.com/yourusername/pycontextify"
Repository = "https://github.com/yourusername/pycontextify"
Issues = "https://github.com/yourusername/pycontextify/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["pycontextify"]

[tool.black]
line-length = 88
target-version = ['py310']
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --cov=pycontextify"
testpaths = ["tests"]
```

This provides a complete UV project configuration with:
- Project metadata and dependencies
- Optional dependencies for future embedding providers
- Development tools configuration
- Build system setup
- Entry point for the MCP server
- Updated description mentioning knowledge graph capabilities

### pycontextify(NEW)

Create the main package directory for the pycontextify project. This follows Python packaging best practices and UV project structure.

### pycontextify/__init__.py(NEW)

Create the main package initialization file:

```python
"""PyContextify - A Python-based MCP server for semantic search.

This package provides semantic search capabilities over codebases, documents,
and webpages using FAISS vector similarity search, various embedding providers,
and lightweight knowledge graph capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Main exports
from .index.manager import IndexManager
from .index.config import Config

__all__ = ["IndexManager", "Config", "__version__"]
```

This provides a clean package interface with version information and main exports, updated to mention knowledge graph capabilities.

### pycontextify/mcp_server.py(NEW)

References: 

- pycontextify/index/manager.py(NEW)

Create the main MCP server entry point using FastMCP with simplified interface and enhanced relationship-aware search capabilities. This file will:

1. **Initialize FastMCP application** with proper metadata and description
2. **Load environment variables** using python-dotenv for configuration
3. **Create a global IndexManager instance** that will be shared across all tool calls with auto-persistence enabled
4. **Implement the 5 original MCP tools** using `@mcp.tool` decorators:
   - `index_code(path: str)`: Validate path exists, call IndexManager.index_codebase, auto-save after completion
   - `index_document(path: str)`: Validate path exists, call IndexManager.index_document, auto-save after completion
   - `index_webpage(url: str, recursive: bool = False, max_depth: int = 1)`: Validate URL format, call IndexManager.index_webpage, auto-save after completion
   - `search(query: str, top_k: int = 5)`: Validate parameters, call IndexManager.search (no save needed)
   - `status()`: Call IndexManager.get_status and return formatted statistics including persistence info

5. **Implement simplified relationship-aware MCP tool**:
   - `search_with_context(query: str, top_k: int = 5, include_related: bool = False)`: Enhanced search that can optionally include related chunks

6. **Add proper error handling** for each tool with meaningful error messages
7. **Include type hints and docstrings** for automatic schema generation
8. **Add graceful shutdown handler** to save index on server termination
9. **Add main function** for the UV script entry point and `mcp.run()` for direct execution

Each indexing tool will trigger an immediate auto-save after successful completion to ensure no data loss. The server will be agnostic to the embedding provider used by the IndexManager and will provide both basic and relationship-aware search capabilities while maintaining simplicity by exposing only essential MCP functions.

### pycontextify/index(NEW)

Create the main package directory for all indexing-related modules. This directory will contain the core business logic separated from the MCP protocol layer.

### pycontextify/index/__init__.py(NEW)

Create the index package initialization file:

```python
"""Index package for PyContextify.

This package contains all the core indexing functionality including
embedding providers, vector storage, content loading, text chunking,
and lightweight knowledge graph capabilities.
"""

from .manager import IndexManager
from .config import Config
from .metadata import ChunkMetadata, SourceType, MetadataStore
from .relationship_store import RelationshipStore

__all__ = [
    "IndexManager",
    "Config", 
    "ChunkMetadata",
    "SourceType",
    "MetadataStore",
    "RelationshipStore"
]
```

This provides clean exports for the index package components including the new relationship store.

### pycontextify/index/config.py(NEW)

Create configuration management for environment variables with embedding provider selection:

1. **Configuration class** that loads and validates environment variables:
   - `PYCONTEXTIFY_INDEX_DIR`: Directory for storing index files (default: './index_data')
   - `PYCONTEXTIFY_AUTO_PERSIST`: Enable auto-persistence (default: 'true')
   - `PYCONTEXTIFY_AUTO_LOAD`: Auto-load on startup (default: 'true')
   - `PYCONTEXTIFY_INDEX_NAME`: Base name for index files (default: 'semantic_index')
   - `PYCONTEXTIFY_COMPRESS_METADATA`: Compress metadata files (default: 'true')
   - `PYCONTEXTIFY_BACKUP_INDICES`: Create backups before overwriting (default: 'false')
   - `PYCONTEXTIFY_MAX_BACKUPS`: Maximum backup files to keep (default: '3')
   - **`PYCONTEXTIFY_EMBEDDING_PROVIDER`**: Embedding provider ('sentence_transformers', 'ollama', 'openai') (default: 'sentence_transformers')
   - `PYCONTEXTIFY_EMBEDDING_MODEL`: Model name for the selected provider (default: 'all-mpnet-base-v2')
   - **`PYCONTEXTIFY_OLLAMA_BASE_URL`**: Ollama server URL (default: 'http://localhost:11434')
   - **`PYCONTEXTIFY_OPENAI_API_KEY`**: OpenAI API key (no default)
   - `PYCONTEXTIFY_CHUNK_SIZE`: Text chunk size in tokens (default: '512')
   - `PYCONTEXTIFY_CHUNK_OVERLAP`: Overlap between chunks in tokens (default: '50')
   - **`PYCONTEXTIFY_ENABLE_RELATIONSHIPS`**: Enable relationship extraction (default: 'true')
   - **`PYCONTEXTIFY_MAX_RELATIONSHIPS_PER_CHUNK`**: Maximum relationships to extract per chunk (default: '10')

2. **Validation methods** for each configuration value with appropriate defaults
3. **Provider-specific validation** to ensure required settings are present for each embedding provider
4. **Path resolution** to handle relative and absolute paths for index directory
5. **Type conversion** for boolean and integer values from string environment variables
6. **Configuration summary** method for logging and status reporting
7. **Provider compatibility checks** to validate embedding provider configurations
8. **Relationship extraction settings** for controlling knowledge graph features

This centralizes all configuration logic and provides a single source of truth for environment variable handling, with extensible support for multiple embedding providers and relationship extraction settings.

### pycontextify/index/metadata.py(NEW)

Define the data structures for storing chunk metadata with lightweight knowledge graph capabilities:

1. **ChunkMetadata dataclass** with enhanced fields for relationship tracking:
   - `chunk_id`: Unique identifier for the chunk
   - `source_path`: Original file path or URL
   - `source_type`: Enum ('code', 'document', 'webpage')
   - `chunk_text`: The actual text content of the chunk
   - `start_char`: Starting character position in original document
   - `end_char`: Ending character position in original document
   - `file_extension`: File extension for code files
   - `created_at`: Timestamp when chunk was indexed
   - **`embedding_provider`**: Provider used to create embeddings for this chunk
   - **`embedding_model`**: Specific model used for embeddings
   - **`tags`**: List of tags for categorization and lightweight relationships
   - **`references`**: List of referenced entities (functions, classes, concepts)
   - **`parent_section`**: Document hierarchy information
   - **`code_symbols`**: List of code symbols (functions, classes) found in this chunk

2. **SourceType enum** with values: CODE, DOCUMENT, WEBPAGE

3. **MetadataStore class** to manage the mapping between FAISS indices and chunk metadata:
   - `add_chunk(metadata: ChunkMetadata) -> int`: Add metadata and return FAISS index
   - `get_chunk(faiss_id: int) -> ChunkMetadata`: Retrieve metadata by FAISS ID
   - `get_all_chunks() -> List[ChunkMetadata]`: Get all stored metadata
   - `get_stats() -> Dict`: Return statistics about stored metadata including embedding provider info and relationship counts
   - `clear()`: Clear all metadata
   - `to_dict()` and `from_dict()`: Serialization for persistence
   - `save_to_file(filepath: str)`: Save metadata to compressed pickle file
   - `load_from_file(filepath: str)`: Load metadata from pickle file
   - **`validate_embedding_compatibility(provider: str, model: str) -> bool`**: Check if existing chunks are compatible with current embedding settings
   - **`find_chunks_by_tag(tag: str) -> List[ChunkMetadata]`**: Find chunks with specific tags
   - **`find_chunks_by_reference(reference: str) -> List[ChunkMetadata]`**: Find chunks referencing specific entities
   - **`get_chunk_relationships(chunk_id: str) -> Dict`**: Get all relationships for a specific chunk

This provides a clean abstraction for managing the relationship between FAISS vector indices and the original source information with built-in persistence capabilities, embedding provider tracking, and lightweight knowledge graph support through enhanced metadata fields.

### pycontextify/index/relationship_store.py(NEW)

References: 

- pycontextify/index/metadata.py(NEW)

Implement a lightweight relationship store for managing entity relationships without full graph database complexity:

1. **RelationshipStore class** for simple relationship management:
   - `__init__()`: Initialize relationship storage with dictionaries for fast lookup
   - `add_relationship(entity: str, chunk_id: str, relationship_type: str)`: Add a relationship between entity and chunk
   - `get_related_chunks(entity: str, relationship_type: str = "all") -> List[str]`: Get chunks related to an entity
   - `get_chunk_entities(chunk_id: str) -> Dict[str, List[str]]`: Get all entities related to a chunk
   - `get_relationship_types() -> List[str]`: Get all available relationship types
   - `get_stats() -> Dict`: Return relationship statistics

2. **Relationship types** for different kinds of connections:
   - `"function_call"`: Code function calls
   - `"import"`: Code imports and dependencies
   - `"reference"`: Document cross-references
   - `"link"`: Web page links
   - `"hierarchy"`: Parent-child relationships
   - `"tag"`: Tag-based relationships

3. **Storage structure** using dictionaries for efficient lookup:
   - `relationships`: Dict[entity -> List[chunk_ids]] for forward lookup
   - `reverse_index`: Dict[chunk_id -> List[entities]] for reverse lookup
   - `relationship_types`: Dict[entity -> Dict[relationship_type -> List[chunk_ids]]] for typed relationships

4. **Query methods** for relationship-aware search:
   - `find_related_entities(entity: str, max_depth: int = 1) -> Set[str]`: Find entities related through relationships
   - `get_entity_graph(entity: str, max_depth: int = 2) -> Dict`: Get a subgraph around an entity
   - `calculate_relationship_strength(entity1: str, entity2: str) -> float`: Calculate relationship strength between entities

5. **Persistence methods**:
   - `save_to_file(filepath: str)`: Save relationships to file
   - `load_from_file(filepath: str)`: Load relationships from file
   - `clear()`: Clear all relationships

6. **Integration helpers**:
   - `extract_code_relationships(chunk: ChunkMetadata) -> List[Tuple[str, str, str]]`: Extract relationships from code chunks
   - `extract_document_relationships(chunk: ChunkMetadata) -> List[Tuple[str, str, str]]`: Extract relationships from document chunks
   - `extract_web_relationships(chunk: ChunkMetadata) -> List[Tuple[str, str, str]]`: Extract relationships from web chunks

This provides a lightweight alternative to full knowledge graphs while enabling relationship-aware search and entity discovery without the complexity of graph databases. The relationship store supports the simplified MCP interface by providing internal relationship management for the search_with_context function.

### pycontextify/index/embedders(NEW)

Create a dedicated directory for embedding implementations. This directory will contain the abstract base class and concrete implementations for different embedding providers, making it easy to add new providers in the future.

### pycontextify/index/embedders/__init__.py(NEW)

Create the embedders package initialization file that exports the main interfaces and factory:

```python
"""Embedders package for PyContextify.

This package provides an extensible embedding system with support for
multiple embedding providers including sentence-transformers, Ollama, and OpenAI.
"""

from .base import BaseEmbedder
from .factory import EmbedderFactory
from .sentence_transformers_embedder import SentenceTransformersEmbedder

__all__ = ['BaseEmbedder', 'EmbedderFactory', 'SentenceTransformersEmbedder']
```

This provides a clean public API for the embedders package while keeping implementation details internal.

### pycontextify/index/embedders/base.py(NEW)

Define the abstract base class for all embedding providers:

1. **BaseEmbedder abstract class** with the following interface:
   - `__init__(model_name: str, **kwargs)`: Initialize with model name and provider-specific parameters
   - `embed_texts(texts: List[str]) -> np.ndarray`: Abstract method to batch embed multiple texts, return float32 array
   - `embed_single(text: str) -> np.ndarray`: Abstract method to embed a single text for queries
   - `get_dimension() -> int`: Abstract method to return embedding dimension for FAISS index initialization
   - `get_model_info() -> Dict`: Abstract method to return model information for status reporting
   - `is_available() -> bool`: Abstract method to check if the provider is available/configured
   - `cleanup()`: Abstract method for resource cleanup

2. **Provider identification**:
   - `provider_name`: Class attribute identifying the provider type
   - `supported_models`: Class attribute listing supported models (can be empty for dynamic providers)

3. **Common validation methods**:
   - `_validate_texts(texts: List[str])`: Validate input texts
   - `_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray`: Normalize to unit vectors for cosine similarity

4. **Error handling**:
   - Custom exception classes for embedding-specific errors
   - `EmbeddingError`: Base exception for embedding operations
   - `ModelNotFoundError`: When requested model is not available
   - `ProviderNotAvailableError`: When provider dependencies are missing

This abstract base class ensures all embedding providers implement the same interface, making them interchangeable while allowing provider-specific optimizations and configurations.

### pycontextify/index/embedders/sentence_transformers_embedder.py(NEW)

References: 

- pycontextify/index/embedders/base.py(NEW)

Implement the sentence-transformers embedding provider:

1. **SentenceTransformersEmbedder class** inheriting from BaseEmbedder:
   - `provider_name = 'sentence_transformers'`
   - `supported_models`: List of recommended sentence-transformer models
   - `__init__(model_name: str, **kwargs)`: Initialize with model name and optional parameters

2. **Core embedding methods**:
   - `embed_texts(texts: List[str]) -> np.ndarray`: Batch embed using sentence-transformers
   - `embed_single(text: str) -> np.ndarray`: Embed single text for queries
   - `get_dimension() -> int`: Return model's embedding dimension
   - `get_model_info() -> Dict`: Return detailed model information
   - `is_available() -> bool`: Check if sentence-transformers is installed and model is accessible

3. **Lazy loading** of the sentence transformer model to avoid loading until first use

4. **Batch processing optimizations**:
   - Process in configurable batches (32-64 texts) to manage memory
   - Progress logging for large batches
   - Memory cleanup between batches
   - Handle CUDA/CPU device selection automatically

5. **Error handling** specific to sentence-transformers:
   - Model download failures
   - CUDA out of memory errors
   - Invalid model names

6. **Performance optimizations**:
   - Model caching to avoid reloading
   - Efficient tensor operations
   - Memory-mapped model loading for large models

7. **Configuration support**:
   - Device selection (CPU/CUDA)
   - Batch size tuning
   - Model cache directory

This implementation provides a robust, production-ready sentence-transformers backend while following the abstract interface for easy swapping with other providers.

### pycontextify/index/embedders/factory.py(NEW)

References: 

- pycontextify/index/embedders/base.py(NEW)
- pycontextify/index/embedders/sentence_transformers_embedder.py(NEW)

Implement the embedder factory for provider selection and instantiation:

1. **EmbedderFactory class** with static methods:
   - `create_embedder(provider: str, model_name: str, **kwargs) -> BaseEmbedder`: Main factory method
   - `get_available_providers() -> List[str]`: Return list of available providers
   - `get_supported_models(provider: str) -> List[str]`: Return supported models for a provider
   - `validate_provider_config(provider: str, **kwargs) -> bool`: Validate provider-specific configuration

2. **Provider registry**:
   - Dictionary mapping provider names to embedder classes
   - Dynamic registration system for easy addition of new providers
   - `register_provider(name: str, embedder_class: Type[BaseEmbedder])`: Register new providers

3. **Built-in provider support**:
   - `sentence_transformers`: SentenceTransformersEmbedder (implemented)
   - `ollama`: Placeholder for future OllamaEmbedder implementation
   - `openai`: Placeholder for future OpenAIEmbedder implementation

4. **Provider availability checking**:
   - Check if required dependencies are installed
   - Validate configuration parameters
   - Test provider connectivity (for remote providers)

5. **Error handling**:
   - Clear error messages for unsupported providers
   - Helpful suggestions for missing dependencies
   - Configuration validation errors

6. **Future provider stubs**:
   ```python
   # Placeholder methods for future implementations:
   def _create_ollama_embedder(model_name: str, base_url: str, **kwargs):
       # Will implement OllamaEmbedder when ready
       raise NotImplementedError("Ollama provider not yet implemented")
   
   def _create_openai_embedder(model_name: str, api_key: str, **kwargs):
       # Will implement OpenAIEmbedder when ready
       raise NotImplementedError("OpenAI provider not yet implemented")
   ```

This factory pattern makes it trivial to add new embedding providers by implementing the BaseEmbedder interface and registering them with the factory.

### pycontextify/index/chunker.py(NEW)

References: 

- pycontextify/index/metadata.py(NEW)
- pycontextify/index/config.py(NEW)

Implement text chunking strategies for different content types with relationship extraction, including a specialized WebPageChunker:

1. **BaseChunker abstract class** with:
   - `chunk_text(text: str, source_path: str, embedding_provider: str, embedding_model: str) -> List[ChunkMetadata]`: Abstract method with embedding info
   - `_create_chunk_metadata()`: Helper to create metadata objects with embedding provider info and relationships
   - `_extract_relationships()`: Helper to extract relationship hints from text
   - Configuration parameters from environment variables

2. **SimpleChunker class** for basic token-based chunking:
   - Configurable `chunk_size` and `overlap` from environment variables
   - Split text by whitespace, group into chunks with overlap
   - Track character positions for each chunk
   - Handle edge cases for very short or very long texts
   - Include embedding provider and model in metadata
   - Extract basic tags and references

3. **CodeChunker class** for code-aware chunking with relationship extraction:
   - Inherit from SimpleChunker but add code-specific logic
   - Attempt to split at function/class boundaries when possible
   - Preserve indentation context
   - Handle different programming languages (Python, JavaScript, etc.)
   - Respect natural code structure while maintaining chunk size limits
   - **Extract code symbols**: function names, class names, variable names
   - **Extract references**: function calls, imports, inheritance
   - **Identify relationships**: caller-callee, import dependencies

4. **DocumentChunker class** for document-specific chunking with structure extraction:
   - Split by paragraphs first, then by sentences if paragraphs are too large
   - Preserve section headers and structure
   - Handle markdown headers and formatting
   - Maintain semantic coherence in chunks
   - **Extract hierarchy**: section/subsection relationships from headers
   - **Extract references**: citations, cross-references, links
   - **Identify tags**: keywords, topics, categories

5. **WebPageChunker class** for web-specific content processing:
   - Inherit from DocumentChunker but add web-specific logic
   - **HTML structure awareness**: Respect HTML semantic elements (article, section, nav, aside)
   - **Content prioritization**: Prioritize main content over navigation, sidebars, footers
   - **Link extraction**: Extract and categorize internal vs external links
   - **Metadata extraction**: Extract page title, meta descriptions, headings hierarchy
   - **URL-based relationships**: Derive relationships from URL structure and breadcrumbs
   - **Navigation structure**: Extract site hierarchy from navigation elements
   - **Content filtering**: Remove boilerplate content (headers, footers, ads, navigation)
   - **Semantic chunking**: Split by HTML sections and maintain semantic boundaries
   - **Tag extraction**: Extract keywords from meta tags, alt text, and heading structure

6. **ChunkerFactory** to select appropriate chunker based on content type:
   - `get_chunker(source_type: SourceType, config: Config) -> BaseChunker`
   - Pass configuration to chunkers for consistent behavior
   - **Enhanced selection logic**: Choose WebPageChunker for WEBPAGE source type

The chunking strategy should balance semantic coherence with search granularity, ensuring chunks are meaningful units that can be effectively retrieved. The WebPageChunker specifically handles the unique challenges of web content including HTML structure, navigation elements, and link relationships. Chunk sizes will be configurable via environment variables. All chunks will include embedding provider information and extracted relationships for lightweight knowledge graph capabilities that support the simplified MCP interface.

### pycontextify/index/vector_store.py(NEW)

References: 

- pycontextify/index/config.py(NEW)

Implement the FAISS vector store wrapper with auto-persistence:

1. **VectorStore class** that encapsulates FAISS operations:
   - `__init__(dimension: int, config: Config)`: Initialize with IndexFlatIP (cosine similarity) for simplicity
   - `add_vectors(vectors: np.ndarray) -> List[int]`: Add vectors and return FAISS IDs
   - `search(query_vector: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]`: Search and return distances and indices
   - `get_total_vectors() -> int`: Return number of indexed vectors
   - `get_index_info() -> Dict`: Return FAISS index statistics
   - `clear()`: Clear all vectors from index
   - **`get_embedding_dimension() -> int`**: Return the dimension of stored embeddings

2. **Persistence methods with backup support**:
   - `save_to_file(filepath: str)`: Save FAISS index to disk with optional backup
   - `load_from_file(filepath: str)`: Load FAISS index from disk
   - `_create_backup(filepath: str)`: Create backup of existing index before overwriting
   - `_cleanup_old_backups(base_path: str)`: Remove old backups beyond max limit
   - `is_empty() -> bool`: Check if index contains any vectors

3. **Memory management**:
   - `get_memory_usage() -> int`: Estimate memory usage in bytes
   - Use float32 vectors to balance precision and memory
   - Efficient memory allocation for large vector additions

4. **Error handling** for FAISS operations and file I/O with detailed error messages

5. **Index type selection and future extensibility**:
   - Start with IndexFlatIP for exact search (good for MVP)
   - Add comments for future optimization with IndexIVFFlat for larger datasets
   - Support for index rebuilding if needed
   - **Dimension validation** to ensure consistency with embedding provider

6. **Embedding provider compatibility**:
   - Store embedding dimension as metadata
   - Validate that new vectors match existing dimension
   - Support for index migration when changing embedding providers (future feature)

The vector store should provide a clean abstraction over FAISS, hiding the complexity of index management while exposing the essential operations needed by the IndexManager. Auto-persistence will be triggered after each indexing operation, and the store will be prepared for handling different embedding dimensions from various providers.

### pycontextify/index/loaders.py(NEW)

References: 

- pycontextify/index/metadata.py(NEW)

Implement content loaders for different source types with relationship extraction:

1. **BaseLoader abstract class**:
   - `load(source: str) -> List[Tuple[str, str]]`: Returns list of (file_path, content) tuples
   - `_is_supported_file()`: Check if file type is supported
   - `_get_file_stats()`: Return file statistics for progress reporting
   - `_extract_metadata()`: Extract basic metadata for relationship building

2. **CodeLoader class** for codebase indexing with dependency extraction:
   - `load(directory_path: str)`: Recursively walk directory
   - **File filtering**: Include common code extensions (.py, .js, .java, .cpp, .h, .md, .txt, .json, .yaml, .yml, .rs, .go, .php, .rb, .swift, .kt)
   - **Exclusion patterns**: Skip .git, node_modules, __pycache__, .venv, dist, build, target, .idea, .vscode directories
   - **Binary file detection**: Skip binary files using heuristics and magic number detection
   - **Size limits**: Skip files larger than configurable threshold
   - Read files with UTF-8 encoding, handle encoding errors gracefully
   - Progress reporting for large directory structures
   - **Extract import relationships**: Identify module dependencies and imports
   - **Identify project structure**: Package hierarchies and file relationships

3. **DocumentLoader class** for individual documents with structure extraction:
   - `load(file_path: str)`: Load single document
   - **PDF handling**: Try PyPDF2 first, fallback to pdfplumber for complex layouts
   - **Text files**: Direct UTF-8 reading for .txt, .md files
   - **Error handling**: Graceful degradation for corrupted or unsupported files
   - **Metadata extraction**: Extract document properties when available
   - **Structure extraction**: Identify headers, sections, and document hierarchy
   - **Reference extraction**: Find citations, cross-references, and external links

4. **WebpageLoader class** for web content with link extraction:
   - `load(url: str, recursive: bool = False, max_depth: int = 1)`: Crawl web content
   - **HTML parsing**: Use BeautifulSoup to extract clean text, remove scripts/styles/navigation
   - **Recursive crawling**: Follow links up to max_depth, avoid external domains
   - **Duplicate detection**: Track visited URLs to avoid cycles
   - **Rate limiting**: Add delays between requests to be respectful (1-2 seconds)
   - **Robots.txt respect**: Basic robots.txt checking for ethical crawling
   - **Error handling**: Handle network errors, timeouts, and invalid HTML
   - **Content filtering**: Extract main content, skip navigation and ads
   - **Link extraction**: Identify internal and external links for relationship building
   - **Structure extraction**: Extract page hierarchy from URL structure and navigation
   - **HTML preservation**: Maintain HTML structure information for WebPageChunker
   - **Metadata extraction**: Extract page titles, meta descriptions, and semantic markup

5. **LoaderFactory** to select appropriate loader:
   - `get_loader(source_type: SourceType) -> BaseLoader`
   - Pass configuration for customizable behavior

Each loader should focus on extracting clean, readable text while preserving important structural information and handling errors gracefully. Progress reporting will be included for long-running operations. The loaders will also extract relationship hints that can be used by the relationship store for building lightweight knowledge graphs. The WebpageLoader specifically preserves HTML structure information that can be utilized by the WebPageChunker for better content processing.

### pycontextify/index/manager.py(NEW)

References: 

- pycontextify/index/config.py(NEW)
- pycontextify/index/metadata.py(NEW)
- pycontextify/index/embedders/__init__.py(NEW)
- pycontextify/index/chunker.py(NEW)
- pycontextify/index/vector_store.py(NEW)
- pycontextify/index/loaders.py(NEW)
- pycontextify/index/relationship_store.py(NEW)

Implement the central IndexManager that orchestrates all indexing operations with immediate auto-persistence, extensible embedding support, and simplified relationship-aware search:

1. **IndexManager class** as the main coordinator:
   - `__init__(config: Config)`: Initialize all components with configuration
   - **Embedder initialization**: Use EmbedderFactory to create embedder based on config
   - **Relationship store initialization**: Create RelationshipStore for lightweight knowledge graph
   - Auto-load existing index on startup if enabled and files exist
   - **Embedding compatibility checking**: Validate that existing index is compatible with current embedding settings

2. **Indexing methods with immediate persistence**:
   - `index_codebase(path: str) -> Dict`: Validate path, use CodeLoader, chunk with CodeChunker, embed all files, extract relationships, **auto-save immediately**
   - `index_document(path: str) -> Dict`: Validate path, use DocumentLoader, chunk with DocumentChunker, process single document, extract relationships, **auto-save immediately**
   - `index_webpage(url: str, recursive: bool, max_depth: int) -> Dict`: Use WebpageLoader with crawling parameters, **chunk with WebPageChunker**, extract relationships, **auto-save immediately**

3. **Auto-persistence implementation**:
   - `_auto_save()`: Internal method to save both FAISS index, metadata, and relationships
   - Called automatically after each successful indexing operation
   - Error handling to ensure indexing doesn't fail due to save errors
   - Atomic save operations to prevent corruption

4. **Search functionality with relationship awareness**:
   - `search(query: str, top_k: int) -> List[Dict]`: Basic semantic search using current embedder
   - **`search_with_context(query: str, top_k: int = 5, include_related: bool = False) -> List[Dict]`**: Enhanced search that can optionally include related chunks
     - Perform normal vector search first
     - If include_related=True, expand results with relationship data
     - Re-rank considering relationship strength and context
   - Format results with source information, chunk text, relevance scores, and relationship context

5. **Status and monitoring**:
   - `get_status() -> Dict`: Return comprehensive statistics including:
     - Total indexed files and chunks
     - Memory usage (using psutil)
     - Vector store statistics
     - **Embedding provider and model information**
     - **Relationship statistics**: total relationships, relationship types, entity counts
     - **Content type breakdown**: code vs document vs webpage chunks
     - **Persistence status**: last save time, auto-save enabled, index file sizes
     - Configuration summary

6. **Persistence operations**:
   - `save_index()`: Manual save method (used internally by auto-save) - saves FAISS index, metadata, and relationships
   - `load_index()`: Load previously saved index, metadata, and relationships on startup
   - `clear_index()`: Clear all indexed data and optionally remove saved files
   - `_ensure_index_directory()`: Create index directory if it doesn't exist
   - **`_validate_embedding_compatibility()`**: Check if existing index is compatible with current embedding settings

7. **Embedding provider management**:
   - **`_initialize_embedder()`**: Create embedder using factory based on configuration
   - **`_check_embedding_compatibility()`**: Validate that new embeddings are compatible with existing index
   - **`get_embedding_info() -> Dict`**: Return current embedding provider and model information
   - Handle embedding provider changes gracefully (warn user about incompatibility)

8. **Relationship management**:
   - **`_extract_relationships(chunks: List[ChunkMetadata])`**: Coordinate relationship extraction from chunks
   - **`_build_relationship_graph()`**: Build relationships between entities and chunks
   - **`_calculate_relationship_scores(results: List[Dict], query: str) -> List[Dict]`**: Calculate relationship-based relevance scores for search_with_context

9. **Chunker coordination**:
   - **`_get_appropriate_chunker(source_type: SourceType)`**: Select the right chunker for content type
   - **WebPageChunker integration**: Use WebPageChunker specifically for webpage content to handle HTML structure and web-specific relationships
   - **Chunker configuration**: Pass embedding provider info to all chunkers

10. **Error handling and logging**:
    - Comprehensive error handling for all operations
    - Progress reporting for long-running indexing operations
    - Validation of inputs and graceful error recovery
    - Detailed logging for debugging and monitoring
    - **Embedding-specific error handling** for provider failures
    - **Relationship extraction error handling**
    - **Web-specific error handling** for crawling and HTML parsing issues

11. **Internal coordination**:
    - Manage the flow: load → chunk (with appropriate chunker) → embed → store → extract relationships → update metadata → **auto-save**
    - Batch processing for efficiency
    - Thread safety considerations for concurrent access
    - Resource cleanup and memory management
    - **Embedder lifecycle management** (initialization, cleanup)
    - **Relationship store coordination**
    - **Content-type-specific processing** with appropriate chunkers

The IndexManager serves as the single point of coordination, ensuring all components work together seamlessly while providing immediate persistence after each indexing operation, supporting multiple embedding providers through the factory pattern, enabling relationship-aware search through lightweight knowledge graph capabilities via the simplified search_with_context function, and utilizing specialized chunkers including the WebPageChunker for optimal content processing.

### .env.example(NEW)

Create an example environment configuration file with all available settings including embedding provider options and relationship extraction settings:

```bash
# PyContextify MCP Server Configuration
# Copy this file to .env and customize as needed

# Index Storage Configuration
PYCONTEXTIFY_INDEX_DIR=./index_data
PYCONTEXTIFY_INDEX_NAME=semantic_index

# Persistence Settings
PYCONTEXTIFY_AUTO_PERSIST=true
PYCONTEXTIFY_AUTO_LOAD=true
PYCONTEXTIFY_COMPRESS_METADATA=true
PYCONTEXTIFY_BACKUP_INDICES=false
PYCONTEXTIFY_MAX_BACKUPS=3

# Embedding Provider Configuration
# Options: sentence_transformers, ollama, openai
PYCONTEXTIFY_EMBEDDING_PROVIDER=sentence_transformers

# Sentence Transformers Settings (default provider)
PYCONTEXTIFY_EMBEDDING_MODEL=all-mpnet-base-v2
# Alternative models: all-MiniLM-L6-v2, all-distilroberta-v1, multi-qa-mpnet-base-dot-v1

# Ollama Settings (for future use)
# PYCONTEXTIFY_EMBEDDING_PROVIDER=ollama
# PYCONTEXTIFY_EMBEDDING_MODEL=nomic-embed-text
# PYCONTEXTIFY_OLLAMA_BASE_URL=http://localhost:11434

# OpenAI Settings (for future use)
# PYCONTEXTIFY_EMBEDDING_PROVIDER=openai
# PYCONTEXTIFY_EMBEDDING_MODEL=text-embedding-3-small
# PYCONTEXTIFY_OPENAI_API_KEY=your_api_key_here

# Text Chunking Configuration
PYCONTEXTIFY_CHUNK_SIZE=512
PYCONTEXTIFY_CHUNK_OVERLAP=50

# Lightweight Knowledge Graph Settings
PYCONTEXTIFY_ENABLE_RELATIONSHIPS=true
PYCONTEXTIFY_MAX_RELATIONSHIPS_PER_CHUNK=10

# Performance Settings (Optional)
# PYCONTEXTIFY_MAX_FILE_SIZE_MB=10
# PYCONTEXTIFY_BATCH_SIZE=32
# PYCONTEXTIFY_CRAWL_DELAY_SECONDS=1
```

This provides a comprehensive template for users to configure the MCP server according to their needs, with clear examples for different embedding providers, relationship extraction settings, and sensible defaults. Future providers are documented but commented out.

### README.md(NEW)

References: 

- pycontextify/mcp_server.py(NEW)
- pyproject.toml(NEW)
- .env.example(NEW)

Create comprehensive documentation for the UV-based MCP server project with knowledge graph capabilities and specialized web content processing:

1. **Project overview** explaining the purpose and capabilities:
   - Semantic search over codebases, documents, and webpages
   - In-memory vector search with FAISS
   - **Lightweight knowledge graph** with relationship extraction and entity linking
   - **Specialized content processing** with dedicated chunkers for different content types
   - Auto-persistence with environment variable configuration
   - FastMCP-based MCP server implementation
   - **Extensible embedding system** supporting multiple providers
   - **Modern UV-based Python project** with proper dependency management
   - **Simplified interface** focusing on essential functionality

2. **Installation instructions**:
   - Python version requirements (3.10+)
   - **UV installation**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - **Project setup**: `uv sync` to install dependencies
   - **Development setup**: `uv sync --extra dev` for development tools
   - Environment variable setup with `.env` file
   - **Provider-specific setup instructions**:
     - Sentence Transformers (default, no additional setup)
     - Ollama (future: `uv sync --extra ollama`)
     - OpenAI (future: `uv sync --extra openai`)

3. **UV Project Benefits**:
   - **Fast dependency resolution** and installation
   - **Lockfile support** for reproducible builds
   - **Virtual environment management** built-in
   - **Optional dependencies** for different embedding providers
   - **Development tools** integration

4. **Usage examples**:
   - **Start server**: `uv run pycontextify` or `uv run fastmcp run pycontextify/mcp_server.py`
   - **Development mode**: `uv run --extra dev fastmcp run pycontextify/mcp_server.py`
   - Example MCP tool calls for each function
   - Integration with Claude Desktop or other MCP clients
   - Sample MCP client configuration

5. **Configuration guide**:
   - Copy `.env.example` to `.env`
   - Detailed explanation of each environment variable
   - **Embedding provider selection and configuration**
   - **Knowledge graph settings** for relationship extraction
   - Performance tuning recommendations
   - Storage and backup configuration

6. **Architecture overview**:
   - High-level component diagram
   - Explanation of the indexing pipeline
   - Vector similarity search process
   - **Lightweight knowledge graph architecture**
   - **Content-specific processing** with specialized chunkers
   - Auto-persistence mechanism
   - **Embedding provider architecture** and extensibility

7. **MCP Functions documentation**:
   - `index_code(path)`: Recursively index codebase with relationship extraction using CodeChunker
   - `index_document(path)`: Index individual documents with structure analysis using DocumentChunker
   - `index_webpage(url, recursive, max_depth)`: Index web content with HTML-aware processing using WebPageChunker
   - `search(query, top_k)`: Basic semantic search
   - `status()`: System status and statistics (including embedding provider and relationship info)
   - **`search_with_context(query, top_k, include_related)`**: Enhanced search with relationship context

8. **Content Processing Specialization**:
   - **CodeChunker**: Function/class boundary awareness, symbol extraction, import relationships
   - **DocumentChunker**: Section hierarchy, citation extraction, cross-references
   - **WebPageChunker**: HTML structure awareness, link extraction, navigation hierarchy, content filtering
   - **Automatic chunker selection** based on content type

9. **Knowledge Graph Features**:
   - **Lightweight approach**: No external graph database required
   - **Relationship types**: Function calls, imports, references, links, hierarchies
   - **Entity extraction**: Automatic extraction from code, documents, and web content
   - **Relationship-aware search**: Enhanced search considering entity relationships via search_with_context
   - **Web-specific relationships**: Page hierarchies, internal/external links, navigation structure
   - **Simplified interface**: Relationships accessible through search_with_context function
   - **Future extensibility**: Prepared for full knowledge graph implementation

10. **Embedding Providers**:
    - **Currently supported**: Sentence Transformers
    - **Planned**: Ollama, OpenAI
    - How to switch between providers
    - Compatibility considerations
    - Performance comparisons

11. **Development**:
    - **Code formatting**: `uv run black .`
    - **Linting**: `uv run flake8`
    - **Type checking**: `uv run mypy pycontextify`
    - **Testing**: `uv run pytest`
    - **Adding dependencies**: `uv add package-name`

12. **Supported file types and formats**:
    - Code files: .py, .js, .java, .cpp, .rs, .go, etc.
    - Documents: .pdf, .md, .txt
    - Web content: HTML pages with optional recursive crawling and specialized processing

13. **Performance considerations**:
    - Memory requirements for large codebases
    - Embedding model selection
    - Chunking strategy impact
    - Auto-persistence overhead
    - Relationship extraction performance
    - Web crawling performance and rate limiting
    - Provider-specific performance characteristics

14. **Troubleshooting section** with common issues and solutions:
    - UV-specific issues
    - Model loading errors
    - Memory issues
    - File permission problems
    - Network connectivity for web crawling
    - **Embedding provider specific issues**
    - **Relationship extraction issues**
    - **Web crawling and HTML parsing issues**

15. **Contributing**:
    - Setting up development environment with UV
    - Code style guidelines
    - Adding new embedding providers
    - Extending relationship extraction
    - Adding new chunker types
    - Testing guidelines

The README should provide clear, actionable information for users to get started quickly with the UV-based project while understanding the system's capabilities, configuration options, auto-persistence behavior, embedding provider extensibility, lightweight knowledge graph features accessible through the simplified MCP interface, and specialized content processing including the WebPageChunker for optimal web content handling.

### .gitignore(NEW)

Create a comprehensive .gitignore file for the UV-based Python project:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# UV specific
.venv/
uv.lock

# Virtual environments (other tools)
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific - Index data and persistence
*.faiss
*.pkl
index_data/
logs/
temp/
backups/

# Model cache (all embedding providers)
.cache/
models/
.sentence_transformers/
.ollama/
.openai_cache/

# Environment configuration
.env

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Type checking
.mypy_cache/
.dmypy.json
dmypy.json

# Jupyter Notebooks
.ipynb_checkpoints

# pyenv
.python-version
```

This ensures that generated files, caches, virtual environments, IDE-specific files, UV-specific files, and most importantly the auto-persisted index data are not tracked in version control. It also includes cache directories for various embedding providers while keeping the example configuration file.

### tests(NEW)

Create the tests directory for the UV project. This follows Python testing best practices and will contain unit tests, integration tests, and test fixtures for the pycontextify package.

### tests/__init__.py(NEW)

Create an empty `__init__.py` file to make the tests directory a Python package. This allows for proper test discovery and organization.

### tests/test_basic.py(NEW)

References: 

- pycontextify/__init__.py(NEW)

Create a basic test file to validate the project structure:

```python
"""Basic tests for PyContextify package."""

import pytest
from pycontextify import __version__, IndexManager, Config


def test_version():
    """Test that version is defined."""
    assert __version__ == "0.1.0"


def test_imports():
    """Test that main classes can be imported."""
    assert IndexManager is not None
    assert Config is not None


def test_config_creation():
    """Test that Config can be instantiated."""
    # This will test basic config creation once implemented
    # config = Config()
    # assert config is not None
    pass


def test_index_manager_creation():
    """Test that IndexManager can be instantiated."""
    # This will test basic manager creation once implemented
    # config = Config()
    # manager = IndexManager(config)
    # assert manager is not None
    pass


def test_relationship_store_import():
    """Test that RelationshipStore can be imported."""
    from pycontextify.index import RelationshipStore
    assert RelationshipStore is not None


def test_chunker_imports():
    """Test that all chunker types can be imported."""
    # This will test chunker imports once implemented
    # from pycontextify.index.chunker import CodeChunker, DocumentChunker, WebPageChunker
    # assert CodeChunker is not None
    # assert DocumentChunker is not None
    # assert WebPageChunker is not None
    pass


def test_simplified_mcp_interface():
    """Test that the simplified MCP interface is maintained."""
    # This will test that only the essential MCP functions are exposed
    # Expected functions: index_code, index_document, index_webpage, search, status, search_with_context
    # Removed functions: find_related, get_relationships (for simplicity)
    pass
```

This provides a foundation for testing the package structure and main components including the new relationship store, chunker types, and simplified MCP interface. More comprehensive tests should be added as the implementation progresses.