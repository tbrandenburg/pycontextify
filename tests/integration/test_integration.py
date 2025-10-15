"""Integration tests for PyContextify.

This module tests the complete pipeline from document processing to embedding generation,
ensuring the system works end-to-end with real data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pycontextify.config import Config
from pycontextify.indexer import IndexManager
from pycontextify.types import SourceType


class TestEmbeddingGeneration:
    """Test embedding generation for documents."""

    @pytest.fixture
    def sample_markdown_content(self):
        """Sample markdown content for testing."""
        return """# Python Programming Guide

## Introduction

Python is a high-level, interpreted programming language with dynamic semantics.
Its high-level built-in data structures, combined with dynamic typing and dynamic
binding, make it very attractive for rapid application development.

## Key Features

### Simple and Easy to Learn
Python has a simple syntax similar to the English language. It allows developers
to write programs with fewer lines of code than many other programming languages.

### Object-Oriented Programming
Python supports object-oriented programming (OOP) concepts like classes, objects,
inheritance, and polymorphism. This makes code more modular and reusable.

### Large Standard Library
Python comes with a vast collection of modules and packages in its standard library,
which means you don't have to write code for every single thing from scratch.

## Code Examples

Here's a simple example of Python code:

```python
def greet(name):
    return f"Hello, {name}!"

# Usage
message = greet("World")
print(message)
```

## Conclusion

Python is an excellent choice for beginners and experienced developers alike.
Its simplicity and powerful features make it suitable for various applications,
from web development to data science and artificial intelligence.
"""

    @pytest.fixture
    def config(self):
        """Test configuration with minimal settings."""
        # Create a temporary directory for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            # Override settings for testing
            config.index_dir = Path(temp_dir)
            config.auto_persist = False  # Don't persist for tests
            config.auto_load = False  # Don't try to load existing
            config.embedding_provider = "sentence_transformers"
            config.embedding_model = "all-MiniLM-L6-v2"  # Faster for testing
            config.chunk_size = 256  # Smaller chunks for testing
            config.chunk_overlap = 25
            config.enable_relationships = True
            yield config

    @pytest.fixture
    def index_manager(self, config):
        """Create IndexManager for testing."""
        manager = IndexManager(config)
        yield manager
        # Cleanup
        try:
            manager.clear_index()
            if hasattr(manager, "embedder") and manager.embedder:
                manager.embedder.cleanup()
        except Exception:
            pass  # Ignore cleanup errors in tests

    def test_markdown_document_embedding_generation(
        self, sample_markdown_content, index_manager
    ):
        """Test complete pipeline: markdown → chunks → embeddings → search."""
        # Create temporary directory with markdown file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            doc_file = temp_path / "python_guide.md"
            doc_file.write_text(sample_markdown_content, encoding="utf-8")

            try:
                # Test document indexing with new unified API
                result = index_manager.index_filebase(
                    base_path=str(temp_path), tags="documents"
                )

                # Verify indexing succeeded with new stats structure
                assert "error" not in result
                assert result["files_loaded"] > 0
                assert result["chunks_created"] > 0
                assert result["vectors_embedded"] > 0

                print(f"Successfully indexed filebase: {result}")

                # Get status to verify embeddings were created
                status = index_manager.get_status()
                assert status["metadata"]["total_chunks"] > 0
                assert status["vector_store"]["total_vectors"] > 0

                # Test that embeddings have correct properties
                self._verify_embeddings_properties(index_manager, status)

                # Test semantic search functionality
                self._test_semantic_search(index_manager)

                # Test relationship-aware search
                self._test_relationship_search(index_manager)

            except Exception as e:
                print(f"Test failed: {e}")
                raise

    def _verify_embeddings_properties(self, index_manager, status):
        """Verify that generated embeddings have expected properties."""
        # Check embedding dimensions
        embedder = index_manager.embedder
        expected_dim = embedder.get_dimension()

        # For all-MiniLM-L6-v2, dimension should be 384
        if "MiniLM-L6-v2" in embedder.model_name:
            assert expected_dim == 384, f"Expected 384 dimensions, got {expected_dim}"

        # Check that vector store has same dimension
        assert index_manager.vector_store.dimension == expected_dim

        # Verify embeddings are normalized (length close to 1.0 for cosine similarity)
        sample_chunks = index_manager.metadata_store.get_all_chunks()
        if sample_chunks:
            sample_chunk = sample_chunks[0]

            # Get embedding for this chunk
            embedding = index_manager.embedder.embed_single(sample_chunk.chunk_text)

            # Check embedding properties
            assert embedding.shape == (
                expected_dim,
            ), f"Wrong embedding shape: {embedding.shape}"
            assert embedding.dtype == np.float32, f"Wrong dtype: {embedding.dtype}"

            # Check if normalized (length should be close to 1.0)
            norm = np.linalg.norm(embedding)
            assert 0.9 <= norm <= 1.1, f"Embedding not normalized, norm: {norm}"

            # Check for reasonable value distribution (no all zeros, no extreme values)
            assert not np.allclose(embedding, 0), "Embedding is all zeros"
            assert np.all(np.abs(embedding) <= 1.0), "Embedding has extreme values"

            print(f"✅ Embedding verification passed:")
            print(f"   - Shape: {embedding.shape}")
            print(f"   - Dtype: {embedding.dtype}")
            print(f"   - Norm: {norm:.4f}")
            print(f"   - Value range: [{embedding.min():.4f}, {embedding.max():.4f}]")

    def _test_semantic_search(self, index_manager):
        """Test semantic search functionality with various queries."""
        test_queries = [
            ("Python programming", "Should find content about Python"),
            ("object-oriented", "Should find OOP section"),
            ("standard library", "Should find library content"),
            ("code example", "Should find code examples"),
            ("artificial intelligence", "Should find AI mention in conclusion"),
        ]

        for query, description in test_queries:
            response = index_manager.search(query, top_k=3)

            # Basic checks
            assert hasattr(
                response, "success"
            ), f"Response should be SearchResponse for query '{query}'"
            assert (
                response.success
            ), f"Search should succeed for query '{query}': {response.error}"
            assert len(response.results) <= 3, f"Too many results for query '{query}'"

            if response.results:  # If we got results
                result = response.results[0]

                # Check result structure (new format)
                assert hasattr(result, "text"), f"Missing text in result for '{query}'"
                assert hasattr(
                    result, "relevance_score"
                ), f"Missing relevance_score in result for '{query}'"
                assert hasattr(
                    result, "source_path"
                ), f"Missing source_path in result for '{query}'"

                # Check similarity score is reasonable
                score = result.relevance_score
                # Different embedding models use different score ranges
                # Some use [0,1], others use [-1,1], and some use arbitrary ranges
                import numpy as np

                assert isinstance(
                    score, (int, float, np.integer, np.floating)
                ), f"Score should be numeric for '{query}': {score}"

                # For semantic search, we expect the score to exist and be finite
                import math

                assert math.isfinite(
                    score
                ), f"Score should be finite for '{query}': {score}"

                print(
                    f"✅ Query '{query}': {len(response.results)} results, best score: {score:.4f}"
                )
            else:
                print(f"⚠️  Query '{query}': No results found")

    def _test_relationship_search(self, index_manager):
        """Test relationship-aware search functionality."""
        # Test basic search with relationship metadata
        query = "Python programming language"

        # Basic search
        basic_response = index_manager.search(query, top_k=5)
        assert hasattr(basic_response, "success")

        print(f"✅ Relationship search test:")
        print(f"   - Basic search: {len(basic_response.results)} results")

        # Verify relationship metadata if chunks have it
        sample_chunks = index_manager.metadata_store.get_all_chunks()
        if sample_chunks and index_manager.config.enable_relationships:
            chunk_with_refs = None
            for chunk in sample_chunks:
                if chunk.references or chunk.tags:
                    chunk_with_refs = chunk
                    break

            if chunk_with_refs:
                print(f"✅ Found chunk with relationships:")
                print(f"   - References: {len(chunk_with_refs.references)}")
                print(f"   - Tags: {len(chunk_with_refs.tags)}")

                if chunk_with_refs.references:
                    print(f"   - Sample references: {chunk_with_refs.references[:3]}")

    @pytest.mark.slow
    def test_multiple_document_types(self, index_manager):
        """Test with different types of content to verify embedding consistency."""
        test_documents = [
            (
                "Technical documentation",
                """
# API Documentation

## Authentication
All API requests require authentication via API key.

## Endpoints
- GET /users: Retrieve user list
- POST /users: Create new user
""",
            ),
            (
                "Narrative text",
                """
Once upon a time, in a land far away, there lived a young developer
who dreamed of creating the perfect search system. She worked day and night,
experimenting with different algorithms and approaches.
""",
            ),
            (
                "Code-heavy content",
                """
```python
class SearchEngine:
    def __init__(self, embedder):
        self.embedder = embedder
        
    def index_document(self, text):
        return self.embedder.embed(text)
```
""",
            ),
        ]

        # Create temp directory for all test documents
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for idx, (doc_type, content) in enumerate(test_documents):
                doc_file = temp_path / f"doc_{idx}.md"
                doc_file.write_text(content, encoding="utf-8")

            # Index all documents at once using unified API
            result = index_manager.index_filebase(
                base_path=str(temp_path), tags="mixed_documents"
            )

            assert "error" not in result
            assert result["files_loaded"] == len(test_documents)
            assert result["chunks_created"] > 0
            print(
                f"✅ Successfully indexed {result['files_loaded']} documents: {result['chunks_created']} chunks"
            )

            # After indexing multiple documents, test cross-document search
            response = index_manager.search("API documentation", top_k=5)
            assert response.success, f"Search should succeed: {response.error}"
            assert (
                len(response.results) > 0
            ), "Should find results across different document types"

            final_status = index_manager.get_status()
            print(
                f"✅ Final index status: {final_status['metadata']['total_chunks']} chunks, "
                f"{final_status['vector_store']['total_vectors']} vectors"
            )

        # Code ingestion flows are covered by the MCP server tests to avoid
        # re-indexing large temporary directories in this integration suite.

    def test_reference_guide_embedding_generation(self, index_manager):
        """Test complete pipeline for supplemental guide content: Markdown → chunks → embeddings → search."""
        # Sample supplemental guide content in Markdown format (equivalent to HTML content)
        guide_content = """# Machine Learning Tutorial: A Complete Guide

**Meta**: Comprehensive guide to machine learning concepts and algorithms

## Introduction to Machine Learning

Machine learning is a subset of artificial intelligence (AI) that provides systems 
the ability to automatically learn and improve from experience without being explicitly 
programmed. It focuses on the development of computer programs that can access data and 
use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct 
experience, or instruction, in order to look for patterns in data and make better decisions 
in the future based on the examples that we provide.

## Popular Machine Learning Algorithms

### Supervised Learning Algorithms

- **Linear Regression**: Used for predicting continuous values
- **Logistic Regression**: Used for binary classification problems  
- **Decision Trees**: Easy to understand tree-like model of decisions
- **Random Forest**: Ensemble method using multiple decision trees
- **Support Vector Machines**: Effective for high-dimensional spaces

### Unsupervised Learning Algorithms

- **K-Means Clustering**: Groups data into k clusters
- **Hierarchical Clustering**: Creates tree of clusters
- **Principal Component Analysis (PCA)**: Dimensionality reduction

## Neural Networks and Deep Learning

Neural networks are computing systems vaguely inspired by the biological neural networks 
that constitute animal brains. They are a set of algorithms, modeled loosely after the human brain, 
that are designed to recognize patterns.

### Types of Neural Networks

- **Feedforward Neural Networks**: Information moves in one direction
- **Convolutional Neural Networks (CNNs)**: Great for image processing
- **Recurrent Neural Networks (RNNs)**: Can use their internal memory
- **Long Short-Term Memory (LSTM)**: Special RNN for sequence problems

Deep learning is a subset of machine learning that uses neural networks with multiple 
layers (deep neural networks) to progressively extract higher-level features from raw input.

## Practical Examples and Applications

### Common Applications

- Image Recognition and Computer Vision
- Natural Language Processing (NLP)
- Recommendation Systems
- Fraud Detection
- Autonomous Vehicles
- Medical Diagnosis
- Financial Trading

### Getting Started Code Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
data = pd.read_csv('your_data.csv')

# Prepare features and target
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [Kaggle Learn](https://www.kaggle.com/learn)

---

**Contact**: info@mltutorial.com  
**Copyright**: 2024 Machine Learning Tutorial. All rights reserved.
"""

        # Create temporary directory with guide file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            guide_file = temp_path / "ml_guide.md"
            guide_file.write_text(guide_content, encoding="utf-8")

            try:
                # Test supplemental guide content indexing
                result = index_manager.index_filebase(
                    base_path=str(temp_path), tags="guides"
                )

                # Verify indexing succeeded
                assert "error" not in result
                assert result["files_loaded"] > 0
                assert result["chunks_created"] > 0

                print(f"✅ Successfully indexed guide content: {result}")

                # Test guide-specific semantic search
                guide_queries = [
                    ("machine learning tutorial", "Should find ML tutorial content"),
                    (
                        "neural networks deep learning",
                        "Should find neural network sections",
                    ),
                    (
                        "supervised learning algorithms",
                        "Should find supervised learning content",
                    ),
                    ("code example python", "Should find code examples"),
                    ("regression classification", "Should find algorithm descriptions"),
                ]

                for query, description in guide_queries:
                    response = index_manager.search(query, top_k=3)

                    assert (
                        response.success
                    ), f"Search should succeed for '{query}': {response.error}"

                    if response.results:
                        result = response.results[0]
                        score = result.relevance_score
                        # Just check that we get a valid numeric score
                        import math

                        import numpy as np

                        assert isinstance(
                            score, (int, float, np.integer, np.floating)
                        ), f"Score should be numeric for '{query}': {score}"
                        assert math.isfinite(
                            score
                        ), f"Score should be finite for '{query}': {score}"
                        print(f"✅ Guide query '{query}': score {score:.4f}")
                    else:
                        print(f"⚠️  Guide query '{query}': No results found")

            except Exception as e:
                print(f"Test failed: {e}")
                raise


@pytest.mark.integration
class TestFullPipeline:
    """Test the complete MCP server pipeline."""

    def test_mcp_status_after_indexing(self, sample_markdown_content=None):
        """Test MCP status function returns proper information after indexing."""
        if not sample_markdown_content:
            sample_markdown_content = """# Test Document
            
This is a simple test document for checking MCP functionality.
It contains some basic content for indexing and search testing.
"""

        # Use a more isolated config for this test
        config = Config()
        config.auto_persist = False
        config.auto_load = False
        config.embedding_model = "all-MiniLM-L6-v2"  # Faster model for testing

        with tempfile.TemporaryDirectory() as temp_dir:
            config.index_dir = Path(temp_dir)

            manager = IndexManager(config)
            try:
                # Create temp directory and test document
                doc_dir = Path(temp_dir) / "docs"
                doc_dir.mkdir()
                doc_file = doc_dir / "test.md"
                doc_file.write_text(sample_markdown_content, encoding="utf-8")

                # Index the document using unified API
                result = manager.index_filebase(
                    base_path=str(doc_dir), tags="test_documents"
                )
                assert "error" not in result
                assert result["files_loaded"] > 0

                # Test status
                status = manager.get_status()

                # Verify status structure
                assert "metadata" in status
                assert "vector_store" in status
                assert "embedding" in status
                assert "performance" in status
                assert "memory_usage_mb" in status["performance"]

                # Verify content was indexed
                assert status["metadata"]["total_chunks"] > 0
                assert status["vector_store"]["total_vectors"] > 0
                assert status["embedding"]["provider"] == "sentence_transformers"

                print(f"✅ MCP Status test passed:")
                print(f"   - Total chunks: {status['metadata']['total_chunks']}")
                print(f"   - Total vectors: {status['vector_store']['total_vectors']}")
                print(
                    f"   - Memory usage: {status['performance']['memory_usage_mb']:.2f} MB"
                )

            finally:
                # Cleanup
                try:
                    manager.clear_index()
                    if hasattr(manager, "embedder") and manager.embedder:
                        manager.embedder.cleanup()
                except Exception:
                    pass
