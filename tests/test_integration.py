"""Integration tests for PyContextify.

This module tests the complete pipeline from document processing to embedding generation,
ensuring the system works end-to-end with real data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pycontextify.index.config import Config
from pycontextify.index.manager import IndexManager
from pycontextify.index.metadata import SourceType


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
        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_file.write(sample_markdown_content)
            temp_file.flush()

            try:
                # Test document indexing
                result = index_manager.index_document(temp_file.name)

                # Verify indexing succeeded
                assert "error" not in result
                assert result["chunks_added"] > 0
                assert result["source_type"] == "document"

                print(f"Successfully indexed document: {result}")

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

            finally:
                # Clean up temporary file (Windows-safe)
                try:
                    Path(temp_file.name).unlink(missing_ok=True)
                except PermissionError:
                    # Windows file permission issue - ignore for test cleanup
                    pass

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

        for doc_type, content in test_documents:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, encoding="utf-8"
            ) as temp_file:
                temp_file.write(content)
                temp_file.flush()

                try:
                    result = index_manager.index_document(temp_file.name)
                    assert "error" not in result
                    assert result["chunks_added"] > 0
                    print(
                        f"✅ Successfully indexed {doc_type}: {result['chunks_added']} chunks"
                    )

                finally:
                    # Windows-safe cleanup
                    try:
                        Path(temp_file.name).unlink(missing_ok=True)
                    except PermissionError:
                        pass

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

    def test_code_file_embedding_generation(self, index_manager):
        """Test complete pipeline for code files: Python code → chunks → embeddings → search."""
        # Sample Python code content
        python_code = '''"""A sample Python module for testing code indexing.

This module demonstrates various Python constructs including classes,
functions, imports, and different coding patterns.
"""

import os
import sys
from typing import List, Dict, Optional
from pathlib import Path


class SearchEngine:
    """A semantic search engine implementation.
    
    This class provides methods for indexing documents and performing
    semantic searches using vector embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the search engine.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.index = {}
        self.documents = []
    
    def add_document(self, doc_id: str, content: str) -> bool:
        """Add a document to the search index."""
        try:
            processed_content = self._preprocess_text(content)
            embedding = self._generate_embedding(processed_content)
            
            self.index[doc_id] = {
                'content': content,
                'embedding': embedding,
                'metadata': self._extract_metadata(content)
            }
            
            self.documents.append(doc_id)
            return True
            
        except Exception as e:
            print(f"Error adding document {doc_id}: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for documents similar to the query."""
        if not self.documents:
            return []
        
        query_embedding = self._generate_embedding(query)
        results = []
        
        for doc_id in self.documents:
            doc_data = self.index[doc_id]
            similarity = self._calculate_similarity(
                query_embedding, 
                doc_data['embedding']
            )
            
            results.append({
                'doc_id': doc_id,
                'content': doc_data['content'][:200] + '...',
                'similarity': similarity,
                'metadata': doc_data['metadata']
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding generation."""
        return ' '.join(text.split()).lower()
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        import hashlib
        
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 
                    for i in range(0, min(32, len(hash_hex)), 2)]
        
        return embedding
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(a * a for a in vec2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _extract_metadata(self, content: str) -> Dict:
        """Extract metadata from content."""
        return {
            'length': len(content),
            'word_count': len(content.split()),
            'has_code': '```' in content or 'def ' in content,
            'language': 'python' if 'import ' in content else 'text'
        }


def main():
    """Main function to demonstrate the search engine."""
    engine = SearchEngine()
    
    engine.add_document("doc1", "Python is a programming language")
    engine.add_document("doc2", "Machine learning with neural networks")
    engine.add_document("doc3", "Web development using Flask and Django")
    
    results = engine.search("programming", top_k=3)
    
    print("Search results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['doc_id']}: {result['similarity']:.4f}")
        print(f"   {result['content']}")
        print()


if __name__ == "__main__":
    main()
'''

        # Create temporary Python file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_file.write(python_code)
            temp_file.flush()

            try:
                # Test code indexing
                result = index_manager.index_codebase(str(Path(temp_file.name).parent))

                # Verify indexing succeeded
                assert "error" not in result
                assert result["files_processed"] > 0
                assert result["chunks_added"] > 0
                assert result["source_type"] == "code"

                print(f"✅ Successfully indexed code file: {result}")

                # Get status to verify embeddings were created
                status = index_manager.get_status()
                assert status["metadata"]["total_chunks"] > 0
                assert status["vector_store"]["total_vectors"] > 0

                # Test code-specific semantic search
                code_queries = [
                    ("search engine class", "Should find SearchEngine class"),
                    ("vector similarity calculation", "Should find similarity methods"),
                    ("document indexing", "Should find indexing functionality"),
                    ("machine learning embeddings", "Should find embedding generation"),
                ]

                for query, description in code_queries:
                    response = index_manager.search(query, top_k=3)

                    assert (
                        response.success
                    ), f"Search should succeed for '{query}': {response.error}"

                    if response.results:
                        result = response.results[0]
                        score = result.relevance_score
                        # For mocked embeddings, just check that we get a valid numeric score
                        import math

                        import numpy as np

                        assert isinstance(
                            score, (int, float, np.integer, np.floating)
                        ), f"Score should be numeric for '{query}': {score}"
                        assert math.isfinite(
                            score
                        ), f"Score should be finite for '{query}': {score}"
                        print(f"✅ Code query '{query}': score {score:.4f}")
                    else:
                        print(f"⚠️  Code query '{query}': No results found")

            finally:
                # Clean up temporary file (Windows-safe)
                try:
                    Path(temp_file.name).unlink(missing_ok=True)
                except PermissionError:
                    pass

    def test_webpage_embedding_generation(self, index_manager):
        """Test complete pipeline for webpage-like content: Markdown → chunks → embeddings → search."""
        # Sample webpage-like content in Markdown format (equivalent to HTML content)
        webpage_content = """# Machine Learning Tutorial: A Complete Guide

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

        # Create temporary Markdown file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_file.write(webpage_content)
            temp_file.flush()

            try:
                # Test webpage-like content indexing
                result = index_manager.index_document(temp_file.name)

                # Verify indexing succeeded
                assert "error" not in result
                assert result["chunks_added"] > 0
                assert result["source_type"] == "document"

                print(f"✅ Successfully indexed webpage content: {result}")

                # Test webpage-specific semantic search
                webpage_queries = [
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

                for query, description in webpage_queries:
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
                        print(f"✅ Webpage query '{query}': score {score:.4f}")
                    else:
                        print(f"⚠️  Webpage query '{query}': No results found")

            finally:
                # Clean up temporary file (Windows-safe)
                try:
                    Path(temp_file.name).unlink(missing_ok=True)
                except PermissionError:
                    pass


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
                # Create and index a test document
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False, encoding="utf-8"
                ) as temp_file:
                    temp_file.write(sample_markdown_content)
                    temp_file.flush()

                    # Index the document
                    result = manager.index_document(temp_file.name)
                    assert "error" not in result

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
                    print(
                        f"   - Total vectors: {status['vector_store']['total_vectors']}"
                    )
                    print(
                        f"   - Memory usage: {status['performance']['memory_usage_mb']:.2f} MB"
                    )

                    # Clean up temp file (Windows-safe)
                    try:
                        Path(temp_file.name).unlink(missing_ok=True)
                    except PermissionError:
                        pass

            finally:
                # Cleanup
                try:
                    manager.clear_index()
                    if hasattr(manager, "embedder") and manager.embedder:
                        manager.embedder.cleanup()
                except Exception:
                    pass
