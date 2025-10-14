"""End-to-end user flow integration test for PyContextify.

This test verifies the complete user workflow:
1. Getting initial status
2. Indexing a document
3. Indexing a supplemental guide (markdown)
4. Indexing a codebase
5. Getting status after indexing
6. Performing searches
7. Resetting the index
8. Verifying clean state after reset
"""

import tempfile
from pathlib import Path

import pytest

from pycontextify.config import Config
from pycontextify.indexer import IndexManager


@pytest.mark.integration
class TestCompleteUserFlow:
    """Test the complete user workflow from start to finish."""

    @pytest.fixture
    def setup_test_environment(self):
        """Set up a temporary test environment with sample files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create sample document
            doc_path = temp_path / "sample_document.md"
            doc_path.write_text(
                """# API Documentation

## Overview
This is a REST API for managing user accounts.

## Authentication
All requests require an API key in the header:
```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### GET /users
Retrieve a list of all users.

### POST /users
Create a new user account.

### PUT /users/{id}
Update an existing user.

### DELETE /users/{id}
Delete a user account.
""",
                encoding="utf-8",
            )

            # Create supplemental guide content (markdown simulating an external reference)
            guide_path = temp_path / "reference_guide.md"
            guide_path.write_text(
                """# Python Best Practices Guide

## Introduction
Python is a powerful programming language that emphasizes code readability.

## Code Style Guidelines
- Use 4 spaces for indentation
- Follow PEP 8 style guide
- Write descriptive variable names
- Add docstrings to functions

## Common Patterns
### Context Managers
Use context managers for resource management:
```python
with open('file.txt', 'r') as f:
    data = f.read()
```

### List Comprehensions
Prefer list comprehensions for simple transformations:
```python
squares = [x**2 for x in range(10)]
```

## Testing
Always write tests for your code:
- Unit tests for individual functions
- Integration tests for workflows
- Use pytest for testing

## Conclusion
Following best practices makes your code more maintainable.
""",
                encoding="utf-8",
            )

            # Create sample codebase
            code_dir = temp_path / "sample_codebase"
            code_dir.mkdir()

            # Python module
            (code_dir / "utils.py").write_text(
                """\"\"\"Utility functions for data processing.\"\"\"

def process_data(data):
    \"\"\"Process incoming data and return results.
    
    Args:
        data: Input data to process
        
    Returns:
        Processed data dictionary
    \"\"\"
    if not data:
        return {}
    
    return {
        'processed': True,
        'data': data,
        'count': len(data)
    }


def validate_input(value):
    \"\"\"Validate input value.
    
    Args:
        value: Value to validate
        
    Returns:
        True if valid, False otherwise
    \"\"\"
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


class DataProcessor:
    \"\"\"Data processor class for batch operations.\"\"\"
    
    def __init__(self, config):
        self.config = config
        self.processed_count = 0
    
    def process_batch(self, batch):
        \"\"\"Process a batch of items.
        
        Args:
            batch: List of items to process
            
        Returns:
            List of processed items
        \"\"\"
        results = []
        for item in batch:
            if validate_input(item):
                results.append(process_data(item))
                self.processed_count += 1
        return results
""",
                encoding="utf-8",
            )

            # JavaScript module
            (code_dir / "api.js").write_text(
                """/**
 * API client for making HTTP requests
 */
class ApiClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
    }

    /**
     * Make a GET request
     * @param {string} endpoint - API endpoint
     * @returns {Promise} Response data
     */
    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            headers: {
                'Authorization': `Bearer ${this.apiKey}`
            }
        });
        return response.json();
    }

    /**
     * Make a POST request
     * @param {string} endpoint - API endpoint
     * @param {object} data - Request body
     * @returns {Promise} Response data
     */
    async post(endpoint, data) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        return response.json();
    }
}

module.exports = ApiClient;
""",
                encoding="utf-8",
            )

            # Configuration with temp directory for index
            config = Config()
            config.index_dir = temp_path / "pycontextify_index"
            config.index_dir.mkdir()
            config.auto_persist = False
            config.auto_load = False
            config.embedding_model = "all-MiniLM-L6-v2"  # Faster for testing
            config.chunk_size = 256
            config.chunk_overlap = 25
            config.enable_relationships = True

            yield {
                "temp_path": temp_path,
                "doc_path": doc_path,
                "guide_path": guide_path,
                "code_dir": code_dir,
                "config": config,
            }

    def test_complete_user_workflow(self, setup_test_environment):
        """Test the complete user workflow from initialization to reset."""
        env = setup_test_environment
        
        print("\n" + "="*70)
        print("🚀 STARTING COMPLETE USER FLOW TEST")
        print("="*70)

        # Initialize the IndexManager
        manager = IndexManager(env["config"])

        try:
            # ================================================================
            # STEP 1: Get initial status (empty index)
            # ================================================================
            print("\n📊 STEP 1: Getting initial status...")
            
            initial_status = manager.get_status()
            
            # Verify initial state
            assert initial_status["metadata"]["total_chunks"] == 0, \
                "Index should be empty initially"
            assert initial_status["vector_store"]["total_vectors"] == 0, \
                "Vector store should be empty initially"
            assert initial_status["embedding"]["provider"] == "sentence_transformers", \
                "Should use sentence_transformers by default"
            
            print(f"✅ Initial status verified:")
            print(f"   - Total chunks: {initial_status['metadata']['total_chunks']}")
            print(f"   - Total vectors: {initial_status['vector_store']['total_vectors']}")
            print(f"   - Embedding provider: {initial_status['embedding']['provider']}")
            print(f"   - Embedding model: {initial_status['embedding']['model']}")

            # ================================================================
            # STEP 2: Index a document
            # ================================================================
            print("\n📄 STEP 2: Indexing document...")
            
            # Create temp dir for doc
            doc_dir = env["temp_path"] / "docs"
            doc_dir.mkdir()
            doc_file = doc_dir / "sample_document.md"
            doc_file.write_text(env["doc_path"].read_text(encoding="utf-8"), encoding="utf-8")
            
            doc_result = manager.index_filebase(
                base_path=str(doc_dir),
                topic="api_documentation"
            )
            
            # Verify document indexing with new stats
            assert "error" not in doc_result, f"Document indexing failed: {doc_result}"
            assert doc_result["files_loaded"] > 0, "Should have loaded files"
            assert doc_result["chunks_created"] > 0, "Should have added chunks from document"
            
            print(f"✅ Document indexed successfully:")
            print(f"   - Files loaded: {doc_result['files_loaded']}")
            print(f"   - Chunks created: {doc_result['chunks_created']}")
            print(f"   - Topic: {doc_result['topic']}")

            # ================================================================
            # STEP 3: Index supplemental guide content (simulated with markdown)
            # ================================================================
            print("\n📘 STEP 3: Indexing supplemental guide content...")

            # Create temp dir for guide
            guide_dir = env["temp_path"] / "guides"
            guide_dir.mkdir()
            guide_file = guide_dir / "reference_guide.md"
            guide_file.write_text(env["guide_path"].read_text(encoding="utf-8"), encoding="utf-8")

            guide_result = manager.index_filebase(
                base_path=str(guide_dir),
                topic="best_practices_guides"
            )

            # Verify guide indexing
            assert "error" not in guide_result, f"Guide indexing failed: {guide_result}"
            assert guide_result["chunks_created"] > 0, "Should have added chunks from the guide"

            print(f"✅ Guide content indexed successfully:")
            print(f"   - Files loaded: {guide_result['files_loaded']}")
            print(f"   - Chunks created: {guide_result['chunks_created']}")
            print(f"   - Topic: {guide_result['topic']}")

            # ================================================================
            # STEP 4: Index codebase
            # ================================================================
            print("\n💻 STEP 4: Indexing codebase...")
            
            code_result = manager.index_filebase(
                base_path=str(env["code_dir"]),
                topic="codebase"
            )
            
            # Verify codebase indexing
            assert "error" not in code_result, f"Codebase indexing failed: {code_result}"
            assert code_result["files_loaded"] > 0, "Should have processed code files"
            assert code_result["chunks_created"] > 0, "Should have added chunks from code"
            
            print(f"✅ Codebase indexed successfully:")
            print(f"   - Directory: {env['code_dir'].name}")
            print(f"   - Files loaded: {code_result['files_loaded']}")
            print(f"   - Chunks created: {code_result['chunks_created']}")
            print(f"   - Topic: {code_result['topic']}")

            # ================================================================
            # STEP 5: Get status after indexing
            # ================================================================
            print("\n📊 STEP 5: Getting status after indexing...")
            
            post_index_status = manager.get_status()
            
            # Calculate expected totals
            total_chunks_expected = (
                doc_result["chunks_created"] +
                guide_result["chunks_created"] +
                code_result["chunks_created"]
            )
            
            # Verify status after indexing
            assert post_index_status["metadata"]["total_chunks"] == total_chunks_expected, \
                f"Expected {total_chunks_expected} chunks, got {post_index_status['metadata']['total_chunks']}"
            assert post_index_status["vector_store"]["total_vectors"] == total_chunks_expected, \
                "Vector count should match chunk count"
            assert post_index_status["metadata"]["total_chunks"] > initial_status["metadata"]["total_chunks"], \
                "Chunk count should have increased"
            
            print(f"✅ Post-indexing status verified:")
            print(f"   - Total chunks: {post_index_status['metadata']['total_chunks']}")
            print(f"   - Total vectors: {post_index_status['vector_store']['total_vectors']}")
            print(f"   - Documents: {post_index_status['metadata']['source_types'].get('document', 0)}")
            print(f"   - Code: {post_index_status['metadata']['source_types'].get('code', 0)}")
            print(f"   - Memory usage: {post_index_status['performance']['memory_usage_mb']:.2f} MB")

            # ================================================================
            # STEP 6: Perform searches
            # ================================================================
            print("\n🔍 STEP 6: Performing searches...")
            
            # Define test queries covering all indexed content
            test_queries = [
                ("API documentation", "Should find document content"),
                ("authentication Bearer token", "Should find auth section"),
                ("Python best practices", "Should find supplemental guide content"),
                ("list comprehension", "Should find Python patterns"),
                ("process_data function", "Should find code content"),
                ("DataProcessor class", "Should find Python class"),
                ("ApiClient fetch", "Should find JavaScript code"),
            ]
            
            search_results = {}
            for query, description in test_queries:
                response = manager.search(query, top_k=3)
                
                # Verify search response
                assert response.success, f"Search failed for '{query}': {response.error}"
                assert hasattr(response, "results"), f"Response should have results for '{query}'"
                
                # Store results
                search_results[query] = {
                    "count": len(response.results),
                    "description": description,
                }
                
                if response.results:
                    top_result = response.results[0]
                    search_results[query]["top_score"] = top_result.relevance_score
                    search_results[query]["source_type"] = top_result.source_type
                    
                    print(f"✅ Query: '{query}'")
                    print(f"   - Results: {len(response.results)}")
                    print(f"   - Top score: {top_result.relevance_score:.4f}")
                    print(f"   - Source type: {top_result.source_type}")
                    print(f"   - Source: {Path(top_result.source_path).name}")
                else:
                    print(f"⚠️  Query: '{query}' - No results found")
            
            # Verify we got results for most queries
            queries_with_results = sum(1 for r in search_results.values() if r["count"] > 0)
            assert queries_with_results >= len(test_queries) * 0.7, \
                f"Expected results for most queries, got {queries_with_results}/{len(test_queries)}"
            
            print(f"\n✅ Search functionality verified:")
            print(f"   - Queries tested: {len(test_queries)}")
            print(f"   - Queries with results: {queries_with_results}")
            print(f"   - Success rate: {queries_with_results/len(test_queries)*100:.1f}%")

            # ================================================================
            # STEP 7: Reset the index
            # ================================================================
            print("\n🔄 STEP 7: Resetting index...")
            
            reset_result = manager.clear_index(remove_files=True)
            
            # Verify reset
            assert reset_result["success"] is True, "Reset should succeed"
            
            print(f"✅ Index reset successfully:")
            print(f"   - Success: {reset_result['success']}")
            print(f"   - Message: {reset_result.get('message', 'Index cleared')}")

            # ================================================================
            # STEP 8: Verify clean state after reset
            # ================================================================
            print("\n🧹 STEP 8: Verifying clean state after reset...")
            
            post_reset_status = manager.get_status()
            
            # Verify everything is back to zero
            assert post_reset_status["metadata"]["total_chunks"] == 0, \
                "Chunks should be zero after reset"
            assert post_reset_status["vector_store"]["total_vectors"] == 0, \
                "Vectors should be zero after reset"
            
            print(f"✅ Clean state verified:")
            print(f"   - Total chunks: {post_reset_status['metadata']['total_chunks']}")
            print(f"   - Total vectors: {post_reset_status['vector_store']['total_vectors']}")
            
            # Try searching after reset - should return empty or error
            empty_search = manager.search("API documentation", top_k=5)
            # After reset, search may return error (no content) or success with empty results
            if empty_search.success:
                assert len(empty_search.results) == 0, "Should have no results after reset"
                print(f"✅ Search after reset returns no results (as expected)")
            else:
                # It's acceptable to return an error when index is empty
                assert "no indexed content" in empty_search.error.lower() or "empty" in empty_search.error.lower(), \
                    f"Expected empty index error, got: {empty_search.error}"
                print(f"✅ Search after reset returns error (as expected): {empty_search.error}")

            # ================================================================
            # FINAL SUMMARY
            # ================================================================
            print("\n" + "="*70)
            print("✅ COMPLETE USER FLOW TEST PASSED!")
            print("="*70)
            print("\nAll steps completed successfully:")
            print("  1. ✅ Initial status retrieved")
            print("  2. ✅ Document indexed")
            print("  3. ✅ Webpage content indexed")
            print("  4. ✅ Codebase indexed")
            print("  5. ✅ Status verified after indexing")
            print("  6. ✅ Search functionality validated")
            print("  7. ✅ Index reset completed")
            print("  8. ✅ Clean state verified")
            print("\n🎉 User flow test completed successfully!\n")

        finally:
            # Cleanup
            try:
                manager.clear_index(remove_files=True)
                if hasattr(manager, "embedder") and manager.embedder:
                    manager.embedder.cleanup()
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")
