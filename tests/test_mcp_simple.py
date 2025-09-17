"""Simplified end-to-end tests for PyContextify MCP server functions.

This module tests all MCP functions directly using the function-based FastMCP approach.
"""

import tempfile
import pytest
from pathlib import Path

# Import MCP functions directly
from pycontextify import mcp_server
from pycontextify.index.config import Config
from pycontextify.index.manager import IndexManager


@pytest.fixture
def mcp_isolated():
    """Setup isolated MCP environment for testing."""
    # Store original manager
    original_manager = mcp_server.manager
    
    try:
        # Use isolated temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Reset global manager
            mcp_server.manager = None
            
            # Create test config
            config = Config()
            config.index_dir = Path(temp_dir)
            config.auto_persist = False
            config.auto_load = False
            config.embedding_model = "all-MiniLM-L6-v2"  # Faster for testing
            
            # Initialize isolated manager
            test_manager = IndexManager(config)
            mcp_server.manager = test_manager
            
            yield mcp_server
            
    finally:
        # Cleanup
        try:
            if mcp_server.manager:
                mcp_server.manager.clear_index()
                if hasattr(mcp_server.manager, 'embedder') and mcp_server.manager.embedder:
                    mcp_server.manager.embedder.cleanup()
        except Exception:
            pass
        finally:
            # Restore original manager
            mcp_server.manager = original_manager


class TestMCPFunctions:
    """Test all MCP server functions."""
    
    def test_status_function(self, mcp_isolated):
        """Test the status MCP function."""
        status = mcp_isolated.status.fn()
        
        # Verify basic structure
        assert "metadata" in status
        assert "vector_store" in status
        assert "embedding" in status
        assert "mcp_server" in status
        
        # Initially empty
        assert status["metadata"]["total_chunks"] == 0
        assert status["vector_store"]["total_vectors"] == 0
        
        # MCP server info
        assert status["mcp_server"]["name"] == "PyContextify"
        assert len(status["mcp_server"]["mcp_functions"]) == 7  # Now includes reset_index
        
        print("✅ Status function working correctly")
    
    def test_index_document_function(self, mcp_isolated):
        """Test the index_document MCP function."""
        # Create test document
        test_content = """# Test Document
        
This is a test document for MCP function testing.
It contains some content for semantic search validation.

## Features
- Document indexing
- Semantic search capabilities
- Vector embeddings
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            f.flush()
            
            try:
                # Test indexing
                result = mcp_isolated.index_document.fn(f.name)
                
                # Verify result
                assert "error" not in result
                assert result["chunks_added"] > 0
                assert result["source_type"] == "document"
                
                chunks_added = result["chunks_added"]
                print(f"✅ index_document: {chunks_added} chunks added")
                
                # Verify status updated
                status = mcp_isolated.status.fn()
                assert status["metadata"]["total_chunks"] == chunks_added
                assert status["vector_store"]["total_vectors"] == chunks_added
                
            finally:
                try:
                    Path(f.name).unlink(missing_ok=True)
                except PermissionError:
                    pass
    
    def test_index_code_function(self, mcp_isolated):
        """Test the index_code MCP function."""
        # Create test codebase
        with tempfile.TemporaryDirectory() as temp_dir:
            code_dir = Path(temp_dir)
            
            # Create Python file
            (code_dir / "main.py").write_text("""
class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process_data(self, input_data):
        '''Process input data and return results.'''
        return [item * 2 for item in input_data]
    
    def save_results(self, results, filename):
        '''Save processing results to file.'''
        with open(filename, 'w') as f:
            for result in results:
                f.write(f"{result}\\n")
""")
            
            # Create config file
            (code_dir / "config.json").write_text("""{
    "database": {
        "host": "localhost",
        "port": 5432
    },
    "processing": {
        "batch_size": 100,
        "max_workers": 4
    }
}""")
            
            # Test codebase indexing
            result = mcp_isolated.index_code.fn(str(code_dir))
            
            # Verify result
            assert "error" not in result
            assert result["files_processed"] >= 2  # At least main.py and config.json
            assert result["chunks_added"] > 0
            assert result["source_type"] == "code"
            
            print(f"✅ index_code: {result['files_processed']} files, {result['chunks_added']} chunks")
    
    def test_search_function(self, mcp_isolated):
        """Test the search MCP function."""
        # First index some content
        test_docs = {
            "api_doc.md": """# API Documentation
            
## Authentication
Use bearer tokens for API authentication.

### Endpoints
- GET /users - List users
- POST /users - Create user
""",
            "user_code.txt": """
class UserManager:
    def authenticate_user(self, token):
        '''Authenticate user with bearer token.'''
        return validate_token(token)
    
    def create_user(self, username, email):
        '''Create new user account.'''
        return {'username': username, 'email': email}
"""
        }
        
        for filename, content in test_docs.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix=Path(filename).suffix, delete=False, encoding='utf-8') as f:
                f.write(content)
                f.flush()
                
                try:
                    result = mcp_isolated.index_document.fn(f.name)
                    assert "error" not in result
                finally:
                    try:
                        Path(f.name).unlink(missing_ok=True)
                    except PermissionError:
                        pass
        
        # Test various searches
        search_tests = [
            ("API authentication", "Should find auth documentation"),
            ("user management", "Should find UserManager code"),
            ("bearer token", "Should find token references"),
            ("create user", "Should find user creation methods"),
        ]
        
        for query, description in search_tests:
            results = mcp_isolated.search.fn(query, top_k=3)
            
            assert isinstance(results, list)
            assert len(results) > 0, f"No results for: {query}"
            
            # Check result structure
            result = results[0]
            assert "chunk_text" in result
            assert "similarity_score" in result or "score" in result
            
            score = result.get("similarity_score", result.get("score", 0))
            # Different embedding models use different score ranges
            # Some use cosine similarity which can be negative
            assert isinstance(score, (int, float)), f"Score should be numeric for '{query}': {score}"
            import math
            assert math.isfinite(score), f"Score should be finite for '{query}': {score}"
            
            print(f"✅ Search '{query}': {len(results)} results, best score: {score:.3f}")
    
    def test_search_with_context_function(self, mcp_isolated):
        """Test the search_with_context MCP function."""
        # Index some content first
        content = """# User Authentication System
        
## Overview
The authentication system handles user login, registration, and session management.

## Components
- UserManager: Main authentication class
- TokenValidator: Validates bearer tokens
- SessionStore: Manages user sessions

## Example Usage
```python
auth = UserManager()
user = auth.authenticate('bearer_token')
```
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            f.flush()
            
            try:
                result = mcp_isolated.index_document.fn(f.name)
                assert "error" not in result
                
                # Test context search
                results = mcp_isolated.search_with_context.fn(
                    "authentication system", 
                    top_k=3, 
                    include_related=True
                )
                
                assert isinstance(results, list)
                assert len(results) > 0
                
                # Verify result structure
                result = results[0]
                assert "chunk_text" in result
                assert "similarity_score" in result or "score" in result
                
                print(f"✅ Context search: {len(results)} results")
                
            finally:
                try:
                    Path(f.name).unlink(missing_ok=True)
                except PermissionError:
                    pass
    
    def test_error_handling(self, mcp_isolated):
        """Test MCP function error handling."""
        # Test non-existent file
        result = mcp_isolated.index_document.fn("/non/existent/file.txt")
        assert "error" in result
        print("✅ Error handling for missing files")
        
        # Test non-existent directory
        result = mcp_isolated.index_code.fn("/non/existent/directory")
        assert "error" in result  
        print("✅ Error handling for missing directories")
        
        # Test invalid search
        results = mcp_isolated.search.fn("", top_k=5)
        assert results == []
        print("✅ Error handling for empty queries")
    
    def test_full_workflow(self, mcp_isolated):
        """Test complete workflow with multiple document types."""
        print("\n🚀 Testing full MCP workflow...")
        
        # Step 1: Check initial status
        status = mcp_isolated.status.fn()
        assert status["metadata"]["total_chunks"] == 0
        print("✅ Initial status verified")
        
        # Step 2: Index various documents
        documents = {
            "API_Guide.md": """# API Reference Guide
            
## REST Endpoints
- GET /api/v1/users
- POST /api/v1/users
- PUT /api/v1/users/{id}

## Authentication
Use Bearer tokens in the Authorization header.
""",
            "UserService.txt": """
class UserService:
    def __init__(self, database):
        self.db = database
    
    async def get_users(self):
        '''Fetch all users from database.'''
        return await self.db.query('SELECT * FROM users')
    
    async def create_user(self, user_data):
        '''Create new user account.'''
        return await self.db.insert('users', user_data)
""",
            "config.txt": """{
    "api": {
        "version": "v1",
        "base_url": "/api/v1",
        "rate_limit": 1000
    },
    "database": {
        "type": "postgresql",
        "host": "localhost",
        "port": 5432
    }
}"""
        }
        
        total_chunks = 0
        for filename, content in documents.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix=Path(filename).suffix, delete=False, encoding='utf-8') as f:
                f.write(content)
                f.flush()
                
                try:
                    result = mcp_isolated.index_document.fn(f.name)
                    assert "error" not in result
                    chunks = result["chunks_added"]
                    total_chunks += chunks
                    print(f"  ✅ Indexed {filename}: {chunks} chunks")
                finally:
                    try:
                        Path(f.name).unlink(missing_ok=True)
                    except PermissionError:
                        pass
        
        print(f"✅ Total indexed: {total_chunks} chunks")
        
        # Step 3: Test searches across document types
        search_queries = [
            "API endpoints documentation",
            "user service Python class", 
            "database configuration settings",
            "bearer token authentication"
        ]
        
        for query in search_queries:
            results = mcp_isolated.search.fn(query, top_k=3)
            assert len(results) > 0, f"No results for: {query}"
            best_score = results[0].get("similarity_score", results[0].get("score", 0))
            print(f"  ✅ '{query}': {len(results)} results (best: {best_score:.3f})")
        
        # Step 4: Verify final status
        final_status = mcp_isolated.status.fn()
        assert final_status["metadata"]["total_chunks"] == total_chunks
        print(f"✅ Final status: {final_status['metadata']['total_chunks']} chunks indexed")
        
        print("🎉 Full workflow completed successfully!")


def test_direct_function_calls():
    """Test that MCP functions can be called directly."""
    # Test without fixture to verify functions exist
    assert hasattr(mcp_server, 'status')
    assert hasattr(mcp_server, 'index_document')
    assert hasattr(mcp_server, 'index_code')
    assert hasattr(mcp_server, 'search')
    assert hasattr(mcp_server, 'search_with_context')
    
    print("✅ All MCP functions are available")


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])