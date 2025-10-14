"""System tests for MCP server lazy loading performance.

These tests specifically address the performance issue where the first indexing
operation after server startup can take a very long time due to lazy loading
of the embedding model. The tests measure timing and provide diagnostics.
"""

import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import pytest


@pytest.mark.system
@pytest.mark.slow
class TestMCPLazyLoading:
    """Test MCP server lazy loading performance characteristics."""
    
    def test_first_indexing_performance(self):
        """Test that first indexing operation completes within reasonable time.
        
        This test specifically addresses the issue where first indexing can take
        a very long time due to lazy loading of the embedding model.
        
        Uses direct MCP tool access (like other system tests) for reliability.
        """
        from pycontextify import mcp
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            
            # Create test content with reasonable size for realistic timing
            test_content = self._create_test_content()
            test_dir = temp_path / "test_files"
            test_dir.mkdir()
            
            # Create multiple files to simulate realistic indexing load
            for filename, content in test_content:
                (test_dir / filename).write_text(content, encoding="utf-8")
            
            # Set up index directory
            index_dir = temp_path / "test_index"
            
            print(f"\n🚀 Starting MCP lazy loading performance test...")
            print(f"   Files to index: {len(test_content)}")
            print(f"   Total content size: ~{sum(len(content) for _, content in test_content)} chars")
            print(f"   Index directory: {index_dir}")
            
            # Initialize manager with fresh state (forces lazy loading)
            mcp.reset_manager()
            mcp.initialize_manager({
                "index_dir": str(index_dir),
                "auto_persist": False,
                "auto_load": False,  # Ensure we start with empty index
                "embedding_model": "all-MiniLM-L6-v2",  # Use faster model
            })
            
            try:
                # Get direct access to MCP tools
                status_fn = mcp.mcp._tool_manager._tools["status"].fn
                index_filebase_fn = mcp.mcp._tool_manager._tools["index_filebase"].fn
                search_fn = mcp.mcp._tool_manager._tools["search"].fn
                
                # Step 1: Test initial status (should be fast)
                print("\n📊 Step 1: Testing initial status...")
                start_time = time.time()
                status = status_fn()
                status_time = time.time() - start_time
                
                assert status["metadata"]["total_chunks"] == 0, "Index should be empty initially"
                assert status["vector_store"]["total_vectors"] == 0, "Vector store should be empty initially"
                print(f"   ✅ Status call: {status_time:.2f}s")
                
                # Step 2: Measure first indexing operation (this triggers lazy loading)
                print("\n🔄 Step 2: Testing first indexing (lazy loading)...")
                print("   ⏱️  This may take time due to embedding model loading...")
                
                start_time = time.time()
                doc_result = index_filebase_fn(base_path=str(test_dir), topic="performance_test")
                index_time = time.time() - start_time
                
                assert "error" not in doc_result, f"Indexing failed: {doc_result}"
                assert doc_result["chunks_created"] > 0, "Should have created chunks"
                
                print(f"   ✅ First indexing: {index_time:.2f}s")
                print(f"   📄 Files loaded: {doc_result['files_loaded']}")
                print(f"   📄 Chunks created: {doc_result['chunks_created']}")
                
                # Performance assertions
                if index_time > 120:  # 2 minutes
                    pytest.fail(
                        f"First indexing took too long: {index_time:.2f}s > 120s. "
                        f"This may indicate a performance issue with lazy loading."
                    )
                elif index_time > 60:  # 1 minute
                    print(f"   ⚠️  Warning: Indexing took {index_time:.2f}s (>60s)")
                    print("      This may indicate slow model loading or network issues")
                else:
                    print(f"   ✅ Performance acceptable: {index_time:.2f}s")
                
                # Step 3: Test subsequent indexing (should be faster)
                print("\n🔄 Step 3: Testing second indexing (model already loaded)...")
                
                # Create additional test file
                additional_dir = temp_path / "additional"
                additional_dir.mkdir()
                (additional_dir / "additional.md").write_text(
                    "# Additional Content\nThis is additional test content for measuring performance.", 
                    encoding="utf-8"
                )
                
                start_time = time.time()
                second_result = index_filebase_fn(base_path=str(additional_dir), topic="performance_test_2")
                second_time = time.time() - start_time
                
                assert "error" not in second_result, f"Second indexing failed: {second_result}"
                print(f"   ✅ Second indexing: {second_time:.2f}s")
                
                # Second indexing should be significantly faster
                if second_time > 30:
                    print(f"   ⚠️  Warning: Second indexing still slow: {second_time:.2f}s")
                else:
                    print(f"   ✅ Second indexing faster as expected")
                
                # Calculate speedup ratio
                if second_time > 0:
                    speedup = index_time / second_time
                    print(f"   📈 Speedup ratio: {speedup:.1f}x faster")
                    
                    # Expect at least some speedup after model is loaded
                    if speedup < 1.2:
                        print("   ⚠️  Warning: Expected more speedup after model loading")
                
                # Step 4: Test search performance
                print("\n🔍 Step 4: Testing search performance...")
                start_time = time.time()
                search_results = search_fn("test content", top_k=3)
                search_time = time.time() - start_time
                
                assert isinstance(search_results, list), "Search should return list"
                assert len(search_results) > 0, "Should find results"
                print(f"   ✅ Search time: {search_time:.2f}s")
                print(f"   🔍 Found {len(search_results)} results")
                
                if search_time > 5:
                    print(f"   ⚠️  Warning: Search took {search_time:.2f}s (>5s)")
                
                # Final summary
                print("\n" + "="*60)
                print("🎯 LAZY LOADING PERFORMANCE SUMMARY")
                print("="*60)
                print(f"Initial status:     {status_time:.2f}s")
                print(f"First indexing:     {index_time:.2f}s (includes model loading)")
                print(f"Second indexing:    {second_time:.2f}s (model already loaded)")
                print(f"Search operation:   {search_time:.2f}s")
                
                if index_time > 60:
                    print(f"\n⚠️  PERFORMANCE CONCERN:")
                    print(f"First indexing took {index_time:.2f}s which may be too slow")
                    print(f"Consider investigating embedding model loading optimization")
                else:
                    print(f"\n✅ Performance looks good!")
                
            finally:
                # Cleanup
                mcp.reset_manager()
    
    def test_embedding_model_loading_diagnostics(self):
        """Test to provide diagnostics on embedding model loading time."""
        from pycontextify import mcp
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            index_dir = temp_path / "test_index"
            
            # Create minimal test file
            test_dir = temp_path / "minimal_test"
            test_dir.mkdir()
            (test_dir / "tiny.txt").write_text("This is a tiny test file for diagnostics.")
            
            print("\n🔬 EMBEDDING MODEL LOADING DIAGNOSTICS")
            print("="*60)
            
            # Initialize manager with minimal configuration
            mcp.reset_manager()
            mcp.initialize_manager({
                "index_dir": str(index_dir),
                "auto_persist": False,
                "auto_load": False,
                "embedding_model": "all-MiniLM-L6-v2",  # Fast model for testing
            })
            
            try:
                # Time just the model loading by using minimal content
                print("Testing with minimal content to isolate model loading time...")
                
                index_filebase_fn = mcp.mcp._tool_manager._tools["index_filebase"].fn
                
                start_time = time.time()
                result = index_filebase_fn(base_path=str(test_dir), topic="diagnostic")
                elapsed = time.time() - start_time
                
                if "error" in result:
                    pytest.skip(f"Indexing failed: {result}")
                
                print(f"Minimal indexing time: {elapsed:.2f}s")
                
                if elapsed > 45:
                    print(f"⚠️  Slow model loading detected: {elapsed:.2f}s > 45s")
                    print("This may be due to:")
                    print("- First-time model download")
                    print("- Slow disk I/O")
                    print("- Network issues during model loading")
                    print("- CPU-intensive model initialization")
                elif elapsed > 20:
                    print(f"⚠️  Moderate delay: {elapsed:.2f}s > 20s")
                    print("Model loading time is acceptable but could be optimized")
                else:
                    print(f"✅ Fast model loading: {elapsed:.2f}s")
                    
            finally:
                mcp.reset_manager()
    
    def _create_test_content(self) -> List[Tuple[str, str]]:
        """Create realistic test content for indexing."""
        return [
            ("README.md", """# Test Project
            
## Overview
This is a comprehensive test project for evaluating indexing performance.
It contains multiple sections and realistic content patterns.

## Features
- Multi-file content indexing
- Performance measurement capabilities  
- Realistic content patterns
- Structured documentation

## Technical Details
The system processes various file types including Python code,
Markdown documentation, and plain text files.
"""),
            ("main.py", """#!/usr/bin/env python3
\"\"\"Main module for test project.

This module demonstrates various Python constructs that should be
properly indexed and made searchable through semantic search.
\"\"\"

import os
import sys
from typing import List, Dict, Optional

class DocumentProcessor:
    \"\"\"Process documents for indexing.\"\"\"
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.processed_count = 0
    
    def process_file(self, filepath: str) -> Optional[Dict]:
        \"\"\"Process a single file.
        
        Args:
            filepath: Path to file to process
            
        Returns:
            Processed file metadata or None if processing failed
        \"\"\"
        if not os.path.exists(filepath):
            return None
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            self.processed_count += 1
            return {
                'path': filepath,
                'size': len(content),
                'lines': content.count('\\n'),
                'processed': True
            }
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
    
    def process_batch(self, filepaths: List[str]) -> List[Dict]:
        \"\"\"Process multiple files in batch.\"\"\"
        results = []
        for filepath in filepaths:
            result = self.process_file(filepath)
            if result:
                results.append(result)
        return results

def main():
    \"\"\"Main entry point.\"\"\"
    config = {'output_dir': './processed', 'format': 'json'}
    processor = DocumentProcessor(config)
    
    # Example usage
    files = ['doc1.txt', 'doc2.md', 'code.py']
    results = processor.process_batch(files)
    
    print(f"Processed {len(results)} files successfully")
    return 0

if __name__ == '__main__':
    sys.exit(main())
"""),
            ("utils.py", """\"\"\"Utility functions for the test project.\"\"\"

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    \"\"\"Load configuration from JSON file.\"\"\"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config: {e}")
        return {}

def validate_paths(paths: List[str]) -> List[str]:
    \"\"\"Validate that all paths exist.\"\"\"
    valid_paths = []
    for path in paths:
        if Path(path).exists():
            valid_paths.append(path)
        else:
            logger.warning(f"Path does not exist: {path}")
    return valid_paths

def format_results(results: List[Dict]) -> str:
    \"\"\"Format results for display.\"\"\"
    output = []
    output.append("Processing Results:")
    output.append("-" * 20)
    
    for i, result in enumerate(results, 1):
        output.append(f"{i}. {result.get('path', 'Unknown')}")
        output.append(f"   Size: {result.get('size', 0)} bytes")
        output.append(f"   Status: {'✓' if result.get('processed') else '✗'}")
        output.append("")
    
    return "\\n".join(output)
"""),
            ("api_docs.md", """# API Documentation

## Authentication

All API requests require authentication using Bearer tokens:

```
Authorization: Bearer YOUR_TOKEN_HERE
```

## Endpoints

### GET /api/v1/documents
Retrieve list of documents.

**Parameters:**
- `limit` (optional): Maximum number of results (default: 100)
- `offset` (optional): Number of results to skip (default: 0)
- `filter` (optional): Filter criteria

**Response:**
```json
{
  "documents": [...],
  "total": 42,
  "limit": 100,
  "offset": 0
}
```

### POST /api/v1/documents
Create a new document.

**Request Body:**
```json
{
  "title": "Document Title",
  "content": "Document content...",
  "tags": ["tag1", "tag2"],
  "metadata": {...}
}
```

**Response:**
```json
{
  "id": "doc-123",
  "title": "Document Title", 
  "created_at": "2024-01-01T00:00:00Z",
  "status": "created"
}
```

### PUT /api/v1/documents/{id}
Update an existing document.

### DELETE /api/v1/documents/{id}
Delete a document.

## Error Handling

All API errors return JSON with error details:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Missing required field: title",
    "details": {...}
  }
}
```

## Rate Limiting

API requests are limited to 1000 requests per hour per user.
""")
        ]
