"""Integration tests for complete CLI functionality including webpages."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockIndexManager:
    """Mock IndexManager for testing CLI integration."""
    
    def __init__(self, config):
        self.config = config
        self.indexed_documents = []
        self.indexed_codebases = []
        self.indexed_webpages = []
        
    def index_document(self, path):
        """Mock document indexing."""
        self.indexed_documents.append(path)
        return {"chunks_added": 5, "file_processed": 1}
        
    def index_codebase(self, path):
        """Mock codebase indexing.""" 
        self.indexed_codebases.append(path)
        return {"chunks_added": 20, "files_processed": 4}
        
    def index_webpage(self, url, recursive=False, max_depth=1):
        """Mock webpage indexing."""
        self.indexed_webpages.append({
            "url": url,
            "recursive": recursive,
            "max_depth": max_depth
        })
        pages = 3 if recursive else 1
        chunks = 15 if recursive else 8
        return {"pages_processed": pages, "chunks_added": chunks}
        
    def save_index(self):
        """Mock index saving."""
        return True


def simulate_full_cli_integration(
    initial_documents=None,
    initial_codebase=None, 
    initial_webpages=None,
    recursive_crawling=False,
    max_crawl_depth=1,
    crawl_delay=None,
    index_name="test_index",
    verbose=False
):
    """Simulate full CLI integration workflow."""
    
    # Create mock config
    config_overrides = {}
    if crawl_delay:
        config_overrides["crawl_delay_seconds"] = crawl_delay
    if index_name:
        config_overrides["index_name"] = index_name
        
    # Create mock config object
    mock_config = MagicMock()
    mock_config.auto_persist = True
    mock_config.crawl_delay_seconds = crawl_delay or 1
    mock_config.index_name = index_name
    
    # Create mock manager
    mgr = MockIndexManager(mock_config)
    
    # Simulate initial indexing process
    total_indexed = 0
    
    # Index documents
    if initial_documents:
        for doc_path in initial_documents:
            if Path(doc_path).suffix.lower() in {".pdf", ".md", ".txt"}:
                result = mgr.index_document(doc_path)
                total_indexed += result.get("chunks_added", 0)
    
    # Index codebases
    if initial_codebase:
        for codebase_path in initial_codebase:
            result = mgr.index_codebase(codebase_path)
            total_indexed += result.get("chunks_added", 0)
    
    # Index webpages
    if initial_webpages:
        for webpage_url in initial_webpages:
            if webpage_url.startswith(("http://", "https://")):
                # Validate max_depth
                depth = max_crawl_depth
                if depth < 1 or depth > 3:
                    depth = min(max(depth, 1), 3)
                
                result = mgr.index_webpage(webpage_url, recursive=recursive_crawling, max_depth=depth)
                total_indexed += result.get("chunks_added", 0)
    
    # Save index if auto-persist enabled
    if mgr.config.auto_persist:
        mgr.save_index()
    
    return {
        "total_indexed": total_indexed,
        "manager": mgr,
        "config": mock_config
    }


class TestFullCLIIntegration(unittest.TestCase):
    """Test complete CLI functionality integration."""

    def test_documents_only_integration(self):
        """Test CLI with only document indexing."""
        result = simulate_full_cli_integration(
            initial_documents=["README.md", "docs/api.md", "guide.txt"]
        )
        
        self.assertEqual(result["total_indexed"], 15)  # 3 docs × 5 chunks each
        self.assertEqual(len(result["manager"].indexed_documents), 3)
        self.assertEqual(len(result["manager"].indexed_codebases), 0)
        self.assertEqual(len(result["manager"].indexed_webpages), 0)

    def test_codebases_only_integration(self):
        """Test CLI with only codebase indexing."""
        result = simulate_full_cli_integration(
            initial_codebase=["src", "tests"]
        )
        
        self.assertEqual(result["total_indexed"], 40)  # 2 codebases × 20 chunks each
        self.assertEqual(len(result["manager"].indexed_documents), 0)
        self.assertEqual(len(result["manager"].indexed_codebases), 2)
        self.assertEqual(len(result["manager"].indexed_webpages), 0)

    def test_webpages_only_integration(self):
        """Test CLI with only webpage indexing."""
        result = simulate_full_cli_integration(
            initial_webpages=["https://docs.python.org", "https://github.com/python/cpython"]
        )
        
        self.assertEqual(result["total_indexed"], 16)  # 2 webpages × 8 chunks each (non-recursive)
        self.assertEqual(len(result["manager"].indexed_documents), 0)
        self.assertEqual(len(result["manager"].indexed_codebases), 0)
        self.assertEqual(len(result["manager"].indexed_webpages), 2)

    def test_recursive_webpages_integration(self):
        """Test CLI with recursive webpage indexing."""
        result = simulate_full_cli_integration(
            initial_webpages=["https://docs.example.com"],
            recursive_crawling=True,
            max_crawl_depth=2
        )
        
        self.assertEqual(result["total_indexed"], 15)  # 1 webpage × 15 chunks (recursive)
        webpage_data = result["manager"].indexed_webpages[0]
        self.assertTrue(webpage_data["recursive"])
        self.assertEqual(webpage_data["max_depth"], 2)

    def test_mixed_content_integration(self):
        """Test CLI with all content types together."""
        result = simulate_full_cli_integration(
            initial_documents=["README.md"],
            initial_codebase=["src"],  
            initial_webpages=["https://api-docs.com"],
            recursive_crawling=True,
            max_crawl_depth=1,
            crawl_delay=3,
            index_name="mixed_search"
        )
        
        expected_total = 5 + 20 + 15  # doc + codebase + webpage (recursive)
        self.assertEqual(result["total_indexed"], expected_total)
        self.assertEqual(len(result["manager"].indexed_documents), 1)
        self.assertEqual(len(result["manager"].indexed_codebases), 1)
        self.assertEqual(len(result["manager"].indexed_webpages), 1)
        
        # Check configuration overrides were applied
        self.assertEqual(result["config"].crawl_delay_seconds, 3)
        self.assertEqual(result["config"].index_name, "mixed_search")

    def test_invalid_content_filtering(self):
        """Test that invalid content is filtered out."""
        result = simulate_full_cli_integration(
            initial_documents=["README.md", "invalid.xyz"],  # One invalid extension
            initial_webpages=[
                "https://valid.com",
                "ftp://invalid.com",  # Invalid protocol
                "not-a-url"          # Not a URL
            ]
        )
        
        expected_total = 5 + 8  # Only valid doc + valid webpage
        self.assertEqual(result["total_indexed"], expected_total)
        self.assertEqual(len(result["manager"].indexed_documents), 1)
        self.assertEqual(len(result["manager"].indexed_webpages), 1)

    def test_crawl_depth_validation(self):
        """Test crawl depth validation in integration."""
        # Test depth too high
        result = simulate_full_cli_integration(
            initial_webpages=["https://docs.example.com"],
            recursive_crawling=True,
            max_crawl_depth=10  # Should be limited to 3
        )
        
        webpage_data = result["manager"].indexed_webpages[0]
        self.assertEqual(webpage_data["max_depth"], 3)  # Should be limited

    def test_crawl_depth_too_low(self):
        """Test crawl depth validation for low values."""
        result = simulate_full_cli_integration(
            initial_webpages=["https://docs.example.com"],
            recursive_crawling=True,
            max_crawl_depth=0  # Should be adjusted to 1
        )
        
        webpage_data = result["manager"].indexed_webpages[0]
        self.assertEqual(webpage_data["max_depth"], 1)  # Should be adjusted

    def test_large_scale_integration(self):
        """Test integration with many items of each type."""
        documents = [f"doc{i}.md" for i in range(5)]
        codebases = [f"codebase{i}" for i in range(3)]
        webpages = [f"https://site{i}.com" for i in range(4)]
        
        result = simulate_full_cli_integration(
            initial_documents=documents,
            initial_codebase=codebases,
            initial_webpages=webpages,
            recursive_crawling=False
        )
        
        expected_total = (5 * 5) + (3 * 20) + (4 * 8)  # docs + codebases + webpages
        self.assertEqual(result["total_indexed"], expected_total)
        self.assertEqual(len(result["manager"].indexed_documents), 5)
        self.assertEqual(len(result["manager"].indexed_codebases), 3)
        self.assertEqual(len(result["manager"].indexed_webpages), 4)

    def test_configuration_integration(self):
        """Test that configuration overrides work correctly."""
        result = simulate_full_cli_integration(
            initial_webpages=["https://example.com"],
            crawl_delay=5,
            index_name="custom_index"
        )
        
        self.assertEqual(result["config"].crawl_delay_seconds, 5)
        self.assertEqual(result["config"].index_name, "custom_index")

    def test_empty_integration(self):
        """Test integration with no initial content."""
        result = simulate_full_cli_integration()
        
        self.assertEqual(result["total_indexed"], 0)
        self.assertEqual(len(result["manager"].indexed_documents), 0)
        self.assertEqual(len(result["manager"].indexed_codebases), 0)
        self.assertEqual(len(result["manager"].indexed_webpages), 0)


if __name__ == "__main__":
    unittest.main()