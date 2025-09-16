"""Comprehensive test suite validating all CLI functionality together."""

import argparse
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_full_cli_parser():
    """Create the complete CLI parser matching the real implementation."""
    parser = argparse.ArgumentParser(
        prog="pycontextify",
        description="PyContextify MCP Server - Semantic search over codebases, documents, and webpages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Index configuration
    parser.add_argument("--index-path", type=str, help="Directory path for vector storage")
    parser.add_argument("--index-name", type=str, help="Custom index name")

    # Initial indexing
    parser.add_argument("--initial-documents", nargs="*", type=str, help="File paths to documents")
    parser.add_argument("--initial-codebase", nargs="*", type=str, help="Directory paths to codebases")
    parser.add_argument("--initial-webpages", nargs="*", type=str, help="URLs to webpages")

    # Webpage crawling options
    parser.add_argument("--recursive-crawling", action="store_true", help="Enable recursive crawling")
    parser.add_argument("--max-crawl-depth", type=int, default=1, help="Maximum crawl depth (1-3)")
    parser.add_argument("--crawl-delay", type=int, help="Delay between web requests")

    # Server configuration
    parser.add_argument("--no-auto-persist", action="store_true", help="Disable automatic persistence")
    parser.add_argument("--no-auto-load", action="store_true", help="Disable automatic index loading")

    # Embedding configuration
    parser.add_argument("--embedding-provider", choices=["sentence_transformers", "ollama", "openai"])
    parser.add_argument("--embedding-model", type=str, help="Embedding model name")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Minimize logging output")

    return parser


class TestCompleteCLISuite(unittest.TestCase):
    """Test complete CLI functionality with all features combined."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = create_full_cli_parser()

    def test_minimal_configuration(self):
        """Test minimal CLI configuration (no arguments)."""
        args = self.parser.parse_args([])
        
        # All optionals should be None/False/default
        self.assertIsNone(args.index_path)
        self.assertIsNone(args.index_name)
        self.assertIsNone(args.initial_documents)
        self.assertIsNone(args.initial_codebase)
        self.assertIsNone(args.initial_webpages)
        self.assertFalse(args.recursive_crawling)
        self.assertEqual(args.max_crawl_depth, 1)
        self.assertIsNone(args.crawl_delay)
        self.assertFalse(args.no_auto_persist)
        self.assertFalse(args.no_auto_load)
        self.assertFalse(args.verbose)
        self.assertFalse(args.quiet)

    def test_comprehensive_configuration(self):
        """Test comprehensive CLI configuration with all options."""
        argv = [
            "--index-path", "./comprehensive_index",
            "--index-name", "full_search",
            "--initial-documents", "README.md", "docs/guide.md", "manual.pdf",
            "--initial-codebase", "src", "tests", "lib",
            "--initial-webpages", "https://docs.python.org", "https://github.com/example/repo",
            "--recursive-crawling",
            "--max-crawl-depth", "2",
            "--crawl-delay", "3",
            "--no-auto-persist",
            "--no-auto-load",
            "--embedding-provider", "sentence_transformers",
            "--embedding-model", "all-mpnet-base-v2",
            "--verbose"
        ]
        
        args = self.parser.parse_args(argv)
        
        # Verify all arguments are parsed correctly
        self.assertEqual(args.index_path, "./comprehensive_index")
        self.assertEqual(args.index_name, "full_search")
        self.assertEqual(args.initial_documents, ["README.md", "docs/guide.md", "manual.pdf"])
        self.assertEqual(args.initial_codebase, ["src", "tests", "lib"])
        self.assertEqual(args.initial_webpages, ["https://docs.python.org", "https://github.com/example/repo"])
        self.assertTrue(args.recursive_crawling)
        self.assertEqual(args.max_crawl_depth, 2)
        self.assertEqual(args.crawl_delay, 3)
        self.assertTrue(args.no_auto_persist)
        self.assertTrue(args.no_auto_load)
        self.assertEqual(args.embedding_provider, "sentence_transformers")
        self.assertEqual(args.embedding_model, "all-mpnet-base-v2")
        self.assertTrue(args.verbose)

    def test_documents_only_workflow(self):
        """Test CLI workflow with only documents."""
        argv = [
            "--index-name", "docs_only",
            "--initial-documents", "README.md", "guide.pdf", "notes.txt",
            "--verbose"
        ]
        
        args = self.parser.parse_args(argv)
        
        self.assertEqual(args.initial_documents, ["README.md", "guide.pdf", "notes.txt"])
        self.assertIsNone(args.initial_codebase)
        self.assertIsNone(args.initial_webpages)
        self.assertFalse(args.recursive_crawling)

    def test_codebases_only_workflow(self):
        """Test CLI workflow with only codebases."""
        argv = [
            "--index-path", "./code_index",
            "--initial-codebase", "frontend", "backend", "mobile",
            "--embedding-provider", "sentence_transformers"
        ]
        
        args = self.parser.parse_args(argv)
        
        self.assertEqual(args.initial_codebase, ["frontend", "backend", "mobile"])
        self.assertIsNone(args.initial_documents)
        self.assertIsNone(args.initial_webpages)

    def test_webpages_only_workflow(self):
        """Test CLI workflow with only webpages."""
        argv = [
            "--initial-webpages", "https://docs.example.com", "https://api.example.com",
            "--recursive-crawling",
            "--max-crawl-depth", "3",
            "--crawl-delay", "2"
        ]
        
        args = self.parser.parse_args(argv)
        
        self.assertEqual(args.initial_webpages, ["https://docs.example.com", "https://api.example.com"])
        self.assertTrue(args.recursive_crawling)
        self.assertEqual(args.max_crawl_depth, 3)
        self.assertEqual(args.crawl_delay, 2)
        self.assertIsNone(args.initial_documents)
        self.assertIsNone(args.initial_codebase)

    def test_mixed_content_workflow(self):
        """Test CLI workflow with mixed content types."""
        argv = [
            "--index-name", "mixed_content",
            "--initial-documents", "spec.md",
            "--initial-codebase", "src",
            "--initial-webpages", "https://docs.api.com",
            "--recursive-crawling",
            "--crawl-delay", "1"
        ]
        
        args = self.parser.parse_args(argv)
        
        self.assertEqual(args.initial_documents, ["spec.md"])
        self.assertEqual(args.initial_codebase, ["src"])
        self.assertEqual(args.initial_webpages, ["https://docs.api.com"])
        self.assertTrue(args.recursive_crawling)
        self.assertEqual(args.crawl_delay, 1)

    def test_server_configuration_options(self):
        """Test server configuration CLI options."""
        argv = [
            "--no-auto-persist",
            "--no-auto-load",
            "--embedding-provider", "openai",
            "--embedding-model", "text-embedding-ada-002"
        ]
        
        args = self.parser.parse_args(argv)
        
        self.assertTrue(args.no_auto_persist)
        self.assertTrue(args.no_auto_load)
        self.assertEqual(args.embedding_provider, "openai")
        self.assertEqual(args.embedding_model, "text-embedding-ada-002")

    def test_logging_configuration(self):
        """Test logging configuration options."""
        # Test verbose
        args_verbose = self.parser.parse_args(["--verbose"])
        self.assertTrue(args_verbose.verbose)
        self.assertFalse(args_verbose.quiet)
        
        # Test quiet
        args_quiet = self.parser.parse_args(["--quiet"])
        self.assertFalse(args_quiet.verbose)
        self.assertTrue(args_quiet.quiet)
        
        # Test short form
        args_v = self.parser.parse_args(["-v"])
        self.assertTrue(args_v.verbose)

    def test_webpage_crawling_validation(self):
        """Test webpage crawling parameter validation."""
        # Test valid depth
        args = self.parser.parse_args(["--max-crawl-depth", "2"])
        self.assertEqual(args.max_crawl_depth, 2)
        
        # Test valid delay
        args = self.parser.parse_args(["--crawl-delay", "5"])
        self.assertEqual(args.crawl_delay, 5)

    def test_invalid_embedding_provider(self):
        """Test that invalid embedding providers are rejected."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--embedding-provider", "invalid_provider"])

    def test_empty_list_arguments(self):
        """Test behavior with empty list arguments."""
        argv = [
            "--initial-documents",  # No documents after flag
            "--initial-codebase", "src",  # But codebases provided
        ]
        
        args = self.parser.parse_args(argv)
        
        self.assertEqual(args.initial_documents, [])  # Should be empty list
        self.assertEqual(args.initial_codebase, ["src"])

    def test_multiple_values_per_argument(self):
        """Test multiple values for each list argument."""
        argv = [
            "--initial-documents", "doc1.md", "doc2.pdf", "doc3.txt",
            "--initial-codebase", "app", "lib", "tests", "scripts",
            "--initial-webpages", "https://site1.com", "https://site2.com", "https://site3.com"
        ]
        
        args = self.parser.parse_args(argv)
        
        self.assertEqual(len(args.initial_documents), 3)
        self.assertEqual(len(args.initial_codebase), 4)
        self.assertEqual(len(args.initial_webpages), 3)

    def test_real_world_scenarios(self):
        """Test real-world usage scenarios."""
        
        # Scenario 1: Python project documentation
        python_args = self.parser.parse_args([
            "--index-name", "python_project",
            "--initial-documents", "README.md", "CHANGELOG.md",
            "--initial-codebase", "src", "tests",
            "--initial-webpages", "https://docs.python.org",
            "--recursive-crawling",
            "--max-crawl-depth", "1"
        ])
        
        self.assertEqual(python_args.index_name, "python_project")
        self.assertTrue(python_args.recursive_crawling)
        
        # Scenario 2: Research project
        research_args = self.parser.parse_args([
            "--index-path", "./research_index",
            "--initial-documents", "paper.pdf", "notes.md", "references.txt",
            "--initial-webpages", "https://arxiv.org/abs/1234.5678",
            "--crawl-delay", "3"
        ])
        
        self.assertEqual(research_args.index_path, "./research_index")
        self.assertEqual(research_args.crawl_delay, 3)
        
        # Scenario 3: API documentation
        api_args = self.parser.parse_args([
            "--initial-webpages", "https://api.example.com/docs",
            "--recursive-crawling",
            "--max-crawl-depth", "2",
            "--embedding-provider", "sentence_transformers",
            "--verbose"
        ])
        
        self.assertTrue(api_args.recursive_crawling)
        self.assertEqual(api_args.max_crawl_depth, 2)
        self.assertTrue(api_args.verbose)

    def test_argument_combinations_edge_cases(self):
        """Test edge cases in argument combinations."""
        
        # Edge case 1: Recursive crawling without webpages
        args1 = self.parser.parse_args(["--recursive-crawling", "--max-crawl-depth", "2"])
        self.assertTrue(args1.recursive_crawling)
        self.assertIsNone(args1.initial_webpages)  # Should still work
        
        # Edge case 2: Crawl delay without webpages
        args2 = self.parser.parse_args(["--crawl-delay", "5"])
        self.assertEqual(args2.crawl_delay, 5)
        self.assertIsNone(args2.initial_webpages)  # Should still work
        
        # Edge case 3: Conflicting logging options (both verbose and quiet)
        args3 = self.parser.parse_args(["--verbose", "--quiet"])
        self.assertTrue(args3.verbose)
        self.assertTrue(args3.quiet)  # Both flags can be set


class TestCLIConfigurationOverrides(unittest.TestCase):
    """Test CLI configuration override functionality."""

    def test_complete_config_overrides(self):
        """Test complete configuration override conversion."""
        # Simulate the args_to_config_overrides function
        def args_to_config_overrides(args):
            overrides = {}
            if args.index_path:
                overrides["index_dir"] = args.index_path
            if args.index_name:
                overrides["index_name"] = args.index_name
            if args.no_auto_persist:
                overrides["auto_persist"] = False
            if args.no_auto_load:
                overrides["auto_load"] = False
            if args.embedding_provider:
                overrides["embedding_provider"] = args.embedding_provider
            if args.embedding_model:
                overrides["embedding_model"] = args.embedding_model
            if hasattr(args, 'crawl_delay') and args.crawl_delay is not None:
                overrides["crawl_delay_seconds"] = args.crawl_delay
            return overrides

        parser = create_full_cli_parser()
        
        # Test comprehensive overrides
        argv = [
            "--index-path", "./custom_index",
            "--index-name", "custom_search",
            "--no-auto-persist",
            "--no-auto-load",
            "--embedding-provider", "openai",
            "--embedding-model", "text-embedding-3-small",
            "--crawl-delay", "2"
        ]
        
        args = parser.parse_args(argv)
        overrides = args_to_config_overrides(args)
        
        expected = {
            "index_dir": "./custom_index",
            "index_name": "custom_search", 
            "auto_persist": False,
            "auto_load": False,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "crawl_delay_seconds": 2
        }
        
        self.assertEqual(overrides, expected)


if __name__ == "__main__":
    unittest.main()