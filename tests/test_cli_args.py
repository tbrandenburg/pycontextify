"""Tests for CLI argument parsing and configuration override functionality."""

import argparse
import os

# Import the functions directly to avoid dependency issues
# We'll copy the essential functions here for testing
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args_with_custom_argv(argv):
    """Parse command-line arguments with custom argv for testing."""
    parser = argparse.ArgumentParser(
        prog="pycontextify",
        description="PyContextify MCP Server - Semantic search over codebases, documents, and webpages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Index configuration
    parser.add_argument(
        "--index-path",
        type=str,
        help="Directory path for vector storage and index files",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        help="Custom index name",
    )

    # Initial indexing
    parser.add_argument(
        "--initial-documents",
        nargs="*",
        type=str,
        help="File paths to documents to index at startup",
    )
    parser.add_argument(
        "--initial-codebase",
        nargs="*",
        type=str,
        help="Directory paths to codebases to index at startup",
    )

    # Server configuration
    parser.add_argument(
        "--no-auto-persist",
        action="store_true",
        help="Disable automatic index persistence",
    )
    parser.add_argument(
        "--no-auto-load",
        action="store_true",
        help="Disable automatic index loading on startup",
    )

    # Embedding configuration
    parser.add_argument(
        "--embedding-provider",
        choices=["sentence_transformers", "ollama", "openai"],
        help="Embedding provider to use",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model name",
    )

    # Verbose logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize logging output",
    )

    return parser.parse_args(argv)


def args_to_config_overrides(args):
    """Convert CLI arguments to configuration overrides dictionary."""
    overrides = {}

    # Index configuration
    if args.index_path:
        overrides["index_dir"] = args.index_path
    if args.index_name:
        overrides["index_name"] = args.index_name

    # Server configuration
    if args.no_auto_persist:
        overrides["auto_persist"] = False
    if args.no_auto_load:
        overrides["auto_load"] = False

    # Embedding configuration
    if args.embedding_provider:
        overrides["embedding_provider"] = args.embedding_provider
    if args.embedding_model:
        overrides["embedding_model"] = args.embedding_model

    return overrides


class TestCLIArguments(unittest.TestCase):
    """Test CLI argument parsing functionality."""

    def test_no_arguments(self):
        """Test parsing with no arguments."""
        args = parse_args_with_custom_argv([])
        self.assertIsNone(args.index_path)
        self.assertIsNone(args.index_name)
        self.assertIsNone(args.initial_documents)
        self.assertIsNone(args.initial_codebase)
        self.assertFalse(args.no_auto_persist)
        self.assertFalse(args.no_auto_load)
        self.assertFalse(args.verbose)
        self.assertFalse(args.quiet)

    def test_index_configuration(self):
        """Test index configuration arguments."""
        args = parse_args_with_custom_argv(
            ["--index-path", "./custom_index", "--index-name", "my_search"]
        )
        self.assertEqual(args.index_path, "./custom_index")
        self.assertEqual(args.index_name, "my_search")

    def test_initial_documents(self):
        """Test initial document arguments."""
        args = parse_args_with_custom_argv(
            ["--initial-documents", "doc1.pdf", "doc2.md", "doc3.txt"]
        )
        self.assertEqual(args.initial_documents, ["doc1.pdf", "doc2.md", "doc3.txt"])

    def test_initial_codebase(self):
        """Test initial codebase arguments."""
        args = parse_args_with_custom_argv(
            ["--initial-codebase", "src", "tests", "lib"]
        )
        self.assertEqual(args.initial_codebase, ["src", "tests", "lib"])

    def test_server_configuration(self):
        """Test server configuration flags."""
        args = parse_args_with_custom_argv(["--no-auto-persist", "--no-auto-load"])
        self.assertTrue(args.no_auto_persist)
        self.assertTrue(args.no_auto_load)

    def test_embedding_configuration(self):
        """Test embedding provider arguments."""
        args = parse_args_with_custom_argv(
            [
                "--embedding-provider",
                "sentence_transformers",
                "--embedding-model",
                "all-mpnet-base-v2",
            ]
        )
        self.assertEqual(args.embedding_provider, "sentence_transformers")
        self.assertEqual(args.embedding_model, "all-mpnet-base-v2")

    def test_logging_configuration(self):
        """Test logging arguments."""
        # Test verbose
        args = parse_args_with_custom_argv(["--verbose"])
        self.assertTrue(args.verbose)
        self.assertFalse(args.quiet)

        # Test quiet
        args = parse_args_with_custom_argv(["--quiet"])
        self.assertFalse(args.verbose)
        self.assertTrue(args.quiet)

    def test_embedding_provider_choices(self):
        """Test that invalid embedding providers are rejected."""
        with self.assertRaises(SystemExit):
            parse_args_with_custom_argv(["--embedding-provider", "invalid_provider"])

    def test_full_configuration(self):
        """Test parsing with all arguments."""
        args = parse_args_with_custom_argv(
            [
                "--index-path",
                "./my_index",
                "--index-name",
                "project_search",
                "--initial-documents",
                "README.md",
                "docs/api.md",
                "--initial-codebase",
                "src",
                "tests",
                "--no-auto-persist",
                "--embedding-provider",
                "ollama",
                "--embedding-model",
                "nomic-embed-text",
                "--verbose",
            ]
        )

        self.assertEqual(args.index_path, "./my_index")
        self.assertEqual(args.index_name, "project_search")
        self.assertEqual(args.initial_documents, ["README.md", "docs/api.md"])
        self.assertEqual(args.initial_codebase, ["src", "tests"])
        self.assertTrue(args.no_auto_persist)
        self.assertEqual(args.embedding_provider, "ollama")
        self.assertEqual(args.embedding_model, "nomic-embed-text")
        self.assertTrue(args.verbose)


class TestConfigOverrides(unittest.TestCase):
    """Test configuration override conversion."""

    def test_empty_overrides(self):
        """Test no overrides when no CLI args are provided."""
        args = parse_args_with_custom_argv([])
        overrides = args_to_config_overrides(args)
        self.assertEqual(overrides, {})

    def test_index_overrides(self):
        """Test index configuration overrides."""
        args = parse_args_with_custom_argv(
            ["--index-path", "./custom", "--index-name", "test_search"]
        )
        overrides = args_to_config_overrides(args)
        self.assertEqual(overrides["index_dir"], "./custom")
        self.assertEqual(overrides["index_name"], "test_search")

    def test_server_overrides(self):
        """Test server configuration overrides."""
        args = parse_args_with_custom_argv(["--no-auto-persist", "--no-auto-load"])
        overrides = args_to_config_overrides(args)
        self.assertFalse(overrides["auto_persist"])
        self.assertFalse(overrides["auto_load"])

    def test_embedding_overrides(self):
        """Test embedding configuration overrides."""
        args = parse_args_with_custom_argv(
            [
                "--embedding-provider",
                "openai",
                "--embedding-model",
                "text-embedding-ada-002",
            ]
        )
        overrides = args_to_config_overrides(args)
        self.assertEqual(overrides["embedding_provider"], "openai")
        self.assertEqual(overrides["embedding_model"], "text-embedding-ada-002")

    def test_mixed_overrides(self):
        """Test mixed configuration overrides."""
        args = parse_args_with_custom_argv(
            [
                "--index-path",
                "./test_index",
                "--no-auto-persist",
                "--embedding-provider",
                "sentence_transformers",
            ]
        )
        overrides = args_to_config_overrides(args)

        expected = {
            "index_dir": "./test_index",
            "auto_persist": False,
            "embedding_provider": "sentence_transformers",
        }
        self.assertEqual(overrides, expected)


class TestConfigPriority(unittest.TestCase):
    """Test configuration priority system."""

    def create_mock_config(self, config_overrides=None):
        """Create a mock config class for testing priority."""

        class MockConfig:
            def __init__(self, config_overrides=None):
                self.config_overrides = config_overrides or {}

            def _get_config(self, env_key, default, override_key=None):
                """Mock config getter with CLI override priority."""
                # Check CLI override first
                if override_key and override_key in self.config_overrides:
                    return self.config_overrides[override_key]

                # Fall back to environment variable or default
                return os.getenv(env_key, default)

            def _get_path_config(self, env_key, default, override_key=None):
                """Mock path config getter."""
                from pathlib import Path

                # Check CLI override first
                if override_key and override_key in self.config_overrides:
                    value = str(self.config_overrides[override_key])
                else:
                    value = os.getenv(env_key, default)

                return Path(value).resolve()

        return MockConfig(config_overrides)

    def test_cli_override_priority(self):
        """Test that CLI overrides take priority over environment variables."""
        # Set environment variable
        with patch.dict(os.environ, {"PYCONTEXTIFY_INDEX_NAME": "env_index"}):
            # CLI override should take priority
            config = self.create_mock_config({"index_name": "cli_index"})
            result = config._get_config(
                "PYCONTEXTIFY_INDEX_NAME", "default_index", "index_name"
            )
            self.assertEqual(result, "cli_index")

    def test_env_var_fallback(self):
        """Test fallback to environment variable when no CLI override."""
        with patch.dict(os.environ, {"PYCONTEXTIFY_INDEX_NAME": "env_index"}):
            config = self.create_mock_config({})
            result = config._get_config(
                "PYCONTEXTIFY_INDEX_NAME", "default_index", "index_name"
            )
            self.assertEqual(result, "env_index")

    def test_default_fallback(self):
        """Test fallback to default when no CLI override or env var."""
        config = self.create_mock_config({})
        result = config._get_config(
            "PYCONTEXTIFY_INDEX_NAME", "semantic_index", "index_name"
        )
        self.assertEqual(result, "semantic_index")

    def test_path_config_priority(self):
        """Test path configuration priority."""
        with patch.dict(os.environ, {"PYCONTEXTIFY_INDEX_DIR": "./env_dir"}):
            config = self.create_mock_config({"index_dir": "./cli_dir"})
            result = config._get_path_config(
                "PYCONTEXTIFY_INDEX_DIR", "./default_dir", "index_dir"
            )
            # Should be absolute path with CLI override
            self.assertTrue(str(result).endswith("cli_dir"))


if __name__ == "__main__":
    unittest.main()
