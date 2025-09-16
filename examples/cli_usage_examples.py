#!/usr/bin/env python3
"""Real-world CLI usage examples for PyContextify MCP server.

This script demonstrates practical CLI usage scenarios combining
documents, codebases, and webpage indexing with different configurations.
"""

import subprocess
import sys
from pathlib import Path


def example_python_project():
    """Example: Index a Python project with documentation."""
    print("🐍 Python Project Setup")
    print("=" * 50)
    
    cmd = [
        "uv", "run", "pycontextify",
        "--index-name", "python_project",
        "--index-path", "./project_index",
        "--initial-documents", "README.md", "CHANGELOG.md", "docs/",
        "--initial-codebase", "src", "tests",
        "--initial-webpages", "https://docs.python.org/3/library/", 
        "--recursive-crawling",
        "--max-crawl-depth", "1",
        "--crawl-delay", "2",
        "--verbose"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis setup will:")
    print("✓ Index project documentation (README, CHANGELOG, docs/)")
    print("✓ Index source code and tests")
    print("✓ Index Python standard library docs with recursive crawling")
    print("✓ Use respectful crawling with 2-second delays")
    print("✓ Store everything in a dedicated project index")
    print()


def example_research_project():
    """Example: Index research materials with academic papers."""
    print("🔬 Research Project Setup")
    print("=" * 50)
    
    cmd = [
        "uv", "run", "pycontextify",
        "--index-name", "research_ai",
        "--index-path", "./research_index",
        "--initial-documents", "papers/*.pdf", "notes.md", "references.txt",
        "--initial-webpages", 
            "https://arxiv.org/abs/1706.03762",  # Attention is All You Need
            "https://openai.com/research/gpt-4",
            "https://huggingface.co/docs/transformers",
        "--recursive-crawling",
        "--max-crawl-depth", "1", 
        "--crawl-delay", "3",
        "--embedding-provider", "sentence_transformers",
        "--embedding-model", "all-mpnet-base-v2"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis setup will:")
    print("✓ Index research papers (PDF files)")
    print("✓ Index personal notes and references")  
    print("✓ Index key AI research websites")
    print("✓ Use high-quality embeddings for academic content")
    print("✓ Respectful crawling with 3-second delays")
    print()


def example_api_documentation():
    """Example: Index API documentation sites."""
    print("📚 API Documentation Setup")
    print("=" * 50)
    
    cmd = [
        "uv", "run", "pycontextify",
        "--index-name", "api_docs",
        "--initial-webpages",
            "https://docs.fastapi.tiangolo.com/",
            "https://requests.readthedocs.io/",
            "https://docs.pydantic.dev/",
        "--recursive-crawling",
        "--max-crawl-depth", "2",
        "--crawl-delay", "1",
        "--no-auto-persist",  # Don't auto-save during development
        "--verbose"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis setup will:")
    print("✓ Index comprehensive API documentation")
    print("✓ Deep crawling (depth 2) for complete coverage")
    print("✓ Fast crawling (1-second delays) for development")
    print("✓ Manual index persistence control")
    print()


def example_knowledge_base():
    """Example: Comprehensive knowledge base with mixed content."""
    print("🧠 Knowledge Base Setup")
    print("=" * 50)
    
    cmd = [
        "uv", "run", "pycontextify",
        "--index-name", "knowledge_base",
        "--index-path", "./kb_index",
        "--initial-documents", "knowledge/*.md", "guides/*.pdf", "specs/*.txt",
        "--initial-codebase", "examples", "demos", "templates",
        "--initial-webpages",
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
            "https://docs.docker.com/",
            "https://kubernetes.io/docs/",
        "--recursive-crawling",
        "--max-crawl-depth", "1",
        "--crawl-delay", "2",
        "--embedding-provider", "sentence_transformers",
        "--no-auto-load"  # Fresh start each time
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis setup will:")
    print("✓ Index comprehensive documentation files")
    print("✓ Index example code and templates")
    print("✓ Index major technology documentation sites")
    print("✓ Create a comprehensive searchable knowledge base")
    print("✓ Fresh indexing on each startup")
    print()


def example_documentation_site():
    """Example: Deep crawl of a documentation website."""
    print("🌐 Documentation Site Deep Crawl")
    print("=" * 50)
    
    cmd = [
        "uv", "run", "pycontextify",
        "--index-name", "django_docs",
        "--initial-webpages", "https://docs.djangoproject.com/en/stable/",
        "--recursive-crawling",
        "--max-crawl-depth", "3",  # Maximum allowed depth
        "--crawl-delay", "2",
        "--verbose"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis setup will:")
    print("✓ Deep crawl Django documentation (depth 3)")
    print("✓ Comprehensive coverage of all documentation sections")
    print("✓ Respectful crawling with 2-second delays")
    print("✓ Detailed logging for monitoring progress")
    print()


def example_development_workflow():
    """Example: Development workflow with quick iteration."""
    print("⚡ Development Workflow")
    print("=" * 50)
    
    cmd = [
        "uv", "run", "pycontextify",
        "--index-name", "dev_workspace",
        "--initial-documents", "README.md",
        "--initial-codebase", "src",
        "--no-auto-persist",
        "--no-auto-load", 
        "--verbose"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis setup will:")
    print("✓ Quick indexing of core development files")
    print("✓ No persistence for rapid iteration")
    print("✓ Fresh start each time for testing changes")
    print("✓ Minimal setup for development/debugging")
    print()


def example_multilingual_content():
    """Example: Mixed content with different embedding models."""
    print("🌍 Multilingual Content Setup")
    print("=" * 50)
    
    cmd = [
        "uv", "run", "pycontextify",
        "--index-name", "multilingual",
        "--initial-documents", "content/en/*.md", "content/es/*.md", "content/fr/*.md",
        "--initial-webpages",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://es.wikipedia.org/wiki/Aprendizaje_automático",
        "--recursive-crawling",
        "--max-crawl-depth", "1",
        "--crawl-delay", "3",
        "--embedding-provider", "sentence_transformers",
        "--embedding-model", "paraphrase-multilingual-mpnet-base-v2"  # Multilingual model
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis setup will:")
    print("✓ Index multilingual documents") 
    print("✓ Index Wikipedia content in multiple languages")
    print("✓ Use multilingual embedding model")
    print("✓ Respectful crawling for Wikipedia")
    print()


def show_help_example():
    """Show how to get help information."""
    print("❓ Getting Help")
    print("=" * 50)
    
    cmd = ["uv", "run", "pycontextify", "--help"]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis will show:")
    print("✓ All available CLI arguments")
    print("✓ Usage examples")
    print("✓ Configuration priority explanation")
    print("✓ Detailed help for each option")
    print()


def main():
    """Display all usage examples."""
    print("PyContextify CLI Usage Examples")
    print("=" * 60)
    print()
    print("These examples show practical ways to use the PyContextify CLI")
    print("for different scenarios. Copy and modify them for your needs!")
    print()
    
    examples = [
        example_python_project,
        example_research_project, 
        example_api_documentation,
        example_knowledge_base,
        example_documentation_site,
        example_development_workflow,
        example_multilingual_content,
        show_help_example
    ]
    
    for example in examples:
        example()
    
    print("💡 Tips:")
    print("- Start with simple setups and add complexity gradually")
    print("- Use --verbose for detailed progress monitoring")
    print("- Adjust --crawl-delay based on the target website's capacity")
    print("- Use --no-auto-persist during development for faster iteration")
    print("- Combine different content types for comprehensive search indexes")
    print()
    
    print("📝 Notes:")
    print("- Replace example URLs with your actual documentation sites")
    print("- Adjust file paths to match your project structure") 
    print("- Consider using environment variables for sensitive configurations")
    print("- Test with small datasets before indexing large amounts of content")


if __name__ == "__main__":
    main()