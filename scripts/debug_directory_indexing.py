#!/usr/bin/env python3
"""
PyContextify Directory Indexing Debug Script

This script analyzes how the directory indexing pipeline processes code files,
converts them to normalized documents, and chunks them for indexing. It provides
detailed reports on the processing pipeline, chunking strategies, and metadata extraction.

Usage:
    python debug_directory_indexing.py [--output-dir PATH] [--dir-path PATH]

Examples:
    python debug_directory_indexing.py
    python debug_directory_indexing.py --dir-path pycontextify
    python debug_directory_indexing.py --output-dir ./debug_reports --dir-path src
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pycontextify.chunker import ChunkerFactory
from pycontextify.config import Config
from pycontextify.loader import FileLoaderFactory
from pycontextify.types import SourceType

logger = logging.getLogger(__name__)


class DirectoryIndexingDebugger:
    """Debug tool for analyzing directory-to-chunks conversion pipeline."""

    def __init__(self, output_dir: Path, dir_path: Path):
        self.output_dir = output_dir
        self.dir_path = dir_path
        self.start_time = datetime.now()
        self.debug_data: Dict[str, Any] = {}

    def debug_directory_pipeline(self) -> None:
        """Run complete directory processing pipeline with debugging."""
        print(f"🔍 Starting directory indexing debug at {self.start_time}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📂 Source directory: {self.dir_path}")

        if not self.dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.dir_path}")

        if not self.dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.dir_path}")

        self.debug_data["directory_info"] = self._analyze_directory_info()
        self.debug_data["loading"] = self._debug_loading()
        self.debug_data["chunking"] = self._debug_chunking()
        self.debug_data["timing"] = self._finalize_timing()

        self._generate_report()
        print(f"✅ Debug completed. Report saved to: {self.output_dir}")

    def _analyze_directory_info(self) -> Dict[str, Any]:
        """Analyze basic directory information and file distribution."""
        print("\n📊 Step 1: Analyzing directory structure")

        directory_info = {
            "directory_path": str(self.dir_path),
            "directory_name": self.dir_path.name,
        }

        # Get file statistics
        file_stats = defaultdict(int)
        file_extensions = defaultdict(int)
        total_files = 0
        total_size = 0
        supported_files = []

        # Common code file extensions
        code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt',
            '.scala', '.r', '.m', '.mm', '.sh', '.bash', '.zsh', '.ps1',
            '.sql', '.html', '.css', '.scss', '.less', '.vue', '.svelte',
            '.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.cfg', '.conf',
            '.md', '.rst', '.txt', '.tex', '.dockerfile', '.makefile'
        }

        try:
            for file_path in self.dir_path.rglob('*'):
                if file_path.is_file():
                    total_files += 1
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    extension = file_path.suffix.lower()
                    file_extensions[extension] += 1
                    
                    # Check if it's a supported code file
                    if extension in code_extensions or file_path.name.lower() in {'dockerfile', 'makefile'}:
                        supported_files.append({
                            'path': str(file_path.relative_to(self.dir_path)),
                            'size': file_size,
                            'extension': extension,
                            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        })
                        file_stats['supported'] += 1
                    else:
                        file_stats['unsupported'] += 1

        except Exception as e:
            print(f"⚠️  Error analyzing directory: {e}")
            directory_info["error"] = str(e)
            return directory_info

        directory_info.update({
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "supported_files_count": file_stats['supported'],
            "unsupported_files_count": file_stats['unsupported'],
            "file_extensions": dict(file_extensions),
            "supported_files": supported_files[:50],  # Limit to first 50 for report size
            "total_supported_files": len(supported_files)
        })

        print(f"📂 Directory contains {total_files} files ({directory_info['total_size_mb']} MB)")
        print(f"📄 Supported code files: {file_stats['supported']}")
        print(f"📄 Unsupported files: {file_stats['unsupported']}")
        print(f"📈 Top file types: {dict(list(sorted(file_extensions.items(), key=lambda x: x[1], reverse=True))[:10])}")

        return directory_info

    def _debug_loading(self) -> Dict[str, Any]:
        """Debug the directory loading and document conversion process."""
        print("\n🔄 Step 2: Loading and converting code files to documents")

        loading_start = time.time()

        # Create loader and attempt to load the directory
        loader = FileLoaderFactory(default_encoding="utf-8")
        tags = f"code,{self.dir_path.name},directory-indexing,debug"

        try:
            # Get all supported files in the directory
            normalized_docs = []
            
            # Process each supported file individually
            for file_info in self.debug_data["directory_info"]["supported_files"]:
                file_path = self.dir_path / file_info["path"]
                if file_path.exists() and file_path.is_file():
                    try:
                        docs = loader.load(
                            path=str(file_path),
                            tags=tags,
                            base_path=str(self.dir_path.parent),
                        )
                        normalized_docs.extend(docs)
                    except Exception as e:
                        loading_debug["processing_errors"].append({
                            "file": str(file_path),
                            "error": str(e)
                        })
                        logger.warning(f"Failed to load {file_path}: {e}")

            loading_time = time.time() - loading_start

            loading_debug = {
                "success": True,
                "loading_time_seconds": round(loading_time, 2),
                "documents_created": len(normalized_docs),
                "tags_applied": tags.split(","),
                "documents": [],
                "file_types": defaultdict(int),
                "processing_errors": []
            }

            print(f"✅ Loaded {len(normalized_docs)} documents in {loading_time:.2f}s")

            # Analyze each document
            total_chars = 0
            for i, doc in enumerate(normalized_docs):
                text = doc["text"]
                metadata = doc["metadata"]
                
                # Get file extension for statistics
                source_path = metadata.get("source_path", "")
                if source_path:
                    ext = Path(source_path).suffix.lower()
                    loading_debug["file_types"][ext] += 1

                doc_info = {
                    "document_index": i,
                    "character_count": len(text),
                    "word_count": len(text.split()),
                    "line_count": len(text.split("\n")),
                    "metadata": metadata,
                    "source_path": metadata.get("source_path", ""),
                    "file_type": metadata.get("file_type", "unknown"),
                    "text_preview": text[:500] + "..." if len(text) > 500 else text,
                    "text_sample_end": text[-200:] if len(text) > 200 else "",
                    "text_full": text,  # Store full content
                }

                # Analyze code patterns
                code_indicators = {
                    "function_definitions": text.count("def ") + text.count("function "),
                    "class_definitions": text.count("class "),
                    "import_statements": text.count("import ") + text.count("from "),
                    "comments": text.count("#") + text.count("//") + text.count("/*"),
                    "brackets": text.count("{") + text.count("["),
                    "semicolons": text.count(";"),
                }
                doc_info["code_indicators"] = code_indicators
                doc_info["appears_to_be_code"] = any(code_indicators.values())

                loading_debug["documents"].append(doc_info)
                total_chars += len(text)

                # Print summary for key files
                if i < 10 or doc_info["appears_to_be_code"]:
                    file_name = Path(source_path).name if source_path else f"doc_{i}"
                    print(f"  📄 {file_name}: {len(text):,} chars, {len(text.split()):,} words")
                    if doc_info["appears_to_be_code"]:
                        print(f"     💻 Code detected: {code_indicators}")

            loading_debug["total_characters"] = total_chars
            loading_debug["file_types"] = dict(loading_debug["file_types"])
            print(f"📊 Total content: {total_chars:,} characters")
            print(f"📈 File types processed: {loading_debug['file_types']}")

            return loading_debug

        except Exception as e:
            loading_time = time.time() - loading_start
            print(f"❌ Loading failed after {loading_time:.2f}s: {e}")
            return {
                "success": False,
                "loading_time_seconds": round(loading_time, 2),
                "error": str(e),
                "documents_created": 0,
            }

    def _debug_chunking(self) -> Dict[str, Any]:
        """Debug the chunking process with detailed analysis."""
        print("\n✂️  Step 3: Analyzing chunking process")

        # First, we need to load the documents again
        if not self.debug_data.get("loading", {}).get("success"):
            print("❌ Cannot debug chunking - loading failed")
            return {"success": False, "error": "Loading failed"}

        chunking_start = time.time()

        # Create config and load documents
        config = Config()
        print(f"🔧 Chunking config:")
        print(f"   • Chunk size: {config.chunk_size} tokens")
        print(f"   • Overlap: {config.chunk_overlap} tokens")
        print(f"   • Relationships enabled: {config.enable_relationships}")

        loader = FileLoaderFactory()
        tags = f"code,{self.dir_path.name},directory-indexing,debug"

        try:
            # Get all supported files in the directory
            normalized_docs = []
            
            # Process each supported file individually
            for file_info in self.debug_data["directory_info"]["supported_files"]:
                file_path = self.dir_path / file_info["path"]
                if file_path.exists() and file_path.is_file():
                    try:
                        docs = loader.load(
                            path=str(file_path),
                            tags=tags,
                            base_path=str(self.dir_path.parent),
                        )
                        normalized_docs.extend(docs)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")

            if not normalized_docs:
                return {"success": False, "error": "No documents to chunk"}

            # Add embedding info to metadata (required for chunking)
            for doc in normalized_docs:
                doc["metadata"]["embedding_provider"] = "sentence_transformers"
                doc["metadata"]["embedding_model"] = "all-mpnet-base-v2"

            # Chunk the documents
            chunks = ChunkerFactory.chunk_normalized_docs(
                normalized_docs=normalized_docs,
                config=config,
            )

            chunking_time = time.time() - chunking_start

            chunking_debug = {
                "success": True,
                "chunking_time_seconds": round(chunking_time, 2),
                "input_documents": len(normalized_docs),
                "output_chunks": len(chunks),
                "config": {
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "enable_relationships": config.enable_relationships,
                },
                "chunks": [],
                "file_chunk_distribution": defaultdict(int),
            }

            print(f"✅ Created {len(chunks)} chunks in {chunking_time:.2f}s")

            # Analyze each chunk
            total_chunk_chars = 0
            chunk_sizes = []

            for i, chunk in enumerate(chunks):
                chunk_text = chunk["text"]
                chunk_metadata = chunk["metadata"]
                
                source_path = chunk_metadata.get("source_path", "unknown")
                if source_path != "unknown":
                    file_name = Path(source_path).name
                    chunking_debug["file_chunk_distribution"][file_name] += 1

                chunk_info = {
                    "chunk_index": i,
                    "character_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "estimated_tokens": round(len(chunk_text.split()) * 1.3),
                    "start_char": chunk_metadata.get("start_char"),
                    "end_char": chunk_metadata.get("end_char"),
                    "source_path": source_path,
                    "file_name": Path(source_path).name if source_path != "unknown" else "unknown",
                    "file_type": chunk_metadata.get("file_type", "unknown"),
                    "tags": chunk_metadata.get("tags", []),
                    "text_preview": (
                        chunk_text[:300] + "..."
                        if len(chunk_text) > 300
                        else chunk_text
                    ),
                    "text_full": chunk_text,  # Store full text for complete review
                }

                # Add relationship info if available
                if config.enable_relationships:
                    chunk_info["references"] = chunk_metadata.get("references", [])
                    chunk_info["code_symbols"] = chunk_metadata.get("code_symbols", [])

                # Show chunk boundaries with visual markers
                preview_lines = chunk_text.split("\n")[:5]  # First 5 lines
                chunk_info["visual_preview"] = "\n".join(
                    [
                        f"┌── CHUNK {i+1} ({chunk_info['file_name']}) ──────────────────────────────",
                        *[
                            f"│ {line[:60]}..." if len(line) > 60 else f"│ {line}"
                            for line in preview_lines
                        ],
                        f"└── {len(chunk_text)} chars, ~{chunk_info['estimated_tokens']} tokens ──",
                    ]
                )

                chunking_debug["chunks"].append(chunk_info)
                total_chunk_chars += len(chunk_text)
                chunk_sizes.append(len(chunk_text))

                # Print every 20th chunk + first few + last few for large codebases
                if i < 5 or i >= len(chunks) - 5 or i % 20 == 0:
                    print(
                        f"  ✂️  Chunk {i+1:3d}: {len(chunk_text):4d} chars (~{chunk_info['estimated_tokens']:3d} tokens) - {chunk_info['file_name']}"
                    )

            # Calculate statistics
            chunking_debug["statistics"] = {
                "total_chunk_characters": total_chunk_chars,
                "average_chunk_size": round(sum(chunk_sizes) / len(chunk_sizes)),
                "min_chunk_size": min(chunk_sizes),
                "max_chunk_size": max(chunk_sizes),
                "chunks_per_document": round(len(chunks) / len(normalized_docs), 1),
            }

            chunking_debug["file_chunk_distribution"] = dict(chunking_debug["file_chunk_distribution"])

            print(f"📊 Chunk statistics:")
            stats = chunking_debug["statistics"]
            print(f"   • Average size: {stats['average_chunk_size']} chars")
            print(f"   • Size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} chars")
            print(f"   • Chunks per document: {stats['chunks_per_document']}")
            print(f"📈 Top files by chunk count: {dict(list(sorted(chunking_debug['file_chunk_distribution'].items(), key=lambda x: x[1], reverse=True))[:10])}")

            return chunking_debug

        except Exception as e:
            chunking_time = time.time() - chunking_start
            print(f"❌ Chunking failed after {chunking_time:.2f}s: {e}")
            return {
                "success": False,
                "chunking_time_seconds": round(chunking_time, 2),
                "error": str(e),
            }

    def _finalize_timing(self) -> Dict[str, Any]:
        """Calculate final timing information."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        return {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_seconds": round(total_duration, 2),
        }

    def _generate_report(self) -> None:
        """Generate comprehensive HTML debug report and markdown chunk analysis."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"directory_indexing_debug_{report_time}.html"
        markdown_file = self.output_dir / f"directory_chunks_analysis_{report_time}.md"

        html_content = self._generate_html_report()
        markdown_content = self._generate_markdown_chunks_analysis()

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Also generate a JSON file with raw data
        json_file = self.output_dir / f"directory_indexing_debug_{report_time}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.debug_data, f, indent=2, default=str)

        print(f"📄 Generated HTML report: {report_file}")
        print(f"📄 Generated Markdown chunk analysis: {markdown_file}")
        print(f"📄 Generated JSON data: {json_file}")

    def _generate_markdown_chunks_analysis(self) -> str:
        """Generate markdown file with chunk analysis for LLM review."""
        directory_info = self.debug_data.get("directory_info", {})
        loading = self.debug_data.get("loading", {})
        chunking = self.debug_data.get("chunking", {})
        timing = self.debug_data.get("timing", {})

        markdown = f"""# PyContextify Directory Chunk Analysis Report

**Directory:** {directory_info.get('directory_name', 'Unknown')}
**Path:** `{directory_info.get('directory_path', 'Unknown')}`
**Total Files:** {directory_info.get('total_files', 0)}
**Supported Files:** {directory_info.get('supported_files_count', 0)}
**Directory Size:** {directory_info.get('total_size_mb', 0)} MB ({directory_info.get('total_size_bytes', 0):,} bytes)
**Analysis Date:** {timing.get('end_time', '?')}

## Executive Summary

- **Loading Status:** {'✅ Success' if loading.get('success') else '❌ Failed'}
- **Chunking Status:** {'✅ Success' if chunking.get('success') else '❌ Failed'}
- **Total Processing Time:** {timing.get('total_duration_seconds', '?')}s
- **Documents Created:** {loading.get('documents_created', 0)}
- **Total Chunks Generated:** {chunking.get('output_chunks', 0)}

## Configuration Used

- **Chunk Size:** {chunking.get('config', {}).get('chunk_size', '?')} tokens
- **Chunk Overlap:** {chunking.get('config', {}).get('chunk_overlap', '?')} tokens
- **Relationships Enabled:** {chunking.get('config', {}).get('enable_relationships', False)}

## Directory Structure Analysis

### File Type Distribution

"""

        if directory_info.get("file_extensions"):
            extensions = directory_info["file_extensions"]
            sorted_extensions = sorted(extensions.items(), key=lambda x: x[1], reverse=True)
            for ext, count in sorted_extensions[:15]:  # Top 15 file types
                ext_name = ext if ext else "(no extension)"
                markdown += f"- **{ext_name}**: {count} files\n"

        markdown += f"""
### Supported Files (Top 20)

"""

        for file_info in directory_info.get("supported_files", [])[:20]:
            file_path = file_info["path"]
            file_size_kb = round(file_info["size"] / 1024, 1)
            markdown += f"- `{file_path}` ({file_size_kb} KB, {file_info['extension']})\n"

        if loading.get('success'):
            markdown += f"""
## File Processing Results

### File Types Processed

"""
            file_types = loading.get("file_types", {})
            for file_type, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                type_name = file_type if file_type else "(no extension)"
                markdown += f"- **{type_name}**: {count} documents\n"

        markdown += f"""
## Chunk Statistics

"""

        if chunking.get('success'):
            stats = chunking.get('statistics', {})
            markdown += f"""- **Total Chunk Characters:** {stats.get('total_chunk_characters', 0):,}
- **Average Chunk Size:** {stats.get('average_chunk_size', 0)} characters
- **Chunk Size Range:** {stats.get('min_chunk_size', 0)} - {stats.get('max_chunk_size', 0)} characters
- **Chunks per Document:** {stats.get('chunks_per_document', 0)}

### Chunk Distribution by File

"""
            
            file_distribution = chunking.get("file_chunk_distribution", {})
            for file_name, chunk_count in sorted(file_distribution.items(), key=lambda x: x[1], reverse=True)[:20]:
                markdown += f"- **{file_name}**: {chunk_count} chunks\n"

            markdown += """
## Sample Documents

> **Purpose:** This section shows a sample of processed documents to analyze conversion quality.

"""
            
            # Add sample documents
            for i, doc in enumerate(loading.get('documents', [])[:5]):  # First 5 documents
                markdown += f"""### Document {i+1}: {Path(doc.get('source_path', 'unknown')).name}

**Statistics:**
- Characters: {doc.get('character_count', 0):,}
- Words: {doc.get('word_count', 0):,}
- Lines: {doc.get('line_count', 0):,}
- File Type: {doc.get('file_type', 'unknown')}
- Code Detected: {'Yes' if doc.get('appears_to_be_code') else 'No'}

**Source Path:** `{doc.get('source_path', 'unknown')}`

**Content Preview:**
```{doc.get('file_type', 'text')}
{doc.get('text_preview', 'No preview available')}
```

---

"""

            markdown += """## Detailed Chunk Analysis

> **Purpose:** This section shows every chunk with clear boundaries for analyzing chunking quality.
> Look for:
> - Chunks that break in the middle of functions/classes
> - Chunks that split logical sections awkwardly
> - Missing context at chunk boundaries
> - Appropriate overlap between adjacent chunks from the same file

"""

            # Add all chunks with clear separation
            chunks = chunking.get('chunks', [])
            current_file = None
            
            for i, chunk in enumerate(chunks):
                chunk_num = i + 1
                file_name = chunk.get('file_name', 'unknown')
                
                # Add file separator when switching files
                if file_name != current_file:
                    current_file = file_name
                    markdown += f"""
## 📁 File: {file_name}

"""

                markdown += f"""### Chunk {chunk_num}

**Metadata:**
- **Index:** {chunk.get('chunk_index', i)}
- **File:** {file_name}
- **File Type:** {chunk.get('file_type', 'unknown')}
- **Characters:** {chunk.get('character_count', 0):,}
- **Words:** {chunk.get('word_count', 0):,}
- **Estimated Tokens:** {chunk.get('estimated_tokens', 0)}
- **Character Range:** {chunk.get('start_char', '?')} - {chunk.get('end_char', '?')}
- **Source:** `{chunk.get('source_path', 'Unknown')}`

**Content:**
```{chunk.get('file_type', 'text')}
{chunk.get('text_full', chunk.get('text_preview', 'No content available'))}
```

---

"""
        else:
            markdown += f"""**❌ Chunking Failed:** {chunking.get('error', 'Unknown error')}
**Failure Time:** {chunking.get('chunking_time_seconds', '?')}s

"""

        markdown += f"""## Analysis Notes

**What to look for when reviewing chunks:**

1. **Code Structure Preservation**: Do chunks break at natural boundaries (functions, classes, modules) or do they cut through code constructs?

2. **Context Preservation**: Does each chunk contain enough context to understand the code's purpose?

3. **Import/Dependency Handling**: Are import statements and dependencies preserved appropriately?

4. **Function/Class Completeness**: Are functions and classes kept together or split across chunks?

5. **Comment and Documentation Context**: Are comments and docstrings kept with their associated code?

**Performance Metrics:**
- Directory Loading: {loading.get('loading_time_seconds', '?')}s
- Chunking: {chunking.get('chunking_time_seconds', '?')}s
- Total: {timing.get('total_duration_seconds', '?')}s

**File Processing:**
- Total Files in Directory: {directory_info.get('total_files', 0)}
- Supported Code Files: {directory_info.get('supported_files_count', 0)}
- Documents Created: {loading.get('documents_created', 0)}
- Final Chunks: {chunking.get('output_chunks', 0)}

---
*Report generated by PyContextify Directory Debug Tool*
"""

        return markdown

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        directory_info = self.debug_data.get("directory_info", {})
        loading = self.debug_data.get("loading", {})
        chunking = self.debug_data.get("chunking", {})
        timing = self.debug_data.get("timing", {})

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PyContextify Directory Indexing Debug Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .section {{ border: 1px solid #e1e5e9; margin: 20px 0; border-radius: 6px; }}
        .section-header {{ background: #f8f9fa; padding: 15px; border-bottom: 1px solid #e1e5e9; }}
        .section-body {{ padding: 15px; }}
        .success {{ border-left: 4px solid #28a745; }}
        .error {{ border-left: 4px solid #dc3545; }}
        .info {{ border-left: 4px solid #007bff; }}
        .json-block {{ background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; margin: 10px 0; }}
        .chunk-preview {{ background: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; font-family: monospace; white-space: pre-line; }}
        .stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .stat-box {{ background: white; border: 1px solid #e1e5e9; padding: 15px; border-radius: 6px; min-width: 150px; }}
        pre {{ margin: 0; white-space: pre-wrap; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #e1e5e9; padding: 8px; text-align: left; }}
        th {{ background: #f8f9fa; }}
        .chunk-nav {{ position: sticky; top: 0; background: #f8f9fa; padding: 10px; border-bottom: 2px solid #007bff; z-index: 100; }}
        .chunk-content {{ font-size: 0.9em; line-height: 1.4; border-left: 3px solid #007bff; padding-left: 10px; }}
        .chunk-separator {{ border-top: 2px solid #e1e5e9; margin: 30px 0; }}
        .collapsible {{ background: #007bff; color: white; cursor: pointer; padding: 10px; border: none; text-align: left; outline: none; font-size: 14px; border-radius: 4px; margin: 10px 0; width: 100%; }}
        .collapsible:hover {{ background: #0056b3; }}
        .collapsible:after {{ content: '+'; font-weight: bold; float: right; margin-left: 5px; }}
        .collapsible.active:after {{ content: '-'; }}
        .collapsible-content {{ max-height: 0; overflow: hidden; transition: max-height 0.3s ease-out; background: #f8f9fa; border-radius: 4px; margin-bottom: 10px; }}
        .collapsible-content.show {{ max-height: none; padding: 15px; border: 1px solid #e1e5e9; }}
        .code-content {{ font-family: 'Consolas', 'Monaco', 'Courier New', monospace; line-height: 1.4; background: white; padding: 20px; border: 1px solid #e1e5e9; border-radius: 6px; }}
        .file-group {{ background: #f8f9fa; margin: 20px 0; padding: 15px; border-radius: 6px; border-left: 4px solid #6f42c1; }}
    </style>
    <script>
        function toggleCollapsible(element) {{
            element.classList.toggle('active');
            var content = element.nextElementSibling;
            content.classList.toggle('show');
        }}
        function expandAll() {{
            var collapsibles = document.getElementsByClassName('collapsible');
            var contents = document.getElementsByClassName('collapsible-content');
            for (var i = 0; i < collapsibles.length; i++) {{
                collapsibles[i].classList.add('active');
                contents[i].classList.add('show');
            }}
        }}
        function collapseAll() {{
            var collapsibles = document.getElementsByClassName('collapsible');
            var contents = document.getElementsByClassName('collapsible-content');
            for (var i = 0; i < collapsibles.length; i++) {{
                collapsibles[i].classList.remove('active');
                contents[i].classList.remove('show');
            }}
        }}
        document.addEventListener('DOMContentLoaded', function() {{
            var collapsibles = document.getElementsByClassName('collapsible');
            for (var i = 0; i < collapsibles.length; i++) {{
                collapsibles[i].addEventListener('click', function() {{
                    toggleCollapsible(this);
                }});
            }}
        }});
    </script>
</head>
<body>
    <div class="header">
        <h1>📂 PyContextify Directory Indexing Debug Report</h1>
        <div class="stats">
            <div class="stat-box">
                <strong>Directory:</strong><br>{directory_info.get('directory_name', 'Unknown')}
            </div>
            <div class="stat-box">
                <strong>Total Files:</strong><br>{directory_info.get('total_files', 0)}
            </div>
            <div class="stat-box">
                <strong>Supported Files:</strong><br>{directory_info.get('supported_files_count', 0)}
            </div>
            <div class="stat-box">
                <strong>Directory Size:</strong><br>{directory_info.get('total_size_mb', 0)} MB
            </div>
            <div class="stat-box">
                <strong>Total Duration:</strong><br>{timing.get('total_duration_seconds', '?')}s
            </div>
            <div style="margin-top: 20px; text-align: center;">
                <button onclick="expandAll()" style="background: #28a745; color: white; border: none; padding: 8px 16px; margin: 0 5px; border-radius: 4px; cursor: pointer;">🔽 Expand All Content</button>
                <button onclick="collapseAll()" style="background: #dc3545; color: white; border: none; padding: 8px 16px; margin: 0 5px; border-radius: 4px; cursor: pointer;">🔼 Collapse All Content</button>
            </div>
        </div>
    </div>
"""

        # Directory Info Section
        html += f"""
    <div class="section info">
        <div class="section-header">
            <h2>📊 Directory Structure Analysis</h2>
        </div>
        <div class="section-body">
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Directory Path</td><td>{directory_info.get('directory_path', '')}</td></tr>
                <tr><td>Total Files</td><td>{directory_info.get('total_files', 0):,}</td></tr>
                <tr><td>Supported Code Files</td><td>{directory_info.get('supported_files_count', 0)}</td></tr>
                <tr><td>Unsupported Files</td><td>{directory_info.get('unsupported_files_count', 0)}</td></tr>
                <tr><td>Directory Size</td><td>{directory_info.get('total_size_bytes', 0):,} bytes ({directory_info.get('total_size_mb', 0)} MB)</td></tr>
            </table>
            
            <h4>📈 File Type Distribution:</h4>
            <table>
                <tr><th>Extension</th><th>Count</th></tr>
"""

        # Add file extensions
        extensions = directory_info.get("file_extensions", {})
        sorted_extensions = sorted(extensions.items(), key=lambda x: x[1], reverse=True)
        for ext, count in sorted_extensions[:20]:  # Top 20 extensions
            ext_name = ext if ext else "(no extension)"
            html += f"<tr><td>{ext_name}</td><td>{count}</td></tr>"

        html += """
            </table>
        </div>
    </div>
"""

        # Loading Section
        loading_class = "success" if loading.get("success") else "error"
        loading_icon = "✅" if loading.get("success") else "❌"

        html += f"""
    <div class="section {loading_class}">
        <div class="section-header">
            <h2>{loading_icon} Directory Loading and Document Conversion</h2>
        </div>
        <div class="section-body">
"""

        if loading.get("success"):
            html += f"""
            <div class="stats">
                <div class="stat-box">
                    <strong>Loading Time:</strong><br>{loading.get('loading_time_seconds', '?')}s
                </div>
                <div class="stat-box">
                    <strong>Documents Created:</strong><br>{loading.get('documents_created', 0)}
                </div>
                <div class="stat-box">
                    <strong>Total Characters:</strong><br>{loading.get('total_characters', 0):,}
                </div>
            </div>
            
            <h4>📄 Document Analysis:</h4>
"""

            # Show sample documents
            for doc in loading.get("documents", [])[:5]:  # Show first 5 documents
                file_name = Path(doc.get('source_path', 'unknown')).name
                html += f"""
            <div class="file-group">
                <h5>📄 {file_name}</h5>
                <div class="stats">
                    <div class="stat-box">
                        <strong>Characters:</strong><br>{doc.get('character_count', 0):,}
                    </div>
                    <div class="stat-box">
                        <strong>Words:</strong><br>{doc.get('word_count', 0):,}
                    </div>
                    <div class="stat-box">
                        <strong>Lines:</strong><br>{doc.get('line_count', 0):,}
                    </div>
                    <div class="stat-box">
                        <strong>File Type:</strong><br>{doc.get('file_type', 'unknown')}
                    </div>
                    <div class="stat-box">
                        <strong>Code:</strong><br>{'Yes' if doc.get('appears_to_be_code') else 'No'}
                    </div>
                </div>
                
                <h6>Content Preview:</h6>
                <div class="chunk-preview">{doc.get('text_preview', '')}</div>
                
                <button class="collapsible">💻 View Full File Content ({doc.get('character_count', 0):,} chars)</button>
                <div class="collapsible-content">
                    <div class="code-content">
                        <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word;">{doc.get('text_full', 'No content available')}</pre>
                    </div>
                </div>
            </div>
"""
        else:
            html += f"""
            <p><strong>❌ Loading failed:</strong> {loading.get('error', 'Unknown error')}</p>
            <p><strong>Time to failure:</strong> {loading.get('loading_time_seconds', '?')}s</p>
"""

        html += "</div></div>"

        # Chunking Section
        chunking_class = "success" if chunking.get("success") else "error"
        chunking_icon = "✅" if chunking.get("success") else "❌"

        html += f"""
    <div class="section {chunking_class}">
        <div class="section-header">
            <h2>{chunking_icon} Text Chunking Analysis</h2>
        </div>
        <div class="section-body">
"""

        if chunking.get("success"):
            stats = chunking.get("statistics", {})
            config = chunking.get("config", {})

            html += f"""
            <div class="stats">
                <div class="stat-box">
                    <strong>Chunking Time:</strong><br>{chunking.get('chunking_time_seconds', '?')}s
                </div>
                <div class="stat-box">
                    <strong>Input Docs:</strong><br>{chunking.get('input_documents', 0)}
                </div>
                <div class="stat-box">
                    <strong>Output Chunks:</strong><br>{chunking.get('output_chunks', 0)}
                </div>
                <div class="stat-box">
                    <strong>Avg Chunk Size:</strong><br>{stats.get('average_chunk_size', 0)} chars
                </div>
            </div>
            
            <h4>🔧 Chunking Configuration:</h4>
            <table>
                <tr><th>Setting</th><th>Value</th></tr>
                <tr><td>Chunk Size</td><td>{config.get('chunk_size', '?')} tokens</td></tr>
                <tr><td>Overlap</td><td>{config.get('chunk_overlap', '?')} tokens</td></tr>
                <tr><td>Relationships Enabled</td><td>{config.get('enable_relationships', False)}</td></tr>
            </table>
            
            <h4>📊 Chunk Statistics:</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Characters</td><td>{stats.get('total_chunk_characters', 0):,}</td></tr>
                <tr><td>Average Size</td><td>{stats.get('average_chunk_size', 0)} characters</td></tr>
                <tr><td>Size Range</td><td>{stats.get('min_chunk_size', 0)} - {stats.get('max_chunk_size', 0)} characters</td></tr>
                <tr><td>Chunks per Document</td><td>{stats.get('chunks_per_document', 0)}</td></tr>
            </table>
            
            <h4>📈 Chunk Distribution by File:</h4>
            <table>
                <tr><th>File Name</th><th>Chunks</th></tr>
"""

            # Add chunk distribution
            file_distribution = chunking.get("file_chunk_distribution", {})
            for file_name, chunk_count in sorted(file_distribution.items(), key=lambda x: x[1], reverse=True)[:15]:
                html += f"<tr><td>{file_name}</td><td>{chunk_count}</td></tr>"

            html += f"""
            </table>
            
            <h4>✂️ All Chunks ({len(chunking.get('chunks', []))}):</h4>
            <div style="margin-bottom: 20px; padding: 10px; background: #e3f2fd; border-radius: 4px;">
                <p><strong>📋 Complete Chunk Review:</strong> This section shows all chunks organized by file for comprehensive review.</p>
            </div>
"""

            # Group chunks by file for better organization
            chunks_by_file = defaultdict(list)
            for chunk in chunking.get('chunks', []):
                file_name = chunk.get('file_name', 'unknown')
                chunks_by_file[file_name].append(chunk)

            for file_name, file_chunks in chunks_by_file.items():
                html += f"""
            <div class="file-group">
                <h5>📁 {file_name} ({len(file_chunks)} chunks)</h5>
"""
                for chunk in file_chunks:
                    chunk_num = chunk.get("chunk_index", 0) + 1
                    html += f"""
                <div style="margin: 15px 0; padding: 15px; border: 1px solid #e1e5e9; border-radius: 6px; background: white;">
                    <h6 style="color: #0366d6; margin-top: 0;">✂️ Chunk {chunk_num}</h6>
                    <div class="stats">
                        <div class="stat-box">
                            <strong>Characters:</strong><br>{chunk.get('character_count', 0):,}
                        </div>
                        <div class="stat-box">
                            <strong>Words:</strong><br>{chunk.get('word_count', 0):,}
                        </div>
                        <div class="stat-box">
                            <strong>Est. Tokens:</strong><br>{chunk.get('estimated_tokens', 0)}
                        </div>
                        <div class="stat-box">
                            <strong>Position:</strong><br>{chunk.get('start_char', '?')} - {chunk.get('end_char', '?')}
                        </div>
                    </div>
                    
                    <div class="chunk-preview">{chunk.get('visual_preview', '')}</div>
                    
                    <button class="collapsible">💻 View Full Chunk Content ({chunk.get('character_count', 0)} chars)</button>
                    <div class="collapsible-content">
                        <div class="code-content">
                            <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word;">{chunk.get('text_full', chunk.get('text_preview', ''))}</pre>
                        </div>
                    </div>
                </div>
"""
                html += "</div>"

        else:
            html += f"""
            <p><strong>❌ Chunking failed:</strong> {chunking.get('error', 'Unknown error')}</p>
            <p><strong>Time to failure:</strong> {chunking.get('chunking_time_seconds', '?')}s</p>
"""

        html += "</div></div>"

        # Timing Summary
        html += f"""
    <div class="section info">
        <div class="section-header">
            <h2>⏱️ Timing Summary</h2>
        </div>
        <div class="section-body">
            <table>
                <tr><th>Phase</th><th>Duration</th></tr>
                <tr><td>Directory Loading</td><td>{loading.get('loading_time_seconds', '?')}s</td></tr>
                <tr><td>Text Chunking</td><td>{chunking.get('chunking_time_seconds', '?')}s</td></tr>
                <tr><td><strong>Total</strong></td><td><strong>{timing.get('total_duration_seconds', '?')}s</strong></td></tr>
            </table>
            
            <p><small>Report generated at: {timing.get('end_time', '?')}</small></p>
        </div>
    </div>

</body>
</html>
"""
        return html


def main():
    """Main entry point for the debug script."""
    parser = argparse.ArgumentParser(
        description="Debug PyContextify directory indexing pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./.debug"),
        help="Output directory for debug reports",
    )
    parser.add_argument(
        "--dir-path",
        type=Path,
        default=Path("pycontextify"),
        help="Path to directory to analyze (relative to project root)",
    )

    args = parser.parse_args()

    # Resolve directory path relative to project root if it's a relative path
    dir_path = args.dir_path
    if not dir_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        dir_path = project_root / dir_path

    debugger = DirectoryIndexingDebugger(args.output_dir, dir_path)
    debugger.debug_directory_pipeline()


if __name__ == "__main__":
    main()