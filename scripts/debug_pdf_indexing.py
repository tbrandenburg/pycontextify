#!/usr/bin/env python3
"""
PyContextify PDF Indexing Debug Script

This script analyzes how the PDF indexing pipeline converts PDF files to markdown
and chunks them for indexing. It provides detailed reports on the conversion
process, chunking strategies, and metadata extraction.

Usage:
    python debug_pdf_indexing.py [--output-dir PATH] [--pdf-path PATH]

Examples:
    python debug_pdf_indexing.py
    python debug_pdf_indexing.py --pdf-path tests/resources/Automotive_SPICE_PAM_30.pdf
    python debug_pdf_indexing.py --output-dir ./debug_reports
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pycontextify.chunker import ChunkerFactory
from pycontextify.config import Config
from pycontextify.loader import FileLoaderFactory
from pycontextify.types import SourceType


class PDFIndexingDebugger:
    """Debug tool for analyzing PDF-to-chunks conversion pipeline."""

    def __init__(self, output_dir: Path, pdf_path: Path):
        self.output_dir = output_dir
        self.pdf_path = pdf_path
        self.start_time = datetime.now()
        self.debug_data: Dict[str, Any] = {}

    def debug_pdf_pipeline(self) -> None:
        """Run complete PDF processing pipeline with debugging."""
        print(f"🔍 Starting PDF indexing debug at {self.start_time}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📄 PDF file: {self.pdf_path}")

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        self.debug_data["pdf_info"] = self._analyze_pdf_info()
        self.debug_data["loading"] = self._debug_loading()
        self.debug_data["chunking"] = self._debug_chunking()
        self.debug_data["timing"] = self._finalize_timing()

        self._generate_report()
        print(f"✅ Debug completed. Report saved to: {self.output_dir}")

    def _analyze_pdf_info(self) -> Dict[str, Any]:
        """Analyze basic PDF file information."""
        print("\n📊 Step 1: Analyzing PDF file info")

        pdf_info = {
            "file_path": str(self.pdf_path),
            "file_size_bytes": self.pdf_path.stat().st_size,
            "file_size_mb": round(self.pdf_path.stat().st_size / (1024 * 1024), 2),
            "filename": self.pdf_path.name,
            "extension": self.pdf_path.suffix,
        }

        # Try to get PDF page count if possible
        try:
            import pymupdf  # type: ignore[import]

            with pymupdf.open(str(self.pdf_path)) as pdf_doc:  # type: ignore[attr-defined]
                pdf_info["total_pages"] = pdf_doc.page_count
                print(
                    f"📄 PDF has {pdf_info['total_pages']} pages ({pdf_info['file_size_mb']} MB)"
                )
        except ImportError:
            print("📄 PyMuPDF not available - cannot determine page count")
            pdf_info["total_pages"] = "unknown"
        except Exception as e:
            print(f"⚠️  Could not read PDF metadata: {e}")
            pdf_info["total_pages"] = "error"

        return pdf_info

    def _debug_loading(self) -> Dict[str, Any]:
        """Debug the PDF loading and markdown conversion process."""
        print("\n🔄 Step 2: Loading and converting PDF to markdown")

        loading_start = time.time()

        # Create loader and attempt to load the PDF
        loader = FileLoaderFactory(default_encoding="utf-8")
        tags = "automotive-spice,pam,process-assessment,documentation,debug"

        try:
            # Load the PDF using the existing pipeline
            normalized_docs = loader.load(
                path=str(self.pdf_path),
                tags=tags,
                base_path=str(self.pdf_path.parent),
            )

            loading_time = time.time() - loading_start

            loading_debug = {
                "success": True,
                "loading_time_seconds": round(loading_time, 2),
                "documents_created": len(normalized_docs),
                "tags_applied": tags.split(","),
                "conversion_method": "unknown",
                "documents": [],
            }

            print(f"✅ Loaded {len(normalized_docs)} documents in {loading_time:.2f}s")

            # Analyze each document
            total_chars = 0
            for i, doc in enumerate(normalized_docs):
                text = doc["text"]
                metadata = doc["metadata"]

                # Detect conversion method
                if "pdf_loader" in metadata:
                    loading_debug["conversion_method"] = metadata["pdf_loader"]

                doc_info = {
                    "document_index": i,
                    "character_count": len(text),
                    "word_count": len(text.split()),
                    "line_count": len(text.split("\n")),
                    "metadata": metadata,
                    "text_preview": text[:500] + "..." if len(text) > 500 else text,
                    "text_sample_end": text[-200:] if len(text) > 200 else "",
                    "text_full": text,  # Store full converted markdown
                }

                # Look for markdown indicators
                markdown_indicators = {
                    "headers": text.count("#"),
                    "bold_text": text.count("**"),
                    "tables": text.count("|"),
                    "links": text.count("["),
                    "code_blocks": text.count("```"),
                }
                doc_info["markdown_indicators"] = markdown_indicators
                doc_info["appears_to_be_markdown"] = any(markdown_indicators.values())

                loading_debug["documents"].append(doc_info)
                total_chars += len(text)

                print(
                    f"  📄 Doc {i+1}: {len(text):,} chars, {len(text.split()):,} words"
                )
                if doc_info["appears_to_be_markdown"]:
                    print(f"     📝 Markdown detected: {markdown_indicators}")

            loading_debug["total_characters"] = total_chars
            print(f"📊 Total content: {total_chars:,} characters")

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
        tags = "automotive-spice,pam,process-assessment,documentation,debug"

        try:
            normalized_docs = loader.load(
                path=str(self.pdf_path),
                tags=tags,
                base_path=str(self.pdf_path.parent),
            )

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
            }

            print(f"✅ Created {len(chunks)} chunks in {chunking_time:.2f}s")

            # Analyze each chunk
            total_chunk_chars = 0
            chunk_sizes = []

            for i, chunk in enumerate(chunks):
                chunk_text = chunk["text"]
                chunk_metadata = chunk["metadata"]

                chunk_info = {
                    "chunk_index": i,
                    "character_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "estimated_tokens": round(len(chunk_text.split()) * 1.3),
                    "start_char": chunk_metadata.get("start_char"),
                    "end_char": chunk_metadata.get("end_char"),
                    "source_path": chunk_metadata.get("full_path"),
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
                        f"┌── CHUNK {i+1} ──────────────────────────────",
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

                # Print every 10th chunk + first few + last few
                if i < 3 or i >= len(chunks) - 3 or i % 10 == 0:
                    print(
                        f"  ✂️  Chunk {i+1:3d}: {len(chunk_text):4d} chars (~{chunk_info['estimated_tokens']:3d} tokens)"
                    )

            # Calculate statistics
            chunking_debug["statistics"] = {
                "total_chunk_characters": total_chunk_chars,
                "average_chunk_size": round(sum(chunk_sizes) / len(chunk_sizes)),
                "min_chunk_size": min(chunk_sizes),
                "max_chunk_size": max(chunk_sizes),
                "chunks_per_document": round(len(chunks) / len(normalized_docs), 1),
            }

            print(f"📊 Chunk statistics:")
            stats = chunking_debug["statistics"]
            print(f"   • Average size: {stats['average_chunk_size']} chars")
            print(
                f"   • Size range: {stats['min_chunk_size']} - {stats['max_chunk_size']} chars"
            )
            print(f"   • Chunks per document: {stats['chunks_per_document']}")

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
        """Generate comprehensive HTML debug report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"pdf_indexing_debug_{report_time}.html"

        html_content = self._generate_html_report()

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Also generate a JSON file with raw data
        json_file = self.output_dir / f"pdf_indexing_debug_{report_time}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.debug_data, f, indent=2, default=str)

        print(f"📄 Generated HTML report: {report_file}")
        print(f"📄 Generated JSON data: {json_file}")

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        pdf_info = self.debug_data.get("pdf_info", {})
        loading = self.debug_data.get("loading", {})
        chunking = self.debug_data.get("chunking", {})
        timing = self.debug_data.get("timing", {})

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PyContextify PDF Indexing Debug Report</title>
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
        .markdown-content {{ font-family: 'Segoe UI', system-ui, sans-serif; line-height: 1.6; background: white; padding: 20px; border: 1px solid #e1e5e9; border-radius: 6px; }}
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
        <h1>📄 PyContextify PDF Indexing Debug Report</h1>
        <div class="stats">
            <div class="stat-box">
                <strong>PDF File:</strong><br>{pdf_info.get('filename', 'Unknown')}
            </div>
            <div class="stat-box">
                <strong>File Size:</strong><br>{pdf_info.get('file_size_mb', '?')} MB
            </div>
            <div class="stat-box">
                <strong>Pages:</strong><br>{pdf_info.get('total_pages', '?')}
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

        # PDF Info Section
        html += f"""
    <div class="section info">
        <div class="section-header">
            <h2>📊 PDF File Information</h2>
        </div>
        <div class="section-body">
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>File Path</td><td>{pdf_info.get('file_path', '')}</td></tr>
                <tr><td>File Size</td><td>{pdf_info.get('file_size_bytes', 0):,} bytes ({pdf_info.get('file_size_mb', 0)} MB)</td></tr>
                <tr><td>Total Pages</td><td>{pdf_info.get('total_pages', 'Unknown')}</td></tr>
                <tr><td>Extension</td><td>{pdf_info.get('extension', '')}</td></tr>
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
            <h2>{loading_icon} PDF Loading and Markdown Conversion</h2>
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
                <div class="stat-box">
                    <strong>Conversion Method:</strong><br>{loading.get('conversion_method', 'Unknown')}
                </div>
            </div>
            
            <h4>📄 Documents Analysis:</h4>
"""

            for doc in loading.get("documents", [])[:3]:  # Show first 3 documents
                html += f"""
            <div style="margin: 15px 0; padding: 15px; border: 1px solid #e1e5e9; border-radius: 4px;">
                <h5>Document {doc.get('document_index', '?') + 1}</h5>
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
                        <strong>Markdown:</strong><br>{'Yes' if doc.get('appears_to_be_markdown') else 'No'}
                    </div>
                </div>
                
                <h6>Content Preview:</h6>
                <div class="chunk-preview">{doc.get('text_preview', '')}</div>
                
                <button class="collapsible">📄 View Full Converted Markdown ({doc.get('character_count', 0):,} chars)</button>
                <div class="collapsible-content">
                    <div class="markdown-content">
                        <pre style="font-family: 'Segoe UI', system-ui, sans-serif; white-space: pre-wrap; word-wrap: break-word; margin: 0;">{doc.get('text_full', 'No content available')}</pre>
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
            
            <h4>✂️ All Chunks ({len(chunking.get('chunks', []))}):</h4>
            <div style="margin-bottom: 20px; padding: 10px; background: #e3f2fd; border-radius: 4px;">
                <p><strong>📋 Complete Chunk Review:</strong> This section shows all chunks with their full content for comprehensive review. Use browser search (Ctrl+F) to find specific content.</p>
            </div>
"""

            # Add chunk navigation
            chunks_list = chunking.get("chunks", [])
            if len(chunks_list) > 10:  # Only show navigation if there are many chunks
                html += f"""
            <div class="chunk-nav">
                <strong>📍 Quick Navigation:</strong> 
                {' | '.join([f'<a href="#chunk-{i+1}">#{i+1}</a>' for i in range(0, len(chunks_list), 10)])}
            </div>
"""

            for chunk in chunks_list:  # Show ALL chunks
                chunk_num = chunk.get("chunk_index", 0) + 1
                html += f"""
            <div id="chunk-{chunk_num}" style="margin: 20px 0; padding: 20px; border: 1px solid #e1e5e9; border-radius: 6px; background: #fafbfc;">
                <h5 style="color: #0366d6; margin-top: 0;">📄 Chunk {chunk_num}</h5>
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
                
                <button class="collapsible">📝 View Full Chunk Content ({chunk.get('character_count', 0)} chars)</button>
                <div class="collapsible-content">
                    <div class="chunk-content" style="background: white; padding: 15px; border-radius: 4px; font-family: 'Segoe UI', system-ui, sans-serif; line-height: 1.6; border-left: 4px solid #0366d6;">
                        <pre style="font-family: inherit; white-space: pre-wrap; word-wrap: break-word; margin: 0;">{chunk.get('text_full', chunk.get('text_preview', ''))}</pre>
                    </div>
                </div>
                <div class="chunk-separator"></div>
            </div>
"""
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
                <tr><td>PDF Loading</td><td>{loading.get('loading_time_seconds', '?')}s</td></tr>
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
        description="Debug PyContextify PDF indexing pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./.debug"),
        help="Output directory for debug reports",
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=Path("tests/resources/Automotive_SPICE_PAM_30.pdf"),
        help="Path to PDF file to analyze (relative to project root)",
    )

    args = parser.parse_args()

    # Resolve PDF path relative to project root if it's a relative path
    pdf_path = args.pdf_path
    if not pdf_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        pdf_path = project_root / pdf_path

    debugger = PDFIndexingDebugger(args.output_dir, pdf_path)
    debugger.debug_pdf_pipeline()


if __name__ == "__main__":
    main()
