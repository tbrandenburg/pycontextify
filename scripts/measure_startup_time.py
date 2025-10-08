#!/usr/bin/env python3
"""Script to measure PyContextify MCP server startup performance."""

import tempfile
import time
from pathlib import Path

from pycontextify.config import Config
from pycontextify.indexer import IndexManager


def measure_startup_performance():
    """Measure different aspects of server startup."""

    print("🔍 Measuring PyContextify MCP Server Startup Performance...\n")

    # Test 1: Config initialization
    print("1️⃣ Config initialization...")
    start_time = time.time()
    config = Config()
    config_time = time.time() - start_time
    print(f"   ⏱️ Config init: {config_time:.2f}s")

    # Test 2: IndexManager initialization (full server startup)
    print("\n2️⃣ IndexManager (MCP Server) initialization...")

    # Use a temporary directory to avoid persistence overhead
    with tempfile.TemporaryDirectory() as temp_dir:
        config.index_dir = Path(temp_dir)
        config.auto_persist = False
        config.auto_load = False

        start_time = time.time()
        manager = IndexManager(config)
        init_time = time.time() - start_time
        print(f"   ⏱️ IndexManager init: {init_time:.2f}s")

        # Test 3: First embedding operation (model loading)
        print("\n3️⃣ First embedding operation (model loading)...")
        start_time = time.time()
        try:
            # This will trigger model loading
            test_embedding = manager.embedder.embed_single("test text")
            embedding_time = time.time() - start_time
            print(f"   ⏱️ First embedding: {embedding_time:.2f}s")
            print(f"   📊 Embedding dimension: {len(test_embedding)}")
        except Exception as e:
            print(f"   ❌ Embedding failed: {e}")

        # Test 4: MCP operations response time
        print("\n4️⃣ MCP operations response time...")

        # Status operation
        start_time = time.time()
        status = manager.get_status()
        status_time = time.time() - start_time
        print(f"   ⏱️ Status query: {status_time:.2f}s")

        # Simple document indexing
        print("\n5️⃣ Document indexing performance...")
        test_content = "This is a test document for performance measurement."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(test_content)
            temp_file = f.name

        try:
            start_time = time.time()
            result = manager.index_document(temp_file)
            indexing_time = time.time() - start_time
            print(f"   ⏱️ Document indexing: {indexing_time:.2f}s")
            print(f"   📄 Chunks added: {result.get('chunks_added', 0)}")
        except Exception as e:
            print(f"   ❌ Indexing failed: {e}")
        finally:
            try:
                Path(temp_file).unlink()
            except:
                pass

        # Search operation
        print("\n6️⃣ Search operation performance...")
        try:
            start_time = time.time()
            search_results = manager.search("test document", top_k=3)
            search_time = time.time() - start_time
            print(f"   ⏱️ Search query: {search_time:.2f}s")
            print(f"   🔍 Results found: {len(search_results)}")
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

        # Cleanup
        try:
            manager.embedder.cleanup()
        except:
            pass

    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY:")
    print("=" * 60)
    print(f"Config initialization:     {config_time:.2f}s")
    print(f"IndexManager init:         {init_time:.2f}s")
    try:
        print(f"Model loading:             {embedding_time:.2f}s")
        print(
            f"Total startup time:        {config_time + init_time + embedding_time:.2f}s"
        )
    except:
        print(f"Total startup time:        {config_time + init_time:.2f}s")
    print(f"Status query:              {status_time:.2f}s")
    try:
        print(f"Document indexing:         {indexing_time:.2f}s")
        print(f"Search operation:          {search_time:.2f}s")
    except:
        pass

    print("\n🎯 PERFORMANCE TARGETS:")
    print("✅ Good: < 2s total startup")
    print("⚠️  Acceptable: 2-5s total startup")
    print("❌ Poor: > 5s total startup")


if __name__ == "__main__":
    measure_startup_performance()
