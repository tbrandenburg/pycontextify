import tempfile
import time
from pathlib import Path

from pycontextify.orchestrator_config import Config
from pycontextify.indexer_manager import IndexManager


def test_fast_startup_config():
    """Test optimized configuration for fast startup."""

    print("Testing FAST STARTUP configuration...\n")

    # Create optimized config
    config = Config()

    # Optimization 1: Disable expensive components
    config.use_hybrid_search = False  # Minor savings
    config.auto_load = False  # Don't load existing indices
    config.auto_persist = False  # Don't persist during tests

    # Optimization 2: Use faster model
    config.embedding_model = (
        "all-MiniLM-L6-v2"  # 384 dim, faster than all-mpnet-base-v2
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        config.index_dir = Path(temp_dir)

        # Measure startup time
        print("Initializing optimized MCP server...")
        start_time = time.time()
        manager = IndexManager(config)
        startup_time = time.time() - start_time

        print(f"✅ Optimized startup time: {startup_time:.2f}s")

        # Test basic operations
        print("\nTesting operations...")

        # Status query
        start_time = time.time()
        status = manager.get_status()
        status_time = time.time() - start_time
        print(f"✅ Status query: {status_time:.2f}s")

        # First embedding (this will trigger model loading)
        print("First embedding operation (model loading)...")
        start_time = time.time()
        embedding = manager.embedder.embed_single("test")
        first_embed_time = time.time() - start_time
        print(f"✅ First embedding: {first_embed_time:.2f}s")

        # Second embedding (should be fast)
        start_time = time.time()
        embedding2 = manager.embedder.embed_single("another test")
        second_embed_time = time.time() - start_time
        print(f"✅ Second embedding: {second_embed_time:.2f}s")

        print(f"\n{'='*50}")
        print("FAST STARTUP RESULTS:")
        print(f"{'='*50}")
        print(f"Server startup:        {startup_time:.2f}s")
        print(f"Status query:          {status_time:.2f}s")
        print(f"First embedding:       {first_embed_time:.2f}s")
        print(f"Subsequent embedding:  {second_embed_time:.2f}s")
        print(f"Total ready time:      {startup_time + first_embed_time:.2f}s")

        if startup_time < 2:
            print("\n🚀 EXCELLENT: Server starts in < 2s!")
        elif startup_time < 5:
            print("\n✅ GOOD: Server starts in < 5s")
        else:
            print("\n⚠️ NEEDS WORK: Server still slow to start")


def test_lazy_loading_concept():
    """Demonstrate lazy loading concept."""

    print("\n" + "=" * 60)
    print("LAZY LOADING CONCEPT DEMONSTRATION")
    print("=" * 60)

    # This shows how we could implement lazy loading
    config = Config()
    config.use_hybrid_search = False
    config.auto_load = False
    config.auto_persist = False

    with tempfile.TemporaryDirectory() as temp_dir:
        config.index_dir = Path(temp_dir)

        print("1. Quick server initialization (no model loading)...")
        start_time = time.time()

        # Initialize basic components only
        from pycontextify.storage_metadata import MetadataStore
        from pycontextify.storage_vector import VectorStore

        metadata_store = MetadataStore()

        quick_init_time = time.time() - start_time
        print(f"✅ Quick init: {quick_init_time:.2f}s")

        print("2. Server ready for basic operations (status, config, etc.)")
        print("3. Model loading happens only when embedding is needed")

        print(f"\nWith lazy loading:")
        print(f"- Server ready: {quick_init_time:.2f}s")
        print(f"- First embedding: ~15-20s (one-time cost)")
        print(f"- Subsequent ops: <0.1s")


if __name__ == "__main__":
    test_fast_startup_config()
    test_lazy_loading_concept()
