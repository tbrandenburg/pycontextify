import tempfile
import time
from pathlib import Path

from pycontextify.index.config import Config
from pycontextify.index.manager import IndexManager


def debug_index_manager():
    """Debug IndexManager initialization step by step."""

    print("🐛 Debugging IndexManager lazy loading...\n")

    # Create config with explicit simple model using config_overrides
    config_overrides = {
        "embedding_provider": "sentence_transformers",
        "embedding_model": "all-MiniLM-L6-v2",  # Simpler, more cached model
        "auto_load": False,
        "auto_persist": False,
    }
    config = Config(config_overrides=config_overrides)
    config.use_hybrid_search = False  # Disable for debugging

    with tempfile.TemporaryDirectory() as temp_dir:
        config.index_dir = Path(temp_dir)

        print("1. Starting IndexManager initialization...")
        start_time = time.time()

        try:
            manager = IndexManager(config)
            init_time = time.time() - start_time
            print(f"   ✅ IndexManager init: {init_time:.2f}s")

            # Check lazy loading state
            print(f"   📊 Embedder initialized: {manager._embedder_initialized}")
            print(f"   📊 Embedder object: {manager.embedder}")
            print(f"   📊 Vector store: {manager.vector_store}")

            # Test status (should not trigger embedder loading)
            print("\n2. Testing status query...")
            start_time = time.time()
            status = manager.get_status()
            status_time = time.time() - start_time
            print(f"   ✅ Status query: {status_time:.2f}s")
            print(f"   📊 Embedding info: {status['embedding']}")

            # Test first embedding operation (should trigger loading)
            print("\n3. Testing first embedding operation...")
            start_time = time.time()
            try:
                embedding = manager.embedder.embed_single("test")
                embed_time = time.time() - start_time
                print(f"   ✅ First embedding: {embed_time:.2f}s")
                print(f"   📊 Embedding dimension: {len(embedding)}")
                print(f"   📊 Embedder initialized: {manager._embedder_initialized}")
            except Exception as e:
                print(f"   ❌ First embedding failed: {e}")
                print(f"   📊 Embedder initialized: {manager._embedder_initialized}")
                print(f"   📊 Embedder object: {manager.embedder}")

                # Try to manually trigger lazy loading
                print("\n4. Manually triggering lazy loading...")
                start_time = time.time()
                try:
                    manager._ensure_embedder_loaded()
                    manual_time = time.time() - start_time
                    print(f"   ✅ Manual embedder loading: {manual_time:.2f}s")
                    print(f"   📊 Embedder now: {manager.embedder}")
                    print(f"   📊 Vector store now: {manager.vector_store}")
                except Exception as e:
                    print(f"   ❌ Manual loading failed: {e}")

        except Exception as e:
            init_time = time.time() - start_time
            print(f"   ❌ IndexManager init failed after {init_time:.2f}s: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    debug_index_manager()
