import time
import tempfile
from pathlib import Path
from pycontextify.index.config import Config

def time_operation(description, func):
    """Time a specific operation."""
    print(f"  {description}...")
    start_time = time.time()
    result = func()
    elapsed = time.time() - start_time
    print(f"    ⏱️ {description}: {elapsed:.2f}s")
    return result, elapsed

def main():
    print("Detailed PyContextify Performance Analysis...\n")
    
    total_time = 0
    
    # Config initialization
    print("1. Config initialization...")
    config, config_time = time_operation("Config creation", lambda: Config())
    total_time += config_time
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config.index_dir = Path(temp_dir)
        config.auto_persist = False
        config.auto_load = False
        
        print("\n2. IndexManager components initialization...")
        
        # Initialize basic stores
        from pycontextify.index.metadata import MetadataStore
        from pycontextify.index.relationship_store import RelationshipStore
        
        metadata_store, metadata_time = time_operation("MetadataStore", lambda: MetadataStore())
        relationship_store, relationship_time = time_operation("RelationshipStore", lambda: RelationshipStore())
        total_time += metadata_time + relationship_time
        
        # Initialize embedder
        print("\n3. Embedder initialization...")
        from pycontextify.index.embedders import EmbedderFactory
        
        def create_embedder():
            embedding_config = config.get_embedding_config()
            return EmbedderFactory.create_embedder(
                provider=embedding_config["provider"],
                model_name=embedding_config["model"]
            )
        
        embedder, embedder_time = time_operation("Embedder creation", create_embedder)
        total_time += embedder_time
        
        # Get dimension (potential model loading)
        dimension, dimension_time = time_operation("Get dimension", lambda: embedder.get_dimension())
        total_time += dimension_time
        
        # Initialize vector store
        print("\n4. Vector store initialization...")
        from pycontextify.index.vector_store import VectorStore
        
        vector_store, vector_time = time_operation("VectorStore", lambda: VectorStore(dimension, config))
        total_time += vector_time
        
        # Initialize hybrid search
        print("\n5. Hybrid search initialization...")
        if config.use_hybrid_search:
            try:
                from pycontextify.index.hybrid_search import HybridSearchEngine
                hybrid_search, hybrid_time = time_operation("HybridSearchEngine", 
                    lambda: HybridSearchEngine(keyword_weight=config.keyword_weight))
                total_time += hybrid_time
            except ImportError as e:
                print(f"    ❌ Hybrid search unavailable: {e}")
                hybrid_time = 0
        else:
            print("    ⏭️ Hybrid search disabled")
            hybrid_time = 0
        
        # Initialize reranker
        print("\n6. Reranker initialization...")
        reranker_time = 0
        if config.use_reranking:
            try:
                from pycontextify.index.reranker import CrossEncoderReranker
                
                def create_and_warmup_reranker():
                    reranker = CrossEncoderReranker(model_name=config.reranking_model)
                    if reranker.is_available:
                        reranker.warmup()
                    return reranker
                
                reranker, reranker_time = time_operation("CrossEncoderReranker + warmup", create_and_warmup_reranker)
                total_time += reranker_time
            except ImportError as e:
                print(f"    ❌ Reranker unavailable: {e}")
                reranker_time = 0
        else:
            print("    ⏭️ Reranker disabled")
        
        print(f"\n{'='*50}")
        print("DETAILED TIMING BREAKDOWN:")
        print(f"{'='*50}")
        print(f"Config initialization:     {config_time:.2f}s")
        print(f"MetadataStore:             {metadata_time:.2f}s")
        print(f"RelationshipStore:         {relationship_time:.2f}s")
        print(f"Embedder creation:         {embedder_time:.2f}s")
        print(f"Get dimension:             {dimension_time:.2f}s")
        print(f"VectorStore:               {vector_time:.2f}s")
        print(f"HybridSearchEngine:        {hybrid_time:.2f}s")
        print(f"Reranker + warmup:         {reranker_time:.2f}s")
        print(f"{'='*50}")
        print(f"TOTAL STARTUP TIME:        {total_time:.2f}s")
        
        # Identify the biggest bottlenecks
        timings = [
            ("Config", config_time),
            ("Metadata", metadata_time),
            ("Relationships", relationship_time),
            ("Embedder", embedder_time),
            ("Dimension", dimension_time),
            ("VectorStore", vector_time),
            ("HybridSearch", hybrid_time),
            ("Reranker", reranker_time)
        ]
        
        # Sort by time descending
        timings.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🐌 SLOWEST COMPONENTS:")
        for i, (name, timing) in enumerate(timings[:3]):
            if timing > 0.1:  # Only show significant times
                percentage = (timing / total_time) * 100
                print(f"{i+1}. {name}: {timing:.2f}s ({percentage:.1f}%)")

if __name__ == "__main__":
    main()