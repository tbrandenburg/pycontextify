"""Working tests for RelationshipStore functionality."""

import pytest
import tempfile
import json
from pathlib import Path

from pycontextify.index.relationship_store import RelationshipStore
from pycontextify.index.config import Config


class TestRelationshipStoreSimple:
    """Test RelationshipStore functionality with correct API."""

    def test_relationship_store_initialization(self):
        """Test RelationshipStore initialization."""
        store = RelationshipStore()
        
        assert store._relationships == {}
        assert store._reverse_index == {}
        assert store._entity_chunks == {}

    def test_add_relationship_basic(self):
        """Test adding basic relationships."""
        store = RelationshipStore()
        
        # Add relationship
        store.add_relationship("entity_1", "chunk_1", "reference")
        
        assert "entity_1" in store._relationships
        assert "reference" in store._relationships["entity_1"]
        assert "chunk_1" in store._relationships["entity_1"]["reference"]
        assert "entity_1" in store._reverse_index["chunk_1"]
        assert "chunk_1" in store._entity_chunks["entity_1"]

    def test_add_multiple_relationships_same_entity(self):
        """Test adding multiple relationships for the same entity."""
        store = RelationshipStore()
        
        # Add multiple relationships
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_1", "chunk_2", "reference") 
        store.add_relationship("entity_1", "chunk_3", "function_call")
        
        assert len(store._relationships["entity_1"]["reference"]) == 2
        assert len(store._relationships["entity_1"]["function_call"]) == 1
        assert len(store._entity_chunks["entity_1"]) == 3

    def test_add_duplicate_relationship(self):
        """Test adding duplicate relationships (should not create duplicates)."""
        store = RelationshipStore()
        
        # Add same relationship multiple times
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_1", "chunk_1", "reference")
        
        # Should only have one relationship
        assert len(store._relationships["entity_1"]["reference"]) == 1
        assert len(store._reverse_index["chunk_1"]) == 1

    def test_get_related_chunks(self):
        """Test getting chunks related to an entity."""
        store = RelationshipStore()
        
        # Add relationships
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_1", "chunk_2", "function_call")
        
        # Get related chunks
        chunks = store.get_related_chunks("entity_1")
        
        assert len(chunks) == 2
        assert "chunk_1" in chunks
        assert "chunk_2" in chunks

    def test_get_related_chunks_specific_type(self):
        """Test getting chunks for specific relationship type."""
        store = RelationshipStore()
        
        # Add relationships
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_1", "chunk_2", "function_call")
        
        # Get chunks for specific type
        ref_chunks = store.get_related_chunks("entity_1", "reference")
        func_chunks = store.get_related_chunks("entity_1", "function_call")
        
        assert len(ref_chunks) == 1
        assert "chunk_1" in ref_chunks
        assert len(func_chunks) == 1
        assert "chunk_2" in func_chunks

    def test_get_related_chunks_nonexistent_entity(self):
        """Test getting chunks for non-existent entity."""
        store = RelationshipStore()
        
        # Get chunks for non-existent entity
        chunks = store.get_related_chunks("nonexistent_entity")
        
        assert chunks == []

    def test_get_chunk_entities(self):
        """Test getting entities related to a chunk."""
        store = RelationshipStore()
        
        # Add relationships
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_2", "chunk_1", "function_call")
        
        # Get related entities
        entities = store.get_chunk_entities("chunk_1")
        
        assert len(entities) == 2
        assert "reference" in entities
        assert "function_call" in entities
        assert "entity_1" in entities["reference"]
        assert "entity_2" in entities["function_call"]

    def test_get_chunk_entities_nonexistent_chunk(self):
        """Test getting entities for non-existent chunk."""
        store = RelationshipStore()
        
        # Get entities for non-existent chunk
        entities = store.get_chunk_entities("nonexistent_chunk")
        
        assert entities == {}

    def test_get_entity_graph_simple(self):
        """Test getting simple entity graph."""
        store = RelationshipStore()
        
        # Add relationships creating connections
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_2", "chunk_1", "reference")
        store.add_relationship("entity_2", "chunk_2", "reference")
        store.add_relationship("entity_3", "chunk_2", "reference")
        
        # Get graph starting from entity_1
        graph = store.get_entity_graph("entity_1", max_depth=2)
        
        assert "center" in graph
        assert graph["center"] == "entity_1"
        assert "nodes" in graph
        assert "entity_1" in graph["nodes"]

    def test_get_entity_graph_max_depth(self):
        """Test entity graph with limited depth."""
        store = RelationshipStore()
        
        # Create a chain of relationships
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_2", "chunk_1", "reference") 
        store.add_relationship("entity_2", "chunk_2", "reference")
        store.add_relationship("entity_3", "chunk_2", "reference")
        
        # Get graph with depth 1
        graph = store.get_entity_graph("entity_1", max_depth=1)
        
        # Should include connected entities within max depth
        assert graph["center"] == "entity_1"
        assert len(graph["nodes"]) >= 1

    def test_get_entity_graph_nonexistent_start(self):
        """Test entity graph for non-existent starting entity."""
        store = RelationshipStore()
        
        # Try to get graph for non-existent entity
        graph = store.get_entity_graph("nonexistent_entity")
        
        assert "center" in graph
        assert graph["center"] == "nonexistent_entity"

    def test_get_all_entities(self):
        """Test getting all entities."""
        store = RelationshipStore()
        
        # Add various relationships
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_2", "chunk_2", "function_call")
        store.add_relationship("entity_3", "chunk_3", "import")
        
        entities = store.get_all_entities()
        
        assert len(entities) == 3
        assert "entity_1" in entities
        assert "entity_2" in entities
        assert "entity_3" in entities

    def test_get_relationship_types(self):
        """Test getting all relationship types."""
        store = RelationshipStore()
        
        # Add relationships with different types
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_2", "chunk_2", "function_call")
        store.add_relationship("entity_3", "chunk_3", "import")
        
        types = store.get_relationship_types()
        
        assert len(types) == 3
        assert "reference" in types
        assert "function_call" in types
        assert "import" in types

    def test_find_related_entities(self):
        """Test finding related entities."""
        store = RelationshipStore()
        
        # Create connections
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_2", "chunk_1", "reference")
        store.add_relationship("entity_2", "chunk_2", "reference") 
        store.add_relationship("entity_3", "chunk_2", "reference")
        
        # Find related entities
        related = store.find_related_entities("entity_1", max_depth=1)
        
        assert "entity_2" in related
        # entity_3 might be there depending on depth traversal

    def test_clear_relationships(self):
        """Test clearing all relationships."""
        store = RelationshipStore()
        
        # Add various relationships
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_2", "chunk_2", "function_call")
        
        # Clear all
        store.clear()
        
        assert len(store._relationships) == 0
        assert len(store._reverse_index) == 0
        assert len(store._entity_chunks) == 0

    def test_get_stats(self):
        """Test getting relationship statistics."""
        store = RelationshipStore()
        
        # Add various relationships
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_1", "chunk_2", "reference")
        store.add_relationship("entity_2", "chunk_1", "function_call")
        
        stats = store.get_stats()
        
        assert stats["total_entities"] == 2
        assert stats["total_relationships"] == 3
        assert "relationship_types" in stats
        assert "chunks_with_relationships" in stats

    def test_calculate_relationship_strength(self):
        """Test calculating relationship strength between entities."""
        store = RelationshipStore()
        
        # Add relationships with shared chunks
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_1", "chunk_2", "reference")
        store.add_relationship("entity_2", "chunk_1", "reference")
        
        strength = store.calculate_relationship_strength("entity_1", "entity_2")
        
        # Should be > 0 since they share chunk_1
        assert strength > 0.0
        assert strength <= 1.0

    def test_save_to_file(self):
        """Test saving relationships to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RelationshipStore()
            
            # Add relationships
            store.add_relationship("entity_1", "chunk_1", "reference")
            store.add_relationship("entity_2", "chunk_2", "function_call")
            
            # Save to file
            filepath = Path(temp_dir) / "relationships.pkl"
            store.save_to_file(str(filepath), compress=False)
            
            # Verify file exists
            assert filepath.exists()

    def test_load_from_file(self):
        """Test loading relationships from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create store with data
            store1 = RelationshipStore()
            store1.add_relationship("entity_1", "chunk_1", "reference")
            store1.add_relationship("entity_2", "chunk_2", "function_call")
            
            # Save to file
            filepath = Path(temp_dir) / "relationships.pkl"
            store1.save_to_file(str(filepath), compress=False)
            
            # Load into new store
            store2 = RelationshipStore()
            store2.load_from_file(str(filepath))
            
            # Verify data loaded correctly
            assert len(store2._relationships) == 2
            assert "entity_1" in store2._relationships
            assert "entity_2" in store2._relationships

    def test_edge_cases_empty_store(self):
        """Test edge cases with empty store."""
        store = RelationshipStore()
        
        # Test operations on empty store
        assert store.get_related_chunks("entity") == []
        assert store.get_chunk_entities("chunk") == {}
        assert store.get_all_entities() == []
        assert store.get_relationship_types() == []
        
        stats = store.get_stats()
        assert stats["total_entities"] == 0
        assert stats["total_relationships"] == 0

    def test_relationship_constants(self):
        """Test relationship type constants."""
        store = RelationshipStore()
        
        # Test that constants are available
        assert hasattr(store, 'FUNCTION_CALL')
        assert hasattr(store, 'IMPORT')
        assert hasattr(store, 'REFERENCE')
        assert hasattr(store, 'LINK')
        assert hasattr(store, 'HIERARCHY')
        assert hasattr(store, 'TAG')
        
        # Use constants in relationships
        store.add_relationship("func1", "chunk1", store.FUNCTION_CALL)
        store.add_relationship("mod1", "chunk2", store.IMPORT)
        
        assert store.get_related_chunks("func1") == ["chunk1"]
        assert store.get_related_chunks("mod1") == ["chunk2"]

    def test_duplicate_relationship_prevention(self):
        """Test that duplicate relationships are prevented."""
        store = RelationshipStore()
        
        # Add same relationship multiple times
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_1", "chunk_1", "reference")  # duplicate
        store.add_relationship("entity_1", "chunk_1", "reference")  # duplicate
        
        # Should only have one relationship
        chunks = store.get_related_chunks("entity_1", "reference")
        assert len(chunks) == 1
        assert chunks[0] == "chunk_1"
        
        # Reverse lookup should also show only one
        entities = store.get_chunk_entities("chunk_1")
        assert len(entities["reference"]) == 1
        assert entities["reference"][0] == "entity_1"

    def test_multiple_relationship_types_same_pair(self):
        """Test multiple relationship types between same entity-chunk pair."""
        store = RelationshipStore()
        
        # Add different relationship types for same entity-chunk pair
        store.add_relationship("entity_1", "chunk_1", "reference")
        store.add_relationship("entity_1", "chunk_1", "function_call")
        store.add_relationship("entity_1", "chunk_1", "import")
        
        # Should have all relationship types
        all_chunks = store.get_related_chunks("entity_1")
        assert "chunk_1" in all_chunks
        
        ref_chunks = store.get_related_chunks("entity_1", "reference")
        func_chunks = store.get_related_chunks("entity_1", "function_call")
        import_chunks = store.get_related_chunks("entity_1", "import")
        
        assert len(ref_chunks) == 1
        assert len(func_chunks) == 1
        assert len(import_chunks) == 1
        
        # Reverse lookup should show all types
        entities = store.get_chunk_entities("chunk_1")
        assert len(entities) == 3
        assert "reference" in entities
        assert "function_call" in entities
        assert "import" in entities

    def test_save_load_roundtrip_complete(self):
        """Test complete save/load roundtrip preserves all data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create store with complex relationships
            store1 = RelationshipStore()
            store1.add_relationship("entity_1", "chunk_1", "reference")
            store1.add_relationship("entity_1", "chunk_2", "function_call")
            store1.add_relationship("entity_2", "chunk_1", "import")
            store1.add_relationship("entity_3", "chunk_3", "reference")
            
            filepath = Path(temp_dir) / "test_relationships.pkl"
            
            # Save and reload
            store1.save_to_file(str(filepath), compress=False)
            store2 = RelationshipStore()
            store2.load_from_file(str(filepath))
            
            # Verify complete data preservation
            assert set(store1.get_all_entities()) == set(store2.get_all_entities())
            assert set(store1.get_relationship_types()) == set(store2.get_relationship_types())
            
            for entity in store1.get_all_entities():
                chunks1 = store1.get_related_chunks(entity)
                chunks2 = store2.get_related_chunks(entity)
                assert set(chunks1) == set(chunks2)

    def test_load_from_nonexistent_file_graceful(self):
        """Test loading from non-existent file handles gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = RelationshipStore()
            nonexistent_path = Path(temp_dir) / "nonexistent.pkl"
            
            # Should not raise error
            store.load_from_file(str(nonexistent_path))
            
            # Should maintain empty state
            assert store.get_all_entities() == []
            assert store.get_relationship_types() == []

    def test_entity_graph_complex(self):
        """Test entity graph generation with complex relationships."""
        store = RelationshipStore()
        
        # Create a more complex relationship network
        store.add_relationship("A", "chunk1", "reference")
        store.add_relationship("B", "chunk1", "reference")
        store.add_relationship("B", "chunk2", "function_call")
        store.add_relationship("C", "chunk2", "reference")
        store.add_relationship("C", "chunk3", "import")
        store.add_relationship("D", "chunk3", "reference")
        
        # Get graph starting from A
        graph = store.get_entity_graph("A", max_depth=3)
        
        assert "center" in graph
        assert graph["center"] == "A"
        assert "nodes" in graph
        
        # Should include entities reachable within max_depth
        assert "A" in graph["nodes"]
        # B should be reachable via chunk1
        if len(graph["nodes"]) > 1:
            # Check basic graph connectivity
            assert isinstance(graph["nodes"], dict)
