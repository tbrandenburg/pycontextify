#!/usr/bin/env python3
"""
Basic functionality tests for advanced relationship and context features.

Tests entity extraction, relationship modeling, and knowledge graph components
to ensure they can be imported and provide basic functionality coverage.
"""

import pytest

from pycontextify.index.advanced_relationships import (
    Relationship,
    RelationshipType,
    SemanticRelationshipExtractor,
)
from pycontextify.index.entity_extractor import (
    CompositeEntityExtractor,
    Entity,
    EntityExtractionResult,
    EntityExtractor,
    RegexEntityExtractor,
    extract_entities_from_text,
)


class TestAdvancedRelationshipsBasic:
    """Basic import and functionality tests for advanced relationship features."""

    @pytest.mark.no_mock_st
    def test_entity_extractor_imports(self):
        """Test that entity extractor modules can be imported."""
        # Test enum values
        extractor = RegexEntityExtractor()
        assert extractor is not None

        # Test basic extraction
        result = extract_entities_from_text("def test(): pass", "code")
        assert isinstance(result, EntityExtractionResult)
        assert hasattr(result, "entities")
        assert hasattr(result, "extraction_stats")
        assert hasattr(result, "processing_time_ms")

    @pytest.mark.no_mock_st
    def test_entity_extractor_basic_functionality(self):
        """Test basic entity extraction functionality."""
        extractor = RegexEntityExtractor()

        # Test with simple code
        entities = extractor.extract_entities("class Test: pass", "code")
        assert isinstance(entities, list)

        # Test with empty string
        entities = extractor.extract_entities("", "code")
        assert isinstance(entities, list)

        # Test supported types
        supported_types = extractor.get_supported_entity_types()
        assert isinstance(supported_types, list)
        assert len(supported_types) > 0

    @pytest.mark.no_mock_st
    def test_composite_extractor(self):
        """Test composite entity extractor."""
        extractor = CompositeEntityExtractor()

        result = extractor.extract_entities("def hello(): return 'world'", "code")

        # CompositeEntityExtractor.extract_entities returns List[Entity], not EntityExtractionResult
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], Entity)

    @pytest.mark.no_mock_st
    def test_relationship_types(self):
        """Test relationship type enumeration."""
        # Test that all relationship types are accessible
        assert RelationshipType.DEFINES.value == "defines"
        assert RelationshipType.USES.value == "uses"
        assert RelationshipType.INHERITS.value == "inherits"
        assert RelationshipType.CONTAINS.value == "contains"

        # Test creating a relationship
        rel = Relationship(
            source_entity="ClassA",
            target_entity="ClassB",
            relationship_type=RelationshipType.INHERITS,
            confidence=0.8,
        )

        assert rel.source_entity == "ClassA"
        assert rel.target_entity == "ClassB"
        assert rel.relationship_type == RelationshipType.INHERITS
        assert rel.confidence == 0.8

    @pytest.mark.no_mock_st
    def test_semantic_relationship_extractor_init(self):
        """Test semantic relationship extractor initialization."""
        extractor = SemanticRelationshipExtractor()
        assert extractor is not None

        # Test that it has the expected attributes
        assert hasattr(extractor, "_semantic_patterns")
        assert hasattr(extractor, "_compiled_patterns")
        assert hasattr(extractor, "_cooccurrence_window")

        # Verify some patterns were compiled
        assert len(extractor._compiled_patterns) > 0

    @pytest.mark.no_mock_st
    def test_relationship_confidence_validation(self):
        """Test that relationship confidence is validated."""
        # Valid confidence
        rel = Relationship(
            source_entity="A",
            target_entity="B",
            relationship_type=RelationshipType.USES,
            confidence=0.7,
        )
        assert rel.confidence == 0.7

        # Invalid confidence should raise error
        with pytest.raises(ValueError):
            Relationship(
                source_entity="A",
                target_entity="B",
                relationship_type=RelationshipType.USES,
                confidence=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValueError):
            Relationship(
                source_entity="A",
                target_entity="B",
                relationship_type=RelationshipType.USES,
                confidence=-0.1,  # Invalid: < 0.0
            )


@pytest.mark.no_mock_st  # Disable sentence transformer mocking for these basic tests
class TestKnowledgeGraphBasic:
    """Test knowledge graph module can be imported and initialized."""

    def test_knowledge_graph_import(self):
        """Test that knowledge graph module can be imported."""
        from pycontextify.index.knowledge_graph import (
            GraphEdge,
            GraphMetricType,
            GraphNode,
            GraphTraversalMode,
            KnowledgeGraph,
        )

        # Test enum values
        assert GraphTraversalMode.BREADTH_FIRST.value == "breadth_first"
        assert GraphMetricType.CENTRALITY.value == "centrality"

        # Test that classes can be instantiated (with mocked relationship store)
        from unittest.mock import Mock

        mock_store = Mock()
        mock_store.get_stats.return_value = {
            "total_entities": 0,
            "total_relationships": 0,
        }

        kg = KnowledgeGraph(mock_store)
        assert kg is not None
        assert hasattr(kg, "nodes")
        assert hasattr(kg, "edges")

    def test_graph_node_creation(self):
        """Test GraphNode creation and basic methods."""
        from pycontextify.index.knowledge_graph import GraphNode

        node = GraphNode(
            entity_id="test_1",
            entity_name="TestEntity",
            entity_type="class",
            confidence=0.9,
        )

        assert node.entity_id == "test_1"
        assert node.entity_name == "TestEntity"
        assert node.entity_type == "class"
        assert node.confidence == 0.9

        # Test methods
        node.add_chunk_reference("chunk_1")
        assert "chunk_1" in node.chunk_ids
        assert node.get_occurrence_count() == 1

        # Test dictionary conversion
        node_dict = node.to_dict()
        assert isinstance(node_dict, dict)
        assert node_dict["entity_name"] == "TestEntity"

    def test_graph_edge_creation(self):
        """Test GraphEdge creation and basic methods."""
        from pycontextify.index.knowledge_graph import GraphEdge

        edge = GraphEdge(
            source_id="entity_1",
            target_id="entity_2",
            relationship_type=RelationshipType.USES,
            confidence=0.8,
        )

        assert edge.source_id == "entity_1"
        assert edge.target_id == "entity_2"
        assert edge.relationship_type == RelationshipType.USES
        assert edge.confidence == 0.8

        # Test methods
        edge.add_context_reference("chunk_1", "some context")
        assert "chunk_1" in edge.chunk_ids
        assert edge.context == "some context"

        # Test strength calculation
        strength = edge.get_strength()
        assert isinstance(strength, float)
        assert 0.0 <= strength <= 1.0

        # Test dictionary conversion
        edge_dict = edge.to_dict()
        assert isinstance(edge_dict, dict)
        assert edge_dict["source_id"] == "entity_1"
