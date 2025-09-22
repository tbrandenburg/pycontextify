#!/usr/bin/env python3
"""
Knowledge Graph Construction and Analysis Module

This module provides comprehensive knowledge graph capabilities for PyContextify,
including graph construction, traversal, analysis, and visualization support.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .advanced_relationships import (
    Relationship,
    RelationshipType,
    SemanticRelationshipExtractor,
)
from .entity_extractor import Entity, extract_entities_from_text
from .metadata import ChunkMetadata
from .models import SearchResponse
from .relationship_store import RelationshipStore

logger = logging.getLogger(__name__)


class GraphTraversalMode(Enum):
    """Graph traversal modes for different analysis purposes."""

    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    WEIGHTED = "weighted"  # Prioritize high-confidence relationships
    SEMANTIC = "semantic"  # Follow semantic similarity paths


class GraphMetricType(Enum):
    """Types of graph metrics for analysis."""

    CENTRALITY = "centrality"
    CLUSTERING = "clustering"
    CONNECTIVITY = "connectivity"
    IMPORTANCE = "importance"


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""

    entity_id: str
    entity_name: str
    entity_type: str
    confidence: float
    chunk_ids: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Graph metrics (calculated dynamically)
    centrality_score: float = 0.0
    importance_score: float = 0.0
    cluster_id: Optional[str] = None

    def add_chunk_reference(self, chunk_id: str) -> None:
        """Add a chunk reference to this node."""
        self.chunk_ids.add(chunk_id)

    def get_occurrence_count(self) -> int:
        """Get the number of chunks this entity appears in."""
        return len(self.chunk_ids)

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "chunk_ids": list(self.chunk_ids),
            "attributes": self.attributes,
            "centrality_score": self.centrality_score,
            "importance_score": self.importance_score,
            "cluster_id": self.cluster_id,
            "occurrence_count": self.get_occurrence_count(),
        }


@dataclass
class GraphEdge:
    """Represents an edge (relationship) in the knowledge graph."""

    source_id: str
    target_id: str
    relationship_type: RelationshipType
    confidence: float
    chunk_ids: Set[str] = field(default_factory=set)
    context: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def add_context_reference(
        self, chunk_id: str, context: Optional[str] = None
    ) -> None:
        """Add contextual reference for this relationship."""
        self.chunk_ids.add(chunk_id)
        if context and not self.context:
            self.context = context

    def get_strength(self) -> float:
        """Calculate relationship strength based on confidence and frequency."""
        frequency_score = min(
            1.0, len(self.chunk_ids) / 5.0
        )  # Normalize to max 5 occurrences
        return (self.confidence * 0.7) + (frequency_score * 0.3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "confidence": self.confidence,
            "chunk_ids": list(self.chunk_ids),
            "context": self.context,
            "attributes": self.attributes,
            "strength": self.get_strength(),
        }


@dataclass
class GraphPath:
    """Represents a path through the knowledge graph."""

    nodes: List[str]
    edges: List[GraphEdge]
    total_confidence: float
    path_length: int
    semantic_coherence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert path to dictionary representation."""
        return {
            "nodes": self.nodes,
            "edges": [edge.to_dict() for edge in self.edges],
            "total_confidence": self.total_confidence,
            "path_length": self.path_length,
            "semantic_coherence": self.semantic_coherence,
        }


@dataclass
class KnowledgeCluster:
    """Represents a cluster of related entities in the knowledge graph."""

    cluster_id: str
    entities: Set[str]
    central_entity: str
    cluster_type: str  # e.g., "code_module", "topic_area", "technical_domain"
    coherence_score: float
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary representation."""
        return {
            "cluster_id": self.cluster_id,
            "entities": list(self.entities),
            "central_entity": self.central_entity,
            "cluster_type": self.cluster_type,
            "coherence_score": self.coherence_score,
            "description": self.description,
            "size": len(self.entities),
        }


class KnowledgeGraph:
    """Comprehensive knowledge graph for entities and relationships."""

    def __init__(self, relationship_store: RelationshipStore):
        """Initialize knowledge graph.

        Args:
            relationship_store: Existing relationship store
        """
        self.relationship_store = relationship_store
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}  # Keyed by f"{source_id}-{target_id}"
        self.adjacency_list: Dict[str, Set[str]] = defaultdict(set)

        # Analysis caches
        self._centrality_cache: Dict[str, float] = {}
        self._clusters: List[KnowledgeCluster] = []
        self._last_analysis_time: Optional[float] = None

        # Build graph from existing data
        self._build_from_relationship_store()

    def _build_from_relationship_store(self) -> None:
        """Build knowledge graph from existing relationship store data."""
        logger.info("Building knowledge graph from relationship store...")

        # Get all entities and relationships
        try:
            stats = self.relationship_store.get_stats()
            logger.info(
                f"Processing {stats['total_entities']} entities and {stats['total_relationships']} relationships"
            )

            # Add nodes for all entities
            # Note: This is a simplified approach - in a real implementation,
            # we'd need access to the actual entity data structure
            all_entities = (
                self.relationship_store._entities.keys()
                if hasattr(self.relationship_store, "_entities")
                else []
            )

            for entity_name in all_entities:
                entity_id = self._generate_entity_id(entity_name)
                related_chunks = self.relationship_store.get_related_chunks(entity_name)

                # Create node with basic information
                node = GraphNode(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    entity_type="unknown",  # Would need entity type information
                    confidence=0.8,  # Default confidence
                    chunk_ids=set(related_chunks[:10]),  # Limit chunk references
                )
                self.nodes[entity_id] = node

            # Add edges from relationship data
            # This would require access to the actual relationships in the store
            logger.info(f"Built knowledge graph with {len(self.nodes)} nodes")

        except Exception as e:
            logger.warning(f"Could not build from relationship store: {e}")
            # Continue with empty graph that can be populated later

    def add_entity(self, entity: Entity, chunk_id: str) -> str:
        """Add or update an entity in the knowledge graph.

        Args:
            entity: Entity to add
            chunk_id: ID of the chunk containing this entity

        Returns:
            Entity ID in the graph
        """
        entity_id = self._generate_entity_id(entity.name)

        if entity_id in self.nodes:
            # Update existing node
            node = self.nodes[entity_id]
            node.add_chunk_reference(chunk_id)
            # Update confidence with weighted average
            old_weight = node.get_occurrence_count() - 1
            total_weight = old_weight + 1
            node.confidence = (
                (node.confidence * old_weight) + entity.confidence
            ) / total_weight
        else:
            # Create new node
            node = GraphNode(
                entity_id=entity_id,
                entity_name=entity.name,
                entity_type=entity.entity_type,
                confidence=entity.confidence,
                chunk_ids={chunk_id},
                attributes={
                    "normalized_name": getattr(entity, "normalized_name", entity.name),
                    "aliases": getattr(entity, "aliases", set()),
                    "source_position": getattr(entity, "source_position", None),
                },
            )
            self.nodes[entity_id] = node

        # Clear analysis caches
        self._invalidate_caches()

        return entity_id

    def add_relationship(self, relationship: Relationship, chunk_id: str) -> None:
        """Add a relationship to the knowledge graph.

        Args:
            relationship: Relationship to add
            chunk_id: ID of the chunk where this relationship was found
        """
        source_id = self._generate_entity_id(relationship.source_entity)
        target_id = self._generate_entity_id(relationship.target_entity)
        edge_id = f"{source_id}-{target_id}"

        if edge_id in self.edges:
            # Update existing edge
            edge = self.edges[edge_id]
            edge.add_context_reference(chunk_id, relationship.context)
            # Update confidence with weighted average
            old_weight = len(edge.chunk_ids) - 1
            total_weight = old_weight + 1
            edge.confidence = (
                (edge.confidence * old_weight) + relationship.confidence
            ) / total_weight
        else:
            # Create new edge
            edge = GraphEdge(
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship.relationship_type,
                confidence=relationship.confidence,
                chunk_ids={chunk_id},
                context=relationship.context,
                attributes=getattr(relationship, "attributes", {}),
            )
            self.edges[edge_id] = edge

        # Update adjacency list
        self.adjacency_list[source_id].add(target_id)

        # Clear analysis caches
        self._invalidate_caches()

    def get_node_neighbors(
        self,
        entity_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
    ) -> List[Tuple[str, GraphEdge]]:
        """Get neighbors of a node with their connecting edges.

        Args:
            entity_id: ID of the entity to get neighbors for
            relationship_types: Optional filter for relationship types

        Returns:
            List of (neighbor_id, edge) tuples
        """
        neighbors = []

        for neighbor_id in self.adjacency_list.get(entity_id, set()):
            edge_id = f"{entity_id}-{neighbor_id}"
            edge = self.edges.get(edge_id)

            if edge and (
                not relationship_types or edge.relationship_type in relationship_types
            ):
                neighbors.append((neighbor_id, edge))

        # Sort by edge strength
        return sorted(neighbors, key=lambda x: x[1].get_strength(), reverse=True)

    def find_shortest_path(
        self, source_entity: str, target_entity: str, max_depth: int = 6
    ) -> Optional[GraphPath]:
        """Find shortest path between two entities.

        Args:
            source_entity: Source entity name
            target_entity: Target entity name
            max_depth: Maximum path length to search

        Returns:
            GraphPath if path exists, None otherwise
        """
        source_id = self._generate_entity_id(source_entity)
        target_id = self._generate_entity_id(target_entity)

        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        # BFS to find shortest path
        queue = deque([(source_id, [source_id], [])])
        visited = {source_id}

        while queue:
            current_id, path, edges = queue.popleft()

            if len(path) > max_depth:
                continue

            if current_id == target_id:
                # Found target, construct path object
                total_confidence = (
                    sum(edge.confidence for edge in edges) / len(edges)
                    if edges
                    else 1.0
                )
                return GraphPath(
                    nodes=path,
                    edges=edges,
                    total_confidence=total_confidence,
                    path_length=len(edges),
                )

            # Explore neighbors
            for neighbor_id, edge in self.get_node_neighbors(current_id):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path + [neighbor_id]
                    new_edges = edges + [edge]
                    queue.append((neighbor_id, new_path, new_edges))

        return None

    def get_entity_subgraph(
        self, entity_name: str, depth: int = 2, min_confidence: float = 0.3
    ) -> Dict[str, Any]:
        """Get subgraph centered around an entity.

        Args:
            entity_name: Name of the central entity
            depth: Maximum depth to explore
            min_confidence: Minimum confidence threshold for relationships

        Returns:
            Subgraph as dictionary with nodes and edges
        """
        entity_id = self._generate_entity_id(entity_name)
        if entity_id not in self.nodes:
            return {"nodes": [], "edges": []}

        # BFS to collect nodes within depth
        subgraph_nodes = {}
        subgraph_edges = []
        queue = deque([(entity_id, 0)])
        visited = {entity_id}

        while queue:
            current_id, current_depth = queue.popleft()
            subgraph_nodes[current_id] = self.nodes[current_id]

            if current_depth < depth:
                for neighbor_id, edge in self.get_node_neighbors(current_id):
                    if edge.confidence >= min_confidence:
                        subgraph_edges.append(edge)

                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            queue.append((neighbor_id, current_depth + 1))

        return {
            "nodes": [node.to_dict() for node in subgraph_nodes.values()],
            "edges": [edge.to_dict() for edge in subgraph_edges],
            "center_entity": entity_name,
            "depth": depth,
            "min_confidence": min_confidence,
        }

    def calculate_centrality_scores(self) -> Dict[str, float]:
        """Calculate centrality scores for all nodes.

        Returns:
            Dictionary mapping entity IDs to centrality scores
        """
        if self._centrality_cache:
            return self._centrality_cache

        centrality_scores = {}

        for entity_id in self.nodes:
            # Simple degree centrality (can be extended to other measures)
            degree = len(self.adjacency_list.get(entity_id, set()))

            # Weight by relationship strengths
            weighted_degree = 0.0
            for neighbor_id in self.adjacency_list.get(entity_id, set()):
                edge_id = f"{entity_id}-{neighbor_id}"
                edge = self.edges.get(edge_id)
                if edge:
                    weighted_degree += edge.get_strength()

            # Combine degree with node importance (occurrence frequency)
            node = self.nodes[entity_id]
            occurrence_score = min(1.0, node.get_occurrence_count() / 10.0)

            centrality_score = (weighted_degree * 0.6) + (occurrence_score * 0.4)
            centrality_scores[entity_id] = centrality_score

            # Update node
            node.centrality_score = centrality_score

        self._centrality_cache = centrality_scores
        return centrality_scores

    def discover_clusters(
        self, min_cluster_size: int = 3, coherence_threshold: float = 0.5
    ) -> List[KnowledgeCluster]:
        """Discover clusters of related entities.

        Args:
            min_cluster_size: Minimum entities per cluster
            coherence_threshold: Minimum coherence score for clusters

        Returns:
            List of discovered clusters
        """
        if self._clusters:
            return self._clusters

        # Simple clustering based on entity types and relationship strength
        entity_type_groups = defaultdict(set)

        # Group by entity type first
        for entity_id, node in self.nodes.items():
            entity_type_groups[node.entity_type].add(entity_id)

        clusters = []
        cluster_id = 0

        for entity_type, entities in entity_type_groups.items():
            if len(entities) >= min_cluster_size:
                # Find most central entity as cluster center
                centrality_scores = self.calculate_centrality_scores()
                central_entity_id = max(
                    entities, key=lambda e: centrality_scores.get(e, 0)
                )
                central_entity_name = self.nodes[central_entity_id].entity_name

                # Calculate cluster coherence
                total_connections = 0
                possible_connections = len(entities) * (len(entities) - 1) / 2

                for entity1 in entities:
                    for entity2 in entities:
                        if entity1 != entity2:
                            edge_id = f"{entity1}-{entity2}"
                            if edge_id in self.edges:
                                total_connections += 1

                coherence_score = (
                    total_connections / possible_connections
                    if possible_connections > 0
                    else 0.0
                )

                if coherence_score >= coherence_threshold:
                    cluster = KnowledgeCluster(
                        cluster_id=f"cluster_{cluster_id}",
                        entities=entities,
                        central_entity=central_entity_name,
                        cluster_type=f"{entity_type}_cluster",
                        coherence_score=coherence_score,
                        description=f"Cluster of {entity_type} entities centered around {central_entity_name}",
                    )
                    clusters.append(cluster)
                    cluster_id += 1

                    # Update nodes with cluster assignment
                    for entity_id in entities:
                        self.nodes[entity_id].cluster_id = cluster.cluster_id

        self._clusters = clusters
        return clusters

    def get_recommendations(
        self, entity_name: str, top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """Get entity recommendations based on graph structure.

        Args:
            entity_name: Entity to base recommendations on
            top_k: Number of recommendations to return

        Returns:
            List of (entity_name, score, reason) tuples
        """
        entity_id = self._generate_entity_id(entity_name)
        if entity_id not in self.nodes:
            return []

        recommendations = {}

        # Direct neighbors with high relationship strength
        for neighbor_id, edge in self.get_node_neighbors(entity_id):
            neighbor_name = self.nodes[neighbor_id].entity_name
            score = edge.get_strength()
            reason = f"Direct relationship ({edge.relationship_type.value})"
            recommendations[neighbor_name] = (score, reason)

        # Second-degree connections (neighbors of neighbors)
        for neighbor_id, _ in self.get_node_neighbors(entity_id):
            for second_neighbor_id, second_edge in self.get_node_neighbors(neighbor_id):
                if second_neighbor_id != entity_id and second_neighbor_id not in [
                    n[0] for n, _ in self.get_node_neighbors(entity_id)
                ]:
                    second_neighbor_name = self.nodes[second_neighbor_id].entity_name
                    score = (
                        second_edge.get_strength() * 0.7
                    )  # Reduce score for indirect connection
                    reason = f"Connected via {self.nodes[neighbor_id].entity_name}"

                    if (
                        second_neighbor_name not in recommendations
                        or recommendations[second_neighbor_name][0] < score
                    ):
                        recommendations[second_neighbor_name] = (score, reason)

        # Cluster-based recommendations
        node = self.nodes[entity_id]
        if node.cluster_id:
            cluster = next(
                (c for c in self._clusters if c.cluster_id == node.cluster_id), None
            )
            if cluster:
                for cluster_entity_id in cluster.entities:
                    if cluster_entity_id != entity_id:
                        cluster_entity_name = self.nodes[cluster_entity_id].entity_name
                        score = 0.5  # Base cluster similarity score
                        reason = f"Same cluster ({cluster.cluster_type})"

                        if (
                            cluster_entity_name not in recommendations
                            or recommendations[cluster_entity_name][0] < score
                        ):
                            recommendations[cluster_entity_name] = (score, reason)

        # Sort by score and return top_k
        sorted_recommendations = sorted(
            [
                (name, score, reason)
                for name, (score, reason) in recommendations.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_recommendations[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics.

        Returns:
            Dictionary with graph statistics
        """
        if not self._centrality_cache:
            self.calculate_centrality_scores()

        # Calculate basic graph metrics
        total_nodes = len(self.nodes)
        total_edges = len(self.edges)

        # Node type distribution
        node_types = defaultdict(int)
        confidence_scores = []

        for node in self.nodes.values():
            node_types[node.entity_type] += 1
            confidence_scores.append(node.confidence)

        # Edge type distribution
        edge_types = defaultdict(int)
        relationship_strengths = []

        for edge in self.edges.values():
            edge_types[edge.relationship_type.value] += 1
            relationship_strengths.append(edge.get_strength())

        # Centrality statistics
        centrality_values = list(self._centrality_cache.values())

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "density": (
                (2 * total_edges) / (total_nodes * (total_nodes - 1))
                if total_nodes > 1
                else 0.0
            ),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "clusters": len(self._clusters),
            "statistics": {
                "avg_node_confidence": (
                    sum(confidence_scores) / len(confidence_scores)
                    if confidence_scores
                    else 0.0
                ),
                "avg_relationship_strength": (
                    sum(relationship_strengths) / len(relationship_strengths)
                    if relationship_strengths
                    else 0.0
                ),
                "avg_centrality": (
                    sum(centrality_values) / len(centrality_values)
                    if centrality_values
                    else 0.0
                ),
                "max_centrality": max(centrality_values) if centrality_values else 0.0,
            },
            "last_analysis_time": self._last_analysis_time,
        }

    def _generate_entity_id(self, entity_name: str) -> str:
        """Generate consistent entity ID from name.

        Args:
            entity_name: Name of the entity

        Returns:
            Consistent entity ID
        """
        # Simple ID generation - could be made more sophisticated
        return f"entity_{hash(entity_name.lower()) % 1000000}"

    def _invalidate_caches(self) -> None:
        """Invalidate analysis caches when graph changes."""
        self._centrality_cache = {}
        self._clusters = []
        self._last_analysis_time = None

    def export_to_dict(self) -> Dict[str, Any]:
        """Export entire knowledge graph to dictionary format.

        Returns:
            Complete graph representation as dictionary
        """
        return {
            "metadata": {"export_time": time.time(), "stats": self.get_stats()},
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "clusters": [cluster.to_dict() for cluster in self._clusters],
        }

    def import_from_dict(self, graph_data: Dict[str, Any]) -> None:
        """Import knowledge graph from dictionary format.

        Args:
            graph_data: Graph data dictionary
        """
        # Clear current graph
        self.nodes = {}
        self.edges = {}
        self.adjacency_list = defaultdict(set)
        self._invalidate_caches()

        # Import nodes
        for node_data in graph_data.get("nodes", []):
            node = GraphNode(
                entity_id=node_data["entity_id"],
                entity_name=node_data["entity_name"],
                entity_type=node_data["entity_type"],
                confidence=node_data["confidence"],
                chunk_ids=set(node_data.get("chunk_ids", [])),
                attributes=node_data.get("attributes", {}),
                centrality_score=node_data.get("centrality_score", 0.0),
                importance_score=node_data.get("importance_score", 0.0),
                cluster_id=node_data.get("cluster_id"),
            )
            self.nodes[node.entity_id] = node

        # Import edges
        for edge_data in graph_data.get("edges", []):
            relationship_type = RelationshipType(edge_data["relationship_type"])
            edge = GraphEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                relationship_type=relationship_type,
                confidence=edge_data["confidence"],
                chunk_ids=set(edge_data.get("chunk_ids", [])),
                context=edge_data.get("context"),
                attributes=edge_data.get("attributes", {}),
            )

            edge_id = f"{edge.source_id}-{edge.target_id}"
            self.edges[edge_id] = edge
            self.adjacency_list[edge.source_id].add(edge.target_id)

        # Import clusters
        self._clusters = []
        for cluster_data in graph_data.get("clusters", []):
            cluster = KnowledgeCluster(
                cluster_id=cluster_data["cluster_id"],
                entities=set(cluster_data["entities"]),
                central_entity=cluster_data["central_entity"],
                cluster_type=cluster_data["cluster_type"],
                coherence_score=cluster_data["coherence_score"],
                description=cluster_data.get("description"),
            )
            self._clusters.append(cluster)

        logger.info(
            f"Imported knowledge graph with {len(self.nodes)} nodes and {len(self.edges)} edges"
        )


def build_knowledge_graph_from_chunks(
    chunks: List[ChunkMetadata],
    relationship_store: RelationshipStore,
    entity_extraction_enabled: bool = True,
) -> KnowledgeGraph:
    """Build a comprehensive knowledge graph from chunks.

    Args:
        chunks: List of chunk metadata
        relationship_store: Relationship store instance
        entity_extraction_enabled: Whether to extract entities from chunks

    Returns:
        Constructed KnowledgeGraph
    """
    logger.info(f"Building knowledge graph from {len(chunks)} chunks")

    # Initialize knowledge graph
    kg = KnowledgeGraph(relationship_store)

    if entity_extraction_enabled:
        # Extract entities and relationships from all chunks
        relationship_extractor = SemanticRelationshipExtractor()

        for chunk in chunks:
            try:
                # Extract entities from chunk
                source_type = (
                    chunk.source_type.value
                    if hasattr(chunk.source_type, "value")
                    else str(chunk.source_type)
                )
                extraction_result = extract_entities_from_text(
                    chunk.chunk_text, source_type
                )
                entities = extraction_result.entities

                # Add entities to graph
                for entity in entities:
                    kg.add_entity(entity, chunk.chunk_id)

                # Extract relationships
                relationships = relationship_extractor.extract_relationships(
                    chunk, entities
                )

                # Add relationships to graph
                for relationship in relationships:
                    kg.add_relationship(relationship, chunk.chunk_id)

            except Exception as e:
                logger.warning(f"Failed to process chunk {chunk.chunk_id}: {e}")
                continue

    # Perform analysis
    logger.info("Analyzing knowledge graph...")
    kg.calculate_centrality_scores()
    kg.discover_clusters()

    stats = kg.get_stats()
    logger.info(
        f"Knowledge graph built: {stats['total_nodes']} nodes, {stats['total_edges']} edges, {stats['clusters']} clusters"
    )

    return kg
