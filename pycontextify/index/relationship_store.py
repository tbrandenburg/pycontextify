"""Lightweight relationship store for PyContextify.

This module implements a simple relationship management system for
lightweight knowledge graph capabilities without the complexity
of full graph databases.
"""

import gzip
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .metadata import ChunkMetadata, SourceType


class RelationshipStore:
    """Lightweight relationship store for managing entity relationships.

    This class provides simple relationship management capabilities
    for building lightweight knowledge graphs from indexed content.
    """

    # Common relationship types
    FUNCTION_CALL = "function_call"
    IMPORT = "import"
    REFERENCE = "reference"
    LINK = "link"
    HIERARCHY = "hierarchy"
    TAG = "tag"

    def __init__(self) -> None:
        """Initialize empty relationship store."""
        # Forward index: entity -> {relationship_type -> [chunk_ids]}
        self._relationships: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Reverse index: chunk_id -> [entities]
        self._reverse_index: Dict[str, List[str]] = defaultdict(list)
        # Simple entity -> chunk mapping for quick lookup
        self._entity_chunks: Dict[str, Set[str]] = defaultdict(set)

    def add_relationship(
        self, entity: str, chunk_id: str, relationship_type: str
    ) -> None:
        """Add a relationship between entity and chunk.

        Args:
            entity: The entity name (function, class, concept, etc.)
            chunk_id: The chunk ID containing this entity
            relationship_type: Type of relationship (function_call, import, etc.)
        """
        if chunk_id not in self._relationships[entity][relationship_type]:
            self._relationships[entity][relationship_type].append(chunk_id)

        if entity not in self._reverse_index[chunk_id]:
            self._reverse_index[chunk_id].append(entity)

        self._entity_chunks[entity].add(chunk_id)

    def get_related_chunks(
        self, entity: str, relationship_type: str = "all"
    ) -> List[str]:
        """Get chunks related to an entity.

        Args:
            entity: The entity to search for
            relationship_type: Specific relationship type or "all" for all types

        Returns:
            List of chunk IDs related to the entity
        """
        if entity not in self._relationships:
            return []

        if relationship_type == "all":
            # Return all chunks for this entity across all relationship types
            chunks = []
            for rel_type, chunk_list in self._relationships[entity].items():
                chunks.extend(chunk_list)
            return list(set(chunks))  # Remove duplicates
        else:
            return self._relationships[entity][relationship_type]

    def get_chunk_entities(self, chunk_id: str) -> Dict[str, List[str]]:
        """Get all entities related to a chunk grouped by relationship type.

        Args:
            chunk_id: The chunk ID

        Returns:
            Dictionary mapping relationship types to entity lists
        """
        result = defaultdict(list)

        for entity in self._reverse_index[chunk_id]:
            for rel_type, chunk_list in self._relationships[entity].items():
                if chunk_id in chunk_list:
                    result[rel_type].append(entity)

        return dict(result)

    def get_relationship_types(self) -> List[str]:
        """Get all available relationship types."""
        types = set()
        for entity_rels in self._relationships.values():
            types.update(entity_rels.keys())
        return list(types)

    def get_all_entities(self) -> List[str]:
        """Get all entities in the store."""
        return list(self._relationships.keys())

    def find_related_entities(self, entity: str, max_depth: int = 1) -> Set[str]:
        """Find entities related through relationships.

        Args:
            entity: Starting entity
            max_depth: Maximum relationship depth to traverse

        Returns:
            Set of related entities
        """
        if max_depth <= 0:
            return set()

        related = set()
        current_level = {entity}

        for depth in range(max_depth):
            next_level = set()

            for current_entity in current_level:
                # Get all chunks related to current entity
                related_chunks = self.get_related_chunks(current_entity)

                # Get all entities from those chunks
                for chunk_id in related_chunks:
                    chunk_entities = self._reverse_index[chunk_id]
                    for ent in chunk_entities:
                        if ent != entity and ent not in related:
                            next_level.add(ent)
                            related.add(ent)

            current_level = next_level
            if not current_level:
                break

        return related

    def get_entity_graph(self, entity: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get a subgraph around an entity.

        Args:
            entity: Center entity
            max_depth: Maximum depth to explore

        Returns:
            Dictionary representing the entity graph
        """
        graph = {"center": entity, "nodes": {}, "edges": [], "depth": max_depth}

        # Start with the center entity
        entities_to_process = {entity: 0}
        processed = set()

        while entities_to_process:
            current_entity, depth = min(entities_to_process.items(), key=lambda x: x[1])
            del entities_to_process[current_entity]

            if current_entity in processed or depth > max_depth:
                continue

            processed.add(current_entity)

            # Add node information
            related_chunks = self.get_related_chunks(current_entity)
            graph["nodes"][current_entity] = {
                "depth": depth,
                "chunk_count": len(related_chunks),
                "relationships": {},
            }

            # Get relationship types for this entity
            for rel_type, chunk_list in self._relationships[current_entity].items():
                graph["nodes"][current_entity]["relationships"][rel_type] = len(
                    chunk_list
                )

            # Find connected entities
            if depth < max_depth:
                for chunk_id in related_chunks:
                    for connected_entity in self._reverse_index[chunk_id]:
                        if (
                            connected_entity != current_entity
                            and connected_entity not in processed
                        ):
                            # Add edge
                            graph["edges"].append(
                                {
                                    "from": current_entity,
                                    "to": connected_entity,
                                    "shared_chunks": [chunk_id],
                                    "depth": depth + 1,
                                }
                            )

                            # Add to processing queue
                            if connected_entity not in entities_to_process:
                                entities_to_process[connected_entity] = depth + 1

        return graph

    def calculate_relationship_strength(self, entity1: str, entity2: str) -> float:
        """Calculate relationship strength between two entities.

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            Relationship strength (0.0 to 1.0)
        """
        chunks1 = set(self.get_related_chunks(entity1))
        chunks2 = set(self.get_related_chunks(entity2))

        if not chunks1 or not chunks2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(chunks1 & chunks2)
        union = len(chunks1 | chunks2)

        return intersection / union if union > 0 else 0.0

    def extract_code_relationships(
        self, chunk: ChunkMetadata
    ) -> List[Tuple[str, str, str]]:
        """Extract relationships from code chunks.

        Args:
            chunk: Code chunk metadata

        Returns:
            List of (entity, chunk_id, relationship_type) tuples
        """
        relationships = []

        if chunk.source_type != SourceType.CODE:
            return relationships

        text = chunk.chunk_text
        chunk_id = chunk.chunk_id

        # Extract function definitions
        func_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        for match in re.finditer(func_pattern, text):
            func_name = match.group(1)
            relationships.append((func_name, chunk_id, self.FUNCTION_CALL))

        # Extract class definitions
        class_pattern = r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]"
        for match in re.finditer(class_pattern, text):
            class_name = match.group(1)
            relationships.append((class_name, chunk_id, self.HIERARCHY))

        # Extract import statements
        import_patterns = [
            r"import\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
            r"from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import",
        ]

        for pattern in import_patterns:
            for match in re.finditer(pattern, text):
                module_name = match.group(1)
                relationships.append((module_name, chunk_id, self.IMPORT))

        # Use existing code symbols from metadata
        for symbol in chunk.code_symbols:
            relationships.append((symbol, chunk_id, self.REFERENCE))

        return relationships

    def extract_document_relationships(
        self, chunk: ChunkMetadata
    ) -> List[Tuple[str, str, str]]:
        """Extract relationships from document chunks.

        Args:
            chunk: Document chunk metadata

        Returns:
            List of (entity, chunk_id, relationship_type) tuples
        """
        relationships = []

        if chunk.source_type != SourceType.DOCUMENT:
            return relationships

        chunk_id = chunk.chunk_id

        # Use parent section for hierarchy
        if chunk.parent_section:
            relationships.append((chunk.parent_section, chunk_id, self.HIERARCHY))

        # Use references from metadata
        for reference in chunk.references:
            relationships.append((reference, chunk_id, self.REFERENCE))

        # Use tags
        for tag in chunk.tags:
            relationships.append((tag, chunk_id, self.TAG))

        return relationships

    def extract_web_relationships(
        self, chunk: ChunkMetadata
    ) -> List[Tuple[str, str, str]]:
        """Extract relationships from web chunks.

        Args:
            chunk: Web chunk metadata

        Returns:
            List of (entity, chunk_id, relationship_type) tuples
        """
        relationships = []

        if chunk.source_type != SourceType.WEBPAGE:
            return relationships

        chunk_id = chunk.chunk_id

        # Extract domain from URL for hierarchy
        if chunk.source_path.startswith(("http://", "https://")):
            try:
                from urllib.parse import urlparse

                parsed = urlparse(chunk.source_path)
                domain = parsed.netloc
                relationships.append((domain, chunk_id, self.HIERARCHY))

                # Extract path segments for hierarchy
                path_parts = [p for p in parsed.path.split("/") if p]
                for part in path_parts:
                    if part:
                        relationships.append((part, chunk_id, self.HIERARCHY))
            except Exception:
                pass

        # Use metadata for links
        if "links" in chunk.metadata:
            for link in chunk.metadata["links"]:
                relationships.append((link, chunk_id, self.LINK))

        # Use tags and references
        for tag in chunk.tags:
            relationships.append((tag, chunk_id, self.TAG))

        for reference in chunk.references:
            relationships.append((reference, chunk_id, self.REFERENCE))

        return relationships

    def build_relationships_from_chunks(self, chunks: List[ChunkMetadata]) -> None:
        """Build relationships from a list of chunks.

        Args:
            chunks: List of chunk metadata to process
        """
        for chunk in chunks:
            relationships = []

            if chunk.source_type == SourceType.CODE:
                relationships.extend(self.extract_code_relationships(chunk))
            elif chunk.source_type == SourceType.DOCUMENT:
                relationships.extend(self.extract_document_relationships(chunk))
            elif chunk.source_type == SourceType.WEBPAGE:
                relationships.extend(self.extract_web_relationships(chunk))

            # Add all relationships to store
            for entity, chunk_id, rel_type in relationships:
                self.add_relationship(entity, chunk_id, rel_type)

    def get_stats(self) -> Dict[str, Any]:
        """Return relationship store statistics."""
        relationship_counts = defaultdict(int)
        total_relationships = 0

        for entity_rels in self._relationships.values():
            for rel_type, chunk_list in entity_rels.items():
                count = len(chunk_list)
                relationship_counts[rel_type] += count
                total_relationships += count

        return {
            "total_entities": len(self._relationships),
            "total_relationships": total_relationships,
            "relationship_types": dict(relationship_counts),
            "chunks_with_relationships": len(self._reverse_index),
        }

    def clear(self) -> None:
        """Clear all relationships."""
        self._relationships.clear()
        self._reverse_index.clear()
        self._entity_chunks.clear()

    def save_to_file(self, filepath: str, compress: bool = True) -> None:
        """Save relationships to file.

        Args:
            filepath: Path to save relationships
            compress: Whether to compress the file with gzip
        """
        data = {
            "relationships": dict(self._relationships),
            "reverse_index": dict(self._reverse_index),
            "entity_chunks": {k: list(v) for k, v in self._entity_chunks.items()},
        }

        if compress:
            with gzip.open(filepath, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

    def load_from_file(self, filepath: str) -> None:
        """Load relationships from file.

        Args:
            filepath: Path to load relationships from
        """
        file_path = Path(filepath)
        if not file_path.exists():
            return

        try:
            # Try to load as compressed file first
            with gzip.open(filepath, "rb") as f:
                data = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            # Fall back to uncompressed file
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        # Restore data structures
        self._relationships = defaultdict(
            lambda: defaultdict(list), data.get("relationships", {})
        )
        self._reverse_index = defaultdict(list, data.get("reverse_index", {}))
        self._entity_chunks = defaultdict(
            set, {k: set(v) for k, v in data.get("entity_chunks", {}).items()}
        )
