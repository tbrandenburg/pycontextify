"""Advanced relationship modeling for PyContextify.

This module provides sophisticated relationship modeling capabilities including
semantic relationships, hierarchical connections, and cross-document relationship
discovery for enhanced knowledge graph construction.
"""

import logging
import re
import math
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Set, Tuple, Optional, Union
from .entity_extractor import Entity, extract_entities_from_text
from .metadata import ChunkMetadata, SourceType

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Enhanced relationship types with semantic meaning."""
    
    # Basic relationships (from original system)
    REFERENCE = "reference"
    FUNCTION_CALL = "function_call"
    IMPORT = "import"
    LINK = "link"
    HIERARCHY = "hierarchy"
    TAG = "tag"
    
    # Enhanced semantic relationships
    DEFINES = "defines"  # Entity A defines entity B
    IMPLEMENTS = "implements"  # Entity A implements entity B
    INHERITS = "inherits"  # Entity A inherits from entity B
    USES = "uses"  # Entity A uses entity B
    CONTAINS = "contains"  # Entity A contains entity B
    PART_OF = "part_of"  # Entity A is part of entity B
    SIMILAR_TO = "similar_to"  # Entity A is similar to entity B
    DEPENDS_ON = "depends_on"  # Entity A depends on entity B
    MENTIONS = "mentions"  # Entity A mentions entity B
    RELATED_TO = "related_to"  # Generic semantic relationship
    
    # Temporal relationships
    PRECEDES = "precedes"  # Entity A comes before entity B
    FOLLOWS = "follows"  # Entity A comes after entity B
    
    # Causal relationships
    CAUSES = "causes"  # Entity A causes entity B
    CAUSED_BY = "caused_by"  # Entity A is caused by entity B
    
    # Comparative relationships
    ALTERNATIVE_TO = "alternative_to"  # Entity A is an alternative to entity B
    CONTRASTS_WITH = "contrasts_with"  # Entity A contrasts with entity B


@dataclass
class Relationship:
    """Represents a relationship between two entities."""
    
    source_entity: str
    target_entity: str
    relationship_type: RelationshipType
    confidence: float
    source_chunk_id: Optional[str] = None
    target_chunk_id: Optional[str] = None
    context: Optional[str] = None  # Textual context where relationship was found
    attributes: Optional[Dict[str, Any]] = None  # Additional relationship metadata
    
    def __post_init__(self):
        """Validate relationship after initialization."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class RelationshipExtractionResult:
    """Result of relationship extraction process."""
    
    relationships: List[Relationship]
    entities_found: List[Entity]
    extraction_stats: Dict[str, Any]
    processing_time_ms: float


class RelationshipExtractor(ABC):
    """Abstract base class for relationship extractors."""
    
    @abstractmethod
    def extract_relationships(
        self,
        chunk: ChunkMetadata,
        entities: Optional[List[Entity]] = None
    ) -> List[Relationship]:
        """Extract relationships from a chunk.
        
        Args:
            chunk: Chunk metadata to analyze
            entities: Pre-extracted entities (optional)
            
        Returns:
            List of extracted relationships
        """
        pass


class SemanticRelationshipExtractor(RelationshipExtractor):
    """Advanced semantic relationship extractor."""
    
    def __init__(self):
        """Initialize semantic patterns for relationship extraction."""
        # Patterns for detecting semantic relationships in text
        self._semantic_patterns = {
            RelationshipType.DEFINES: [
                r'(\w+)\s+(?:is\s+)?(?:defined\s+as|defines?)\s+(\w+)',
                r'(\w+):\s*(.+?)(?:\.|$)',  # Definition patterns
                r'(\w+)\s+(?:means|refers\s+to)\s+(.+?)(?:\.|$)',
            ],
            RelationshipType.IMPLEMENTS: [
                r'(\w+)\s+implements?\s+(\w+)',
                r'(\w+)\s+(?:is\s+an?\s+)?implementation\s+of\s+(\w+)',
                r'class\s+(\w+)\((\w+)\)',  # Python inheritance (also implements interface)
            ],
            RelationshipType.INHERITS: [
                r'(\w+)\s+inherits?\s+(?:from\s+)?(\w+)',
                r'(\w+)\s+extends\s+(\w+)',
                r'class\s+(\w+)\((\w+)\)',  # Python inheritance
            ],
            RelationshipType.USES: [
                r'(\w+)\s+uses?\s+(\w+)',
                r'(?:using|with)\s+(\w+).*?(\w+)',
                r'(\w+)\.(\w+)\(',  # Method calls
                r'import\s+(\w+).*?(\w+)',  # Import usage
            ],
            RelationshipType.DEPENDS_ON: [
                r'(\w+)\s+depends?\s+on\s+(\w+)',
                r'(\w+)\s+requires?\s+(\w+)',
                r'(\w+)\s+needs?\s+(\w+)',
            ],
            RelationshipType.CONTAINS: [
                r'(\w+)\s+contains?\s+(\w+)',
                r'(\w+)\s+(?:has|includes?)\s+(\w+)',
                r'(\w+)\s+consists?\s+of\s+(\w+)',
            ],
            RelationshipType.PART_OF: [
                r'(\w+)\s+(?:is\s+)?part\s+of\s+(\w+)',
                r'(\w+)\s+belongs?\s+to\s+(\w+)',
                r'(\w+)\s+(?:is\s+)?(?:a\s+)?(?:component|element|member)\s+of\s+(\w+)',
            ],
            RelationshipType.SIMILAR_TO: [
                r'(\w+)\s+(?:is\s+)?similar\s+to\s+(\w+)',
                r'(\w+)\s+(?:is\s+)?like\s+(\w+)',
                r'(\w+)\s+resembles\s+(\w+)',
            ],
            RelationshipType.ALTERNATIVE_TO: [
                r'(\w+)\s+(?:is\s+an?\s+)?alternative\s+to\s+(\w+)',
                r'(\w+)\s+(?:vs|versus)\s+(\w+)',
                r'(?:instead\s+of|rather\s+than)\s+(\w+).*?(\w+)',
            ],
            RelationshipType.CAUSES: [
                r'(\w+)\s+causes?\s+(\w+)',
                r'(\w+)\s+(?:leads?\s+to|results?\s+in)\s+(\w+)',
                r'(?:because\s+of|due\s+to)\s+(\w+).*?(\w+)',
            ],
            RelationshipType.MENTIONS: [
                r'(?:see\s+also|refers?\s+to|mentions?)\s+(\w+)',
                r'(\w+)\s+(?:discusses?|talks?\s+about)\s+(\w+)',
            ]
        }
        
        # Compile patterns for efficiency
        self._compiled_patterns = {}
        for rel_type, patterns in self._semantic_patterns.items():
            compiled = []
            for pattern in patterns:
                try:
                    compiled.append(re.compile(pattern, re.IGNORECASE | re.MULTILINE))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for {rel_type}: {pattern} - {e}")
            self._compiled_patterns[rel_type] = compiled
        
        # Entity co-occurrence patterns
        self._cooccurrence_window = 100  # Characters within which entities are considered co-occurring
    
    def extract_relationships(
        self,
        chunk: ChunkMetadata,
        entities: Optional[List[Entity]] = None
    ) -> List[Relationship]:
        """Extract semantic relationships from chunk."""
        if not entities:
            # Extract entities if not provided
            extraction_result = extract_entities_from_text(
                chunk.chunk_text,
                chunk.source_type.value if hasattr(chunk.source_type, 'value') else str(chunk.source_type)
            )
            entities = extraction_result.entities
        
        relationships = []
        
        # Extract pattern-based relationships
        pattern_relationships = self._extract_pattern_relationships(chunk, entities)
        relationships.extend(pattern_relationships)
        
        # Extract co-occurrence relationships
        cooccurrence_relationships = self._extract_cooccurrence_relationships(chunk, entities)
        relationships.extend(cooccurrence_relationships)
        
        # Extract domain-specific relationships
        domain_relationships = self._extract_domain_relationships(chunk, entities)
        relationships.extend(domain_relationships)
        
        return relationships
    
    def _extract_pattern_relationships(
        self, 
        chunk: ChunkMetadata,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract relationships using semantic patterns."""
        relationships = []
        text = chunk.chunk_text
        
        # Create entity lookup for quick matching
        entity_names = {e.name.lower(): e for e in entities}
        
        for rel_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    if len(match.groups()) >= 2:
                        source_name = match.group(1).strip()
                        target_name = match.group(2).strip()
                        
                        # Check if both entities exist in our extracted entities
                        source_key = source_name.lower()
                        target_key = target_name.lower()
                        
                        if source_key in entity_names and target_key in entity_names:
                            # Extract context around the match
                            start, end = match.span()
                            context_start = max(0, start - 50)
                            context_end = min(len(text), end + 50)
                            context = text[context_start:context_end].strip()
                            
                            # Calculate confidence based on pattern specificity and entity confidence
                            source_entity = entity_names[source_key]
                            target_entity = entity_names[target_key]
                            confidence = self._calculate_pattern_confidence(
                                rel_type, match, source_entity, target_entity
                            )
                            
                            relationship = Relationship(
                                source_entity=source_entity.name,
                                target_entity=target_entity.name,
                                relationship_type=rel_type,
                                confidence=confidence,
                                source_chunk_id=chunk.chunk_id,
                                context=context,
                                attributes={
                                    'extraction_method': 'pattern',
                                    'pattern_match': match.group(0),
                                    'source_entity_type': source_entity.entity_type,
                                    'target_entity_type': target_entity.entity_type
                                }
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def _extract_cooccurrence_relationships(
        self,
        chunk: ChunkMetadata,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract relationships based on entity co-occurrence."""
        relationships = []
        text = chunk.chunk_text
        
        # Find entities that co-occur within a window
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1.source_position and entity2.source_position:
                    # Calculate distance between entities
                    pos1 = entity1.source_position
                    pos2 = entity2.source_position
                    
                    distance = min(
                        abs(pos1[0] - pos2[1]),  # Distance from end of entity1 to start of entity2
                        abs(pos2[0] - pos1[1])   # Distance from end of entity2 to start of entity1
                    )
                    
                    if distance <= self._cooccurrence_window:
                        # Entities co-occur, create a generic relationship
                        confidence = self._calculate_cooccurrence_confidence(
                            entity1, entity2, distance
                        )
                        
                        # Determine relationship direction and type based on entity types and positions
                        rel_type, source_entity, target_entity = self._determine_cooccurrence_relationship(
                            entity1, entity2, pos1, pos2
                        )
                        
                        # Extract context around both entities
                        context_start = max(0, min(pos1[0], pos2[0]) - 30)
                        context_end = min(len(text), max(pos1[1], pos2[1]) + 30)
                        context = text[context_start:context_end].strip()
                        
                        relationship = Relationship(
                            source_entity=source_entity.name,
                            target_entity=target_entity.name,
                            relationship_type=rel_type,
                            confidence=confidence,
                            source_chunk_id=chunk.chunk_id,
                            context=context,
                            attributes={
                                'extraction_method': 'cooccurrence',
                                'distance_chars': distance,
                                'source_entity_type': source_entity.entity_type,
                                'target_entity_type': target_entity.entity_type
                            }
                        )
                        relationships.append(relationship)
                else:
                    # If no positions available, create a generic relationship based on entity types
                    if self._should_create_generic_relationship(entity1, entity2):
                        confidence = 0.3  # Lower confidence for generic relationships
                        rel_type = RelationshipType.MENTIONS  # Generic relationship type
                        
                        relationship = Relationship(
                            source_entity=entity1.name,
                            target_entity=entity2.name,
                            relationship_type=rel_type,
                            confidence=confidence,
                            source_chunk_id=chunk.chunk_id,
                            context=chunk.chunk_text[:100] + "..." if len(chunk.chunk_text) > 100 else chunk.chunk_text,
                            attributes={
                                'extraction_method': 'generic_cooccurrence',
                                'source_entity_type': entity1.entity_type,
                                'target_entity_type': entity2.entity_type
                            }
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _extract_domain_relationships(
        self,
        chunk: ChunkMetadata,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract domain-specific relationships based on source type and entity types."""
        relationships = []
        
        if chunk.source_type == SourceType.CODE:
            relationships.extend(self._extract_code_relationships(chunk, entities))
        elif chunk.source_type == SourceType.DOCUMENT:
            relationships.extend(self._extract_document_relationships(chunk, entities))
        elif chunk.source_type == SourceType.WEBPAGE:
            relationships.extend(self._extract_web_relationships(chunk, entities))
        
        return relationships
    
    def _extract_code_relationships(
        self,
        chunk: ChunkMetadata,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract code-specific relationships."""
        relationships = []
        text = chunk.chunk_text
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.entity_type].append(entity)
        
        # Class-method relationships
        classes = entities_by_type.get('class', [])
        methods = entities_by_type.get('method', []) + entities_by_type.get('function', [])
        
        for class_entity in classes:
            for method_entity in methods:
                # Check if method is defined within class scope
                if (class_entity.source_position and method_entity.source_position and
                    class_entity.source_position[0] < method_entity.source_position[0]):
                    # For same-chunk relationships, assume they're related if method comes after class
                    relationship = Relationship(
                        source_entity=class_entity.name,
                        target_entity=method_entity.name,
                        relationship_type=RelationshipType.CONTAINS,
                        confidence=0.8,
                        source_chunk_id=chunk.chunk_id,
                        attributes={
                            'extraction_method': 'code_structure',
                            'relationship_context': 'class_method'
                        }
                    )
                    relationships.append(relationship)
        
        # Import-usage relationships
        imports = entities_by_type.get('import', [])
        for import_entity in imports:
            for other_entity in entities:
                if (other_entity.entity_type in ['function', 'class', 'variable'] and
                    other_entity != import_entity):
                    # Check if the imported module is used
                    import_name = import_entity.name.split('.')[-1]  # Last part of module name
                    context_text = other_entity.context if other_entity.context else ""
                    if import_name.lower() in context_text.lower():
                        relationship = Relationship(
                            source_entity=other_entity.name,
                            target_entity=import_entity.name,
                            relationship_type=RelationshipType.USES,
                            confidence=0.7,
                            source_chunk_id=chunk.chunk_id,
                            attributes={
                                'extraction_method': 'code_usage',
                                'relationship_context': 'import_usage'
                            }
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _extract_document_relationships(
        self,
        chunk: ChunkMetadata,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract document-specific relationships."""
        relationships = []
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.entity_type].append(entity)
        
        # Heading-content relationships
        headings = entities_by_type.get('heading', [])
        other_entities = [e for e in entities if e.entity_type != 'heading']
        
        for heading in headings:
            for entity in other_entities:
                # Check if entity appears after heading and before next heading
                if (heading.source_position and entity.source_position and
                    heading.source_position[1] < entity.source_position[0]):
                    
                    # Find if there's a closer heading
                    closer_heading = None
                    for other_heading in headings:
                        if (other_heading != heading and
                            other_heading.source_position and
                            heading.source_position[1] < other_heading.source_position[0] < entity.source_position[0]):
                            if not closer_heading or other_heading.source_position[0] < closer_heading.source_position[0]:
                                closer_heading = other_heading
                    
                    if not closer_heading:  # Entity belongs to this heading section
                        relationship = Relationship(
                            source_entity=heading.name,
                            target_entity=entity.name,
                            relationship_type=RelationshipType.CONTAINS,
                            confidence=0.8,
                            source_chunk_id=chunk.chunk_id,
                            attributes={
                                'extraction_method': 'document_structure',
                                'relationship_context': 'heading_content'
                            }
                        )
                        relationships.append(relationship)
        
        # Reference relationships
        references = entities_by_type.get('reference', [])
        for ref_entity in references:
            for other_entity in entities:
                if other_entity != ref_entity and other_entity.entity_type in ['heading', 'technical_term']:
                    # Check if reference mentions the other entity
                    if other_entity.name.lower() in ref_entity.name.lower():
                        relationship = Relationship(
                            source_entity=ref_entity.name,
                            target_entity=other_entity.name,
                            relationship_type=RelationshipType.MENTIONS,
                            confidence=0.7,
                            source_chunk_id=chunk.chunk_id,
                            attributes={
                                'extraction_method': 'document_reference',
                                'relationship_context': 'cross_reference'
                            }
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _extract_web_relationships(
        self,
        chunk: ChunkMetadata,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Extract web-specific relationships."""
        relationships = []
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.entity_type].append(entity)
        
        # URL-domain relationships
        urls = entities_by_type.get('url', [])
        for url_entity in urls:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url_entity.name)
                domain = parsed.netloc
                
                # Create domain entity if not exists
                domain_entities = [e for e in entities if e.name.lower() == domain.lower()]
                if not domain_entities:
                    continue
                
                domain_entity = domain_entities[0]
                relationship = Relationship(
                    source_entity=url_entity.name,
                    target_entity=domain_entity.name,
                    relationship_type=RelationshipType.PART_OF,
                    confidence=0.9,
                    source_chunk_id=chunk.chunk_id,
                    attributes={
                        'extraction_method': 'web_structure',
                        'relationship_context': 'url_domain'
                    }
                )
                relationships.append(relationship)
            except Exception:
                pass
        
        return relationships
    
    def _calculate_pattern_confidence(
        self,
        rel_type: RelationshipType,
        match: re.Match,
        source_entity: Entity,
        target_entity: Entity
    ) -> float:
        """Calculate confidence for pattern-based relationships."""
        base_confidence = 0.8
        
        # Adjust based on entity confidences
        entity_confidence_factor = (source_entity.confidence + target_entity.confidence) / 2
        confidence = base_confidence * entity_confidence_factor
        
        # Boost confidence for specific relationship types
        high_confidence_types = {
            RelationshipType.DEFINES,
            RelationshipType.IMPLEMENTS,
            RelationshipType.INHERITS
        }
        if rel_type in high_confidence_types:
            confidence *= 1.2
        
        # Reduce confidence for generic relationships
        generic_types = {RelationshipType.MENTIONS, RelationshipType.RELATED_TO}
        if rel_type in generic_types:
            confidence *= 0.8
        
        return min(confidence, 1.0)
    
    def _calculate_cooccurrence_confidence(
        self,
        entity1: Entity,
        entity2: Entity,
        distance: int
    ) -> float:
        """Calculate confidence for co-occurrence relationships."""
        # Base confidence decreases with distance
        distance_factor = max(0.1, 1.0 - (distance / self._cooccurrence_window))
        
        # Factor in entity confidences
        entity_confidence = (entity1.confidence + entity2.confidence) / 2
        
        # Base confidence for co-occurrence is lower than pattern-based
        base_confidence = 0.5
        
        confidence = base_confidence * distance_factor * entity_confidence
        return min(confidence, 0.8)  # Cap at 0.8 for co-occurrence
    
    def _determine_cooccurrence_relationship(
        self,
        entity1: Entity,
        entity2: Entity,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> Tuple[RelationshipType, Entity, Entity]:
        """Determine relationship type and direction for co-occurring entities."""
        # Default to generic relationship
        rel_type = RelationshipType.RELATED_TO
        
        # Determine direction based on position (left-to-right reading)
        if pos1[0] < pos2[0]:
            source_entity, target_entity = entity1, entity2
        else:
            source_entity, target_entity = entity2, entity1
        
        # Adjust relationship type based on entity types
        source_type = source_entity.entity_type
        target_type = target_entity.entity_type
        
        # Code-specific relationships
        if source_type == 'class' and target_type in ['method', 'function']:
            rel_type = RelationshipType.CONTAINS
        elif source_type in ['function', 'method'] and target_type == 'variable':
            rel_type = RelationshipType.USES
        elif source_type == 'import' and target_type in ['class', 'function']:
            rel_type = RelationshipType.USES
            source_entity, target_entity = target_entity, source_entity  # Reverse direction
        
        # Document-specific relationships
        elif source_type == 'heading' and target_type in ['technical_term', 'reference']:
            rel_type = RelationshipType.CONTAINS
        
        # Domain relationships
        elif source_type in ['programming_concept', 'technical_term'] and target_type in ['function', 'class']:
            rel_type = RelationshipType.RELATED_TO
        
        return rel_type, source_entity, target_entity
    
    def _should_create_generic_relationship(
        self,
        entity1: Entity,
        entity2: Entity
    ) -> bool:
        """Determine if a generic relationship should be created between entities."""
        # Don't create relationships between entities of the same type unless they're code entities
        if (entity1.entity_type == entity2.entity_type and
            entity1.entity_type not in ['function', 'class', 'variable', 'method']):
            return False
        
        # Don't create relationships between generic terms
        generic_types = ['technical_term', 'programming_concept']
        if entity1.entity_type in generic_types and entity2.entity_type in generic_types:
            return False
        
        # Create relationships between code entities and other types
        code_types = ['function', 'class', 'method', 'variable', 'import']
        if (entity1.entity_type in code_types or entity2.entity_type in code_types):
            return True
        
        # Create relationships between different document entity types
        if entity1.entity_type != entity2.entity_type:
            return True
        
        return False


class CrossDocumentRelationshipDetector:
    """Detects relationships across different documents and chunks."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        """Initialize cross-document relationship detector.
        
        Args:
            similarity_threshold: Minimum similarity for creating relationships
        """
        self.similarity_threshold = similarity_threshold
    
    def find_cross_document_relationships(
        self,
        chunks: List[ChunkMetadata],
        entities_by_chunk: Dict[str, List[Entity]]
    ) -> List[Relationship]:
        """Find relationships between entities across different chunks/documents.
        
        Args:
            chunks: List of chunk metadata
            entities_by_chunk: Dictionary mapping chunk IDs to their entities
            
        Returns:
            List of cross-document relationships
        """
        relationships = []
        
        # Group entities by normalized name for similarity matching
        entity_groups = self._group_similar_entities(entities_by_chunk)
        
        # Find cross-chunk relationships for each entity group
        for group_entities in entity_groups.values():
            if len(group_entities) > 1:  # Only create relationships if entities appear in multiple chunks
                cross_relationships = self._create_cross_chunk_relationships(group_entities)
                relationships.extend(cross_relationships)
        
        # Find semantic similarity relationships
        semantic_relationships = self._find_semantic_relationships(entities_by_chunk, chunks)
        relationships.extend(semantic_relationships)
        
        return relationships
    
    def _group_similar_entities(
        self,
        entities_by_chunk: Dict[str, List[Entity]]
    ) -> Dict[str, List[Tuple[Entity, str]]]:
        """Group similar entities across chunks."""
        entity_groups = defaultdict(list)
        
        for chunk_id, entities in entities_by_chunk.items():
            for entity in entities:
                # Create a normalized key for grouping
                normalized_name = self._normalize_entity_name(entity.name, entity.entity_type)
                entity_groups[normalized_name].append((entity, chunk_id))
        
        return entity_groups
    
    def _normalize_entity_name(self, name: str, entity_type: str) -> str:
        """Normalize entity name for cross-document matching."""
        normalized = name.lower().strip()
        
        # Remove common prefixes/suffixes for better matching
        if entity_type in ['function', 'method']:
            # Remove common function prefixes
            prefixes = ['get_', 'set_', 'is_', 'has_', 'can_', 'should_']
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    break
        
        elif entity_type == 'class':
            # Remove common class suffixes
            suffixes = ['_class', '_impl', '_implementation', '_service', '_manager']
            for suffix in suffixes:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)]
                    break
        
        return normalized
    
    def _create_cross_chunk_relationships(
        self,
        group_entities: List[Tuple[Entity, str]]
    ) -> List[Relationship]:
        """Create relationships between entities that appear in multiple chunks."""
        relationships = []
        
        if len(group_entities) < 2:
            return relationships
        
        # Sort by confidence to get the best representative entity
        group_entities.sort(key=lambda x: x[0].confidence, reverse=True)
        primary_entity, primary_chunk = group_entities[0]
        
        # Create relationships from other occurrences to the primary entity
        for entity, chunk_id in group_entities[1:]:
            if chunk_id != primary_chunk:  # Only cross-chunk relationships
                confidence = self._calculate_cross_document_confidence(
                    primary_entity, entity
                )
                
                relationship = Relationship(
                    source_entity=primary_entity.name,
                    target_entity=entity.name,
                    relationship_type=RelationshipType.SIMILAR_TO,
                    confidence=confidence,
                    source_chunk_id=primary_chunk,
                    target_chunk_id=chunk_id,
                    attributes={
                        'extraction_method': 'cross_document',
                        'relationship_context': 'entity_correspondence',
                        'primary_entity_confidence': primary_entity.confidence,
                        'secondary_entity_confidence': entity.confidence
                    }
                )
                relationships.append(relationship)
        
        return relationships
    
    def _find_semantic_relationships(
        self,
        entities_by_chunk: Dict[str, List[Entity]],
        chunks: List[ChunkMetadata]
    ) -> List[Relationship]:
        """Find semantic relationships between entities across chunks."""
        relationships = []
        
        # Create chunk lookup
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Compare entities across different chunks
        chunk_pairs = []
        chunk_ids = list(entities_by_chunk.keys())
        for i, chunk_id1 in enumerate(chunk_ids):
            for chunk_id2 in chunk_ids[i+1:]:
                chunk_pairs.append((chunk_id1, chunk_id2))
        
        for chunk_id1, chunk_id2 in chunk_pairs:
            entities1 = entities_by_chunk[chunk_id1]
            entities2 = entities_by_chunk[chunk_id2]
            
            # Find semantically related entities
            for entity1 in entities1:
                for entity2 in entities2:
                    if entity1.entity_type == entity2.entity_type:  # Same type entities
                        similarity = self._calculate_semantic_similarity(entity1, entity2)
                        
                        if similarity >= self.similarity_threshold:
                            relationship = Relationship(
                                source_entity=entity1.name,
                                target_entity=entity2.name,
                                relationship_type=RelationshipType.SIMILAR_TO,
                                confidence=similarity,
                                source_chunk_id=chunk_id1,
                                target_chunk_id=chunk_id2,
                                attributes={
                                    'extraction_method': 'semantic_similarity',
                                    'similarity_score': similarity,
                                    'entity_type': entity1.entity_type
                                }
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def _calculate_cross_document_confidence(
        self,
        entity1: Entity,
        entity2: Entity
    ) -> float:
        """Calculate confidence for cross-document relationships."""
        # Base confidence for cross-document relationships
        base_confidence = 0.6
        
        # Factor in entity confidences
        avg_confidence = (entity1.confidence + entity2.confidence) / 2
        
        # Boost confidence for exact name matches
        if entity1.name == entity2.name:
            confidence = base_confidence * avg_confidence * 1.3
        else:
            confidence = base_confidence * avg_confidence
        
        return min(confidence, 1.0)
    
    def _calculate_semantic_similarity(
        self,
        entity1: Entity,
        entity2: Entity
    ) -> float:
        """Calculate semantic similarity between entities."""
        # Simple string similarity for now
        name1 = entity1.name.lower()
        name2 = entity2.name.lower()
        
        # Exact match
        if name1 == name2:
            return 1.0
        
        # Jaccard similarity for words
        words1 = set(name1.split('_') + name1.split())
        words2 = set(name2.split('_') + name2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Boost similarity for entities with similar contexts
        context_similarity = 0.0
        if entity1.context and entity2.context:
            context1_words = set(entity1.context.lower().split())
            context2_words = set(entity2.context.lower().split())
            context_intersection = len(context1_words & context2_words)
            context_union = len(context1_words | context2_words)
            context_similarity = context_intersection / context_union if context_union > 0 else 0.0
        
        # Combine similarities
        combined_similarity = (jaccard * 0.7) + (context_similarity * 0.3)
        
        return combined_similarity