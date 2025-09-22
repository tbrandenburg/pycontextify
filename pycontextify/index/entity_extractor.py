"""Advanced entity extraction for PyContextify.

This module provides sophisticated entity recognition capabilities including
Named Entity Recognition (NER), custom pattern matching, and domain-specific
entity detection for enhanced relationship modeling.
"""

import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity with metadata."""

    name: str
    entity_type: str
    confidence: float
    source_position: Optional[Tuple[int, int]] = None  # Start, end character positions
    context: Optional[str] = None  # Surrounding context
    normalized_name: Optional[str] = None  # Normalized form of the name
    aliases: Optional[Set[str]] = None  # Alternative names for the entity
    attributes: Optional[Dict[str, Any]] = None  # Additional entity attributes


@dataclass
class EntityExtractionResult:
    """Result of entity extraction process."""

    entities: List[Entity]
    extraction_stats: Dict[str, Any]
    processing_time_ms: float


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""

    @abstractmethod
    def extract_entities(self, text: str, source_type: str = "unknown") -> List[Entity]:
        """Extract entities from text.

        Args:
            text: Text content to analyze
            source_type: Type of source (code, document, webpage)

        Returns:
            List of extracted entities
        """
        pass

    @abstractmethod
    def get_supported_entity_types(self) -> List[str]:
        """Get list of entity types this extractor supports."""
        pass


class RegexEntityExtractor(EntityExtractor):
    """Enhanced regex-based entity extractor with advanced patterns."""

    def __init__(self):
        """Initialize regex patterns for entity extraction."""
        self._code_patterns = {
            # Python patterns
            "function": [
                r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                r"async\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            ],
            "class": [
                r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]",
            ],
            "variable": [
                r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=",  # Variable assignment
                r"self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=",  # Instance variables
            ],
            "import": [
                r"import\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
                r"from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import",
            ],
            "constant": [
                r"^([A-Z][A-Z0-9_]*)\s*=",  # Constants (all caps)
            ],
            "method": [
                r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*self",  # Methods with self
            ],
            # Add more language patterns as needed
            "javascript_function": [
                r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*function\s*\(",
                r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=>\s*",  # Arrow functions
            ],
            "api_endpoint": [
                r'@app\.route\(["\']([^"\']*)["\']',  # Flask routes
                r'@[a-zA-Z_][a-zA-Z0-9_]*\.[a-z]+\(["\']([^"\']*)["\']',  # RESTful decorators
                r'(?:GET|POST|PUT|DELETE|PATCH)\s+["\']?([^"\'\\s]*)["\']?',  # HTTP methods
            ],
        }

        self._document_patterns = {
            "heading": [
                r"^#{1,6}\s+(.+)$",  # Markdown headings
                r"<h[1-6][^>]*>(.+?)</h[1-6]>",  # HTML headings
            ],
            "url": [
                r'(https?://[^\s<>"\'{}|\\^`]+)',
                r'(www\.[^\s<>"\'{}|\\^`]+)',
            ],
            "email": [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            ],
            "code_block": [
                r"```([a-zA-Z0-9+]*)\n(.*?)\n```",  # Markdown code blocks
                r"`([^`]+)`",  # Inline code
            ],
            "reference": [
                r"\[([^\]]+)\]\([^)]+\)",  # Markdown links
                r"see\s+([A-Z][a-zA-Z\s]+)",  # "see Chapter 5" style references
                r"(?:Section|Chapter|Figure|Table)\s+(\d+(?:\.\d+)*)",  # Numbered references
            ],
            "technical_term": [
                r"\b([A-Z]{2,})\b",  # Acronyms
                r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b",  # CamelCase terms
            ],
            "filename": [
                r"\b([a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)\b",  # File names with extensions
            ],
            "version": [
                r"\bv?(\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?)\b",  # Version numbers
            ],
        }

        self._domain_patterns = {
            # Programming domains
            "programming_concept": [
                r"\b(algorithm|data\s+structure|design\s+pattern|API|framework|library|module)\b",
                r"\b((?:object[- ]oriented|functional|procedural)\s+(?:programming|paradigm))\b",
            ],
            "database_term": [
                r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b",  # SQL keywords
                r"\b(database|table|schema|index|query|transaction)\b",
                r"\b([a-zA-Z_][a-zA-Z0-9_]*_id)\b",  # Database ID patterns
            ],
            "web_tech": [
                r"\b(HTML|CSS|JavaScript|React|Vue|Angular|Node\.js|Express)\b",
                r"\b(REST|GraphQL|API|endpoint|HTTP|HTTPS|JSON|XML)\b",
            ],
            "ai_ml_term": [
                r"\b(machine\s+learning|artificial\s+intelligence|neural\s+network|deep\s+learning)\b",
                r"\b(model|dataset|training|inference|prediction|classification)\b",
                r"\b(TensorFlow|PyTorch|scikit-learn|Keras|Pandas|NumPy)\b",
            ],
            "devops_term": [
                r"\b(Docker|Kubernetes|CI/CD|deployment|container|orchestration)\b",
                r"\b(AWS|Azure|GCP|cloud|microservice|serverless)\b",
            ],
        }

        # Compile all patterns for efficiency
        self._compiled_patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        for category, patterns_dict in [
            ("code", self._code_patterns),
            ("document", self._document_patterns),
            ("domain", self._domain_patterns),
        ]:
            self._compiled_patterns[category] = {}
            for pattern_type, patterns in patterns_dict.items():
                compiled = []
                for pattern in patterns:
                    try:
                        compiled.append(
                            re.compile(pattern, re.MULTILINE | re.IGNORECASE)
                        )
                    except re.error as e:
                        logger.warning(
                            f"Invalid regex pattern for {category}.{pattern_type}: {pattern} - {e}"
                        )
                self._compiled_patterns[category][pattern_type] = compiled

    def extract_entities(self, text: str, source_type: str = "unknown") -> List[Entity]:
        """Extract entities using regex patterns based on source type."""
        entities = []

        # Choose appropriate pattern categories based on source type
        if source_type in ["code", "codebase"]:
            categories = ["code", "domain"]
        elif source_type in ["document", "pdf", "markdown"]:
            categories = ["document", "domain"]
        elif source_type in ["webpage", "web"]:
            categories = ["document", "domain"]  # Web content is similar to documents
        else:
            categories = ["code", "document", "domain"]  # Extract all types if unknown

        # Extract entities from each category
        for category in categories:
            if category in self._compiled_patterns:
                category_entities = self._extract_from_category(text, category)
                entities.extend(category_entities)

        # Deduplicate and merge similar entities
        entities = self._deduplicate_entities(entities)

        # Add normalized names and calculate confidence scores
        entities = self._enhance_entities(entities, text)

        return entities

    def _extract_from_category(self, text: str, category: str) -> List[Entity]:
        """Extract entities from a specific pattern category."""
        entities = []

        for entity_type, compiled_patterns in self._compiled_patterns[category].items():
            for pattern in compiled_patterns:
                try:
                    for match in pattern.finditer(text):
                        # Handle different match group structures
                        if match.groups():
                            if len(match.groups()) == 1:
                                entity_name = match.group(1).strip()
                            else:
                                # For patterns with multiple groups, use the most relevant one
                                entity_name = match.group(
                                    -1
                                ).strip()  # Last group is usually the entity
                        else:
                            entity_name = match.group(0).strip()

                        if (
                            entity_name and len(entity_name) > 1
                        ):  # Filter out single characters
                            # Extract surrounding context
                            start, end = match.span()
                            context_start = max(0, start - 50)
                            context_end = min(len(text), end + 50)
                            context = text[context_start:context_end].strip()

                            entity = Entity(
                                name=entity_name,
                                entity_type=entity_type,
                                confidence=0.7,  # Base confidence for regex matches
                                source_position=(start, end),
                                context=context,
                            )
                            entities.append(entity)
                except IndexError as e:
                    logger.warning(f"Entity extractor {entity_type} failed: {e}")
                    continue

        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities and merge similar ones."""
        # Group entities by normalized name
        entity_groups = defaultdict(list)

        for entity in entities:
            normalized_name = entity.name.lower().strip()
            entity_groups[normalized_name].append(entity)

        deduplicated = []
        for name_group in entity_groups.values():
            if len(name_group) == 1:
                deduplicated.append(name_group[0])
            else:
                # Merge entities with same name but different types
                merged_entity = self._merge_entities(name_group)
                deduplicated.append(merged_entity)

        return deduplicated

    def _merge_entities(self, entities: List[Entity]) -> Entity:
        """Merge multiple entities with the same name."""
        if not entities:
            return None

        # Use the entity with highest confidence as base
        base_entity = max(entities, key=lambda e: e.confidence)

        # Combine entity types
        entity_types = [e.entity_type for e in entities]
        type_counts = Counter(entity_types)
        primary_type = type_counts.most_common(1)[0][0]

        # Combine aliases
        all_names = {e.name for e in entities}
        aliases = all_names - {base_entity.name}

        # Calculate merged confidence (average weighted by type frequency)
        merged_confidence = sum(e.confidence for e in entities) / len(entities)

        merged_entity = Entity(
            name=base_entity.name,
            entity_type=primary_type,
            confidence=min(
                merged_confidence * 1.1, 1.0
            ),  # Slight boost for multiple occurrences
            source_position=base_entity.source_position,
            context=base_entity.context,
            aliases=aliases if aliases else None,
            attributes={
                "occurrence_count": len(entities),
                "type_distribution": dict(type_counts),
            },
        )

        return merged_entity

    def _enhance_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """Enhance entities with additional metadata and confidence scoring."""
        enhanced = []

        for entity in entities:
            # Calculate confidence based on various factors
            confidence = entity.confidence

            # Boost confidence for entities that appear multiple times
            if entity.attributes and "occurrence_count" in entity.attributes:
                occurrence_count = entity.attributes["occurrence_count"]
                confidence = min(confidence * (1 + 0.1 * (occurrence_count - 1)), 1.0)

            # Boost confidence for well-formed entities
            if self._is_well_formed_entity(entity.name, entity.entity_type):
                confidence = min(confidence * 1.2, 1.0)

            # Add normalized name
            normalized_name = self._normalize_entity_name(
                entity.name, entity.entity_type
            )

            enhanced_entity = Entity(
                name=entity.name,
                entity_type=entity.entity_type,
                confidence=round(confidence, 3),
                source_position=entity.source_position,
                context=entity.context,
                normalized_name=normalized_name,
                aliases=entity.aliases,
                attributes=entity.attributes,
            )

            enhanced.append(enhanced_entity)

        return enhanced

    def _is_well_formed_entity(self, name: str, entity_type: str) -> bool:
        """Check if an entity name is well-formed for its type."""
        if not name or len(name) < 2:
            return False

        # Type-specific validation
        if entity_type in ["function", "method", "variable", "class"]:
            # Valid Python identifier
            return name.isidentifier() and not name.startswith("_")
        elif entity_type == "constant":
            # All uppercase with underscores
            return name.isupper() and name.replace("_", "").isalnum()
        elif entity_type == "url":
            # Basic URL validation
            return name.startswith(("http://", "https://", "www."))
        elif entity_type == "email":
            # Basic email validation
            return "@" in name and "." in name.split("@")[-1]

        return True  # Default to valid for unknown types

    def _normalize_entity_name(self, name: str, entity_type: str) -> str:
        """Normalize entity name for consistent comparison."""
        normalized = name.strip().lower()

        # Type-specific normalization
        if entity_type in ["function", "method", "class", "variable"]:
            # Keep original casing for code entities
            normalized = name.strip()
        elif entity_type == "url":
            # Normalize URLs
            if not normalized.startswith(("http://", "https://")):
                if normalized.startswith("www."):
                    normalized = "https://" + normalized
                else:
                    normalized = "https://www." + normalized
        elif entity_type == "heading":
            # Remove markdown heading markers
            normalized = re.sub(r"^#+\s*", "", normalized).strip()

        return normalized

    def get_supported_entity_types(self) -> List[str]:
        """Get list of entity types this extractor supports."""
        types = set()
        for category in self._compiled_patterns.values():
            types.update(category.keys())
        return sorted(list(types))


class NEREntityExtractor(EntityExtractor):
    """Named Entity Recognition extractor using spaCy (optional dependency)."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize NER extractor.

        Args:
            model_name: spaCy model name to use
        """
        self.model_name = model_name
        self._nlp = None
        self._available = self._try_load_spacy()

    def _try_load_spacy(self) -> bool:
        """Try to load spaCy model."""
        try:
            import spacy

            self._nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
            return True
        except (ImportError, OSError) as e:
            logger.warning(f"spaCy not available or model not found: {e}")
            return False

    def extract_entities(self, text: str, source_type: str = "unknown") -> List[Entity]:
        """Extract named entities using spaCy NER."""
        if not self._available or not self._nlp:
            logger.warning("spaCy NER not available, skipping NER extraction")
            return []

        entities = []

        try:
            # Process text with spaCy
            doc = self._nlp(text)

            # Extract named entities
            for ent in doc.ents:
                # Map spaCy entity labels to our entity types
                entity_type = self._map_spacy_label(ent.label_)

                # Calculate confidence (spaCy doesn't provide confidence directly)
                confidence = self._calculate_ner_confidence(ent, doc)

                # Extract context
                start_char = ent.start_char
                end_char = ent.end_char
                context_start = max(0, start_char - 50)
                context_end = min(len(text), end_char + 50)
                context = text[context_start:context_end].strip()

                entity = Entity(
                    name=ent.text.strip(),
                    entity_type=entity_type,
                    confidence=confidence,
                    source_position=(start_char, end_char),
                    context=context,
                    attributes={
                        "spacy_label": ent.label_,
                        "spacy_label_desc": (
                            spacy.explain(ent.label_) if "spacy" in globals() else None
                        ),
                    },
                )
                entities.append(entity)

        except Exception as e:
            logger.error(f"Error in spaCy NER extraction: {e}")

        return entities

    def _map_spacy_label(self, spacy_label: str) -> str:
        """Map spaCy entity labels to our entity types."""
        label_mapping = {
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "location",  # Geopolitical entity
            "LOC": "location",
            "PRODUCT": "product",
            "EVENT": "event",
            "WORK_OF_ART": "work_of_art",
            "LAW": "law",
            "LANGUAGE": "language",
            "DATE": "date",
            "TIME": "time",
            "PERCENT": "percent",
            "MONEY": "money",
            "QUANTITY": "quantity",
            "ORDINAL": "ordinal",
            "CARDINAL": "cardinal",
            "FACILITY": "facility",
            "FAC": "facility",
        }
        return label_mapping.get(spacy_label, spacy_label.lower())

    def _calculate_ner_confidence(self, ent, doc) -> float:
        """Calculate confidence score for NER entities."""
        # Base confidence for spaCy NER
        confidence = 0.8

        # Boost confidence for longer entities
        if len(ent.text) > 10:
            confidence = min(confidence * 1.1, 1.0)

        # Boost confidence for proper capitalization
        if ent.text.istitle() or ent.text.isupper():
            confidence = min(confidence * 1.1, 1.0)

        # Reduce confidence for very short entities
        if len(ent.text) < 3:
            confidence *= 0.8

        return round(confidence, 3)

    def get_supported_entity_types(self) -> List[str]:
        """Get list of entity types this extractor supports."""
        if not self._available:
            return []

        return [
            "person",
            "organization",
            "location",
            "product",
            "event",
            "work_of_art",
            "law",
            "language",
            "date",
            "time",
            "percent",
            "money",
            "quantity",
            "ordinal",
            "cardinal",
            "facility",
        ]


class CompositeEntityExtractor(EntityExtractor):
    """Composite entity extractor that combines multiple extraction methods."""

    def __init__(self, use_ner: bool = True):
        """Initialize composite extractor.

        Args:
            use_ner: Whether to use NER extraction (requires spaCy)
        """
        self.extractors = [RegexEntityExtractor()]

        if use_ner:
            ner_extractor = NEREntityExtractor()
            if ner_extractor._available:  # Only add if spaCy is available
                self.extractors.append(ner_extractor)

    def extract_entities(self, text: str, source_type: str = "unknown") -> List[Entity]:
        """Extract entities using all available extractors."""
        all_entities = []

        # Run all extractors
        for extractor in self.extractors:
            try:
                entities = extractor.extract_entities(text, source_type)
                all_entities.extend(entities)
            except Exception as e:
                logger.warning(
                    f"Entity extractor {type(extractor).__name__} failed: {e}"
                )

        # Merge and deduplicate results
        merged_entities = self._merge_extraction_results(all_entities)

        return merged_entities

    def _merge_extraction_results(self, entities: List[Entity]) -> List[Entity]:
        """Merge and deduplicate entities from multiple extractors."""
        # Group entities by normalized name
        entity_groups = defaultdict(list)

        for entity in entities:
            # Create a key for grouping similar entities
            key = self._create_entity_key(entity)
            entity_groups[key].append(entity)

        merged_entities = []
        for entity_group in entity_groups.values():
            if len(entity_group) == 1:
                merged_entities.append(entity_group[0])
            else:
                # Merge entities with similar names/types
                merged = self._merge_similar_entities(entity_group)
                merged_entities.append(merged)

        # Sort by confidence (highest first)
        merged_entities.sort(key=lambda e: e.confidence, reverse=True)

        return merged_entities

    def _create_entity_key(self, entity: Entity) -> str:
        """Create a key for grouping similar entities."""
        name = entity.name.lower().strip()
        # Group entities with very similar names and compatible types
        return f"{name}::{entity.entity_type}"

    def _merge_similar_entities(self, entities: List[Entity]) -> Entity:
        """Merge multiple similar entities into one."""
        # Choose the entity with highest confidence as base
        base_entity = max(entities, key=lambda e: e.confidence)

        # Calculate merged confidence (weighted average)
        total_confidence = sum(e.confidence for e in entities)
        avg_confidence = total_confidence / len(entities)

        # Boost confidence for entities found by multiple extractors
        confidence_boost = 1.0 + (0.1 * (len(entities) - 1))
        merged_confidence = min(avg_confidence * confidence_boost, 1.0)

        # Combine attributes
        merged_attributes = base_entity.attributes or {}
        merged_attributes["extractor_count"] = len(entities)
        merged_attributes["source_extractors"] = [type(e).__name__ for e in entities]

        # Combine aliases
        all_names = {e.name for e in entities}
        aliases = all_names - {base_entity.name}

        merged_entity = Entity(
            name=base_entity.name,
            entity_type=base_entity.entity_type,
            confidence=round(merged_confidence, 3),
            source_position=base_entity.source_position,
            context=base_entity.context,
            normalized_name=base_entity.normalized_name,
            aliases=aliases if aliases else base_entity.aliases,
            attributes=merged_attributes,
        )

        return merged_entity

    def get_supported_entity_types(self) -> List[str]:
        """Get list of entity types supported by all extractors."""
        all_types = set()
        for extractor in self.extractors:
            all_types.update(extractor.get_supported_entity_types())
        return sorted(list(all_types))


def extract_entities_from_text(
    text: str,
    source_type: str = "unknown",
    use_ner: bool = True,
    min_confidence: float = 0.5,
) -> EntityExtractionResult:
    """Main function to extract entities from text.

    Args:
        text: Text content to analyze
        source_type: Type of source (code, document, webpage)
        use_ner: Whether to use NER extraction
        min_confidence: Minimum confidence threshold for entities

    Returns:
        EntityExtractionResult with extracted entities and metadata
    """
    import time

    start_time = time.time()

    # Create extractor
    extractor = CompositeEntityExtractor(use_ner=use_ner)

    # Extract entities
    entities = extractor.extract_entities(text, source_type)

    # Filter by confidence
    filtered_entities = [e for e in entities if e.confidence >= min_confidence]

    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Calculate statistics
    stats = {
        "total_entities": len(entities),
        "filtered_entities": len(filtered_entities),
        "entity_types": list(set(e.entity_type for e in filtered_entities)),
        "avg_confidence": (
            sum(e.confidence for e in filtered_entities) / len(filtered_entities)
            if filtered_entities
            else 0
        ),
        "extractors_used": [type(ext).__name__ for ext in extractor.extractors],
    }

    return EntityExtractionResult(
        entities=filtered_entities,
        extraction_stats=stats,
        processing_time_ms=processing_time,
    )
