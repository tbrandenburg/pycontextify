"""
Search Intelligence Framework for PyContextify.

This module provides intelligent search capabilities including query learning,
pattern recognition, and adaptive search improvements based on usage patterns.
"""

import json
import logging
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import threading

logger = logging.getLogger(__name__)


class SearchIntelligenceFramework:
    """Framework for learning from search patterns and providing intelligent improvements."""
    
    def __init__(self, storage_path: Optional[str] = None, max_history: int = 10000):
        """Initialize the search intelligence framework.
        
        Args:
            storage_path: Path to store intelligence data
            max_history: Maximum number of search records to keep
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_history = max_history
        self._lock = threading.Lock()
        
        # Intelligence data structures
        self.search_history: List[Dict[str, Any]] = []
        self.query_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.domain_preferences: Dict[str, float] = defaultdict(float)
        self.successful_strategies: Dict[str, List[str]] = defaultdict(list)
        self.query_refinement_paths: List[List[str]] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Load existing data if available
        self._load_intelligence_data()
        
    def record_search(self, query: str, query_analysis: Dict[str, Any], 
                     search_response: 'SearchResponse', user_feedback: Optional[Dict] = None) -> None:
        """Record a search interaction for learning.
        
        Args:
            query: Original search query
            query_analysis: Analysis results from query analysis
            search_response: Search response object
            user_feedback: Optional user feedback on search quality
        """
        with self._lock:
            search_record = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "normalized_query": query_analysis.get("normalized_query", query),
                "intent": query_analysis.get("detected_intent", "unknown"),
                "domain": query_analysis.get("domain_analysis", {}).get("primary_domain", "general"),
                "complexity": query_analysis.get("complexity_analysis", {}).get("complexity_level", "moderate"),
                "results_count": len(search_response.results),
                "success": search_response.success,
                "response_time": search_response.performance.get("total_time_ms", 0) if search_response.performance else 0,
                "search_mode": search_response.performance.get("search_mode", "unknown") if search_response.performance else "unknown",
                "user_feedback": user_feedback or {}
            }
            
            # Add to search history
            self.search_history.append(search_record)
            
            # Maintain history limit
            if len(self.search_history) > self.max_history:
                self.search_history = self.search_history[-self.max_history:]
            
            # Update patterns and metrics
            self._update_patterns(search_record, query_analysis)
            self._update_performance_metrics(search_record)
            
            # Auto-save periodically
            if len(self.search_history) % 100 == 0:
                self._save_intelligence_data()
    
    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """Generate intelligent query suggestions based on learned patterns.
        
        Args:
            partial_query: Partial query string
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggestion dictionaries with query and metadata
        """
        suggestions = []
        partial_lower = partial_query.lower().strip()
        
        if not partial_lower or len(partial_lower) < 2:
            # Return popular queries for very short input
            return self._get_popular_queries(max_suggestions)
        
        # Find similar queries from history
        similar_queries = []
        for record in self.search_history[-1000:]:  # Check recent history
            query = record["query"].lower()
            if partial_lower in query and query != partial_lower:
                similarity_score = self._calculate_query_similarity(partial_lower, query)
                if similarity_score > 0.3:  # Minimum similarity threshold
                    similar_queries.append({
                        "query": record["query"],
                        "score": similarity_score,
                        "success_rate": 1.0 if record["success"] else 0.0,
                        "domain": record["domain"],
                        "intent": record["intent"]
                    })
        
        # Sort by score and success rate
        similar_queries.sort(key=lambda x: (x["score"], x["success_rate"]), reverse=True)
        
        # Generate suggestions with metadata
        seen_queries = set()
        for item in similar_queries[:max_suggestions]:
            if item["query"] not in seen_queries:
                suggestions.append({
                    "query": item["query"],
                    "confidence": round(item["score"], 3),
                    "domain": item["domain"],
                    "intent": item["intent"],
                    "suggestion_type": "similar_query"
                })
                seen_queries.add(item["query"])
        
        # Add pattern-based suggestions if we don't have enough
        if len(suggestions) < max_suggestions:
            pattern_suggestions = self._generate_pattern_suggestions(partial_query, max_suggestions - len(suggestions))
            suggestions.extend(pattern_suggestions)
        
        return suggestions[:max_suggestions]
    
    def get_adaptive_search_recommendations(self, query: str, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get adaptive search recommendations based on learned patterns.
        
        Args:
            query: Search query
            query_analysis: Query analysis results
            
        Returns:
            List of adaptive recommendations
        """
        recommendations = []
        
        domain = query_analysis.get("domain_analysis", {}).get("primary_domain", "general")
        intent = query_analysis.get("detected_intent", "unknown")
        complexity = query_analysis.get("complexity_analysis", {}).get("complexity_level", "moderate")
        
        # Check if we have successful patterns for similar queries
        successful_patterns = self._find_successful_patterns(domain, intent, complexity)
        
        for pattern in successful_patterns:
            recommendations.append({
                "type": "strategy_recommendation",
                "strategy": pattern["strategy"],
                "description": pattern["description"],
                "confidence": pattern["success_rate"],
                "based_on": f"{pattern['usage_count']} similar successful searches"
            })
        
        # Domain-specific recommendations
        if domain in self.domain_preferences and self.domain_preferences[domain] > 0.7:
            recommendations.append({
                "type": "domain_optimization",
                "description": f"Consider adding {domain.replace('_', ' ')} specific terminology",
                "confidence": self.domain_preferences[domain],
                "based_on": "Your search history shows strong preference for this domain"
            })
        
        # Query refinement recommendations
        refinement_suggestions = self._get_refinement_suggestions(query, intent)
        recommendations.extend(refinement_suggestions)
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def analyze_search_patterns(self) -> Dict[str, Any]:
        """Analyze overall search patterns and provide insights.
        
        Returns:
            Dictionary with pattern analysis insights
        """
        if not self.search_history:
            return {"insights": [], "patterns": {}, "recommendations": []}
        
        # Analyze recent search history (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_searches = [
            record for record in self.search_history 
            if datetime.fromisoformat(record["timestamp"]) > recent_cutoff
        ]
        
        if not recent_searches:
            recent_searches = self.search_history[-100:]  # Fallback to last 100 searches
        
        # Pattern analysis
        patterns = {
            "most_common_intents": Counter(record["intent"] for record in recent_searches).most_common(5),
            "most_common_domains": Counter(record["domain"] for record in recent_searches).most_common(5),
            "complexity_distribution": Counter(record["complexity"] for record in recent_searches),
            "success_rate": sum(1 for r in recent_searches if r["success"]) / len(recent_searches),
            "average_response_time": sum(r["response_time"] for r in recent_searches) / len(recent_searches)
        }
        
        # Generate insights
        insights = []
        
        # Success rate insights
        if patterns["success_rate"] < 0.8:
            insights.append({
                "type": "success_rate",
                "message": f"Search success rate is {patterns['success_rate']:.1%}. Consider refining query strategies.",
                "severity": "medium"
            })
        
        # Performance insights
        if patterns["average_response_time"] > 2000:
            insights.append({
                "type": "performance",
                "message": f"Average search time is {patterns['average_response_time']:.0f}ms. Consider optimizing search configuration.",
                "severity": "low"
            })
        
        # Domain focus insights
        top_domain = patterns["most_common_domains"][0] if patterns["most_common_domains"] else None
        if top_domain and top_domain[1] > len(recent_searches) * 0.5:
            insights.append({
                "type": "domain_focus",
                "message": f"You frequently search in {top_domain[0].replace('_', ' ')} domain. Consider domain-specific search optimization.",
                "severity": "info"
            })
        
        # Generate recommendations
        recommendations = self._generate_usage_recommendations(patterns, recent_searches)
        
        return {
            "insights": insights,
            "patterns": patterns,
            "recommendations": recommendations,
            "analysis_period": f"Last {len(recent_searches)} searches",
            "data_quality": self._assess_data_quality()
        }
    
    def get_search_quality_metrics(self) -> Dict[str, Any]:
        """Get comprehensive search quality metrics.
        
        Returns:
            Dictionary with quality metrics
        """
        if not self.search_history:
            return {}
        
        recent_searches = self.search_history[-500:]  # Last 500 searches
        
        metrics = {
            "total_searches": len(recent_searches),
            "success_rate": sum(1 for r in recent_searches if r["success"]) / len(recent_searches),
            "average_response_time": sum(r["response_time"] for r in recent_searches) / len(recent_searches),
            "average_results_per_search": sum(r["results_count"] for r in recent_searches) / len(recent_searches)
        }
        
        # Intent distribution
        intent_dist = Counter(r["intent"] for r in recent_searches)
        metrics["intent_distribution"] = dict(intent_dist.most_common())
        
        # Domain distribution
        domain_dist = Counter(r["domain"] for r in recent_searches)
        metrics["domain_distribution"] = dict(domain_dist.most_common())
        
        # Performance by search mode
        mode_performance = defaultdict(list)
        for record in recent_searches:
            mode_performance[record["search_mode"]].append(record["response_time"])
        
        metrics["performance_by_mode"] = {
            mode: {
                "average_time": sum(times) / len(times),
                "count": len(times)
            }
            for mode, times in mode_performance.items()
        }
        
        return metrics
    
    def _update_patterns(self, search_record: Dict[str, Any], query_analysis: Dict[str, Any]) -> None:
        """Update learned patterns based on search record."""
        query = search_record["query"]
        domain = search_record["domain"]
        intent = search_record["intent"]
        
        # Update query patterns
        query_key = query.lower()[:50]  # Limit key length
        if query_key not in self.query_patterns:
            self.query_patterns[query_key] = {
                "frequency": 0,
                "success_count": 0,
                "total_searches": 0,
                "domains": Counter(),
                "intents": Counter()
            }
        
        pattern = self.query_patterns[query_key]
        pattern["frequency"] += 1
        pattern["total_searches"] += 1
        if search_record["success"]:
            pattern["success_count"] += 1
        pattern["domains"][domain] += 1
        pattern["intents"][intent] += 1
        
        # Update domain preferences
        if search_record["success"]:
            self.domain_preferences[domain] += 1
        
        # Update successful strategies
        if search_record["success"] and search_record["results_count"] > 0:
            strategy_key = f"{intent}_{domain}_{search_record['complexity']}"
            search_mode = search_record["search_mode"]
            if search_mode not in self.successful_strategies[strategy_key]:
                self.successful_strategies[strategy_key].append(search_mode)
    
    def _update_performance_metrics(self, search_record: Dict[str, Any]) -> None:
        """Update performance metrics."""
        self.performance_metrics["response_times"].append(search_record["response_time"])
        self.performance_metrics["success_rates"].append(1.0 if search_record["success"] else 0.0)
        
        # Keep only recent metrics
        max_metrics = 1000
        for metric_list in self.performance_metrics.values():
            if len(metric_list) > max_metrics:
                metric_list[:] = metric_list[-max_metrics:]
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries."""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_popular_queries(self, max_suggestions: int) -> List[Dict[str, Any]]:
        """Get popular queries for autocomplete."""
        if not self.query_patterns:
            return []
        
        # Sort patterns by frequency and success rate
        popular = []
        for query, pattern in self.query_patterns.items():
            success_rate = pattern["success_count"] / max(pattern["total_searches"], 1)
            score = pattern["frequency"] * success_rate
            
            popular.append({
                "query": query,
                "score": score,
                "success_rate": success_rate,
                "frequency": pattern["frequency"]
            })
        
        popular.sort(key=lambda x: x["score"], reverse=True)
        
        return [{
            "query": item["query"],
            "confidence": min(item["score"] / 10, 1.0),  # Normalize score
            "domain": "general",
            "intent": "general",
            "suggestion_type": "popular_query"
        } for item in popular[:max_suggestions]]
    
    def _generate_pattern_suggestions(self, partial_query: str, max_suggestions: int) -> List[Dict[str, Any]]:
        """Generate suggestions based on learned patterns."""
        suggestions = []
        
        # Find patterns that start with or contain the partial query
        for query, pattern in self.query_patterns.items():
            if partial_query in query and pattern["frequency"] > 1:
                success_rate = pattern["success_count"] / max(pattern["total_searches"], 1)
                
                suggestions.append({
                    "query": query,
                    "confidence": round(success_rate, 3),
                    "domain": pattern["domains"].most_common(1)[0][0] if pattern["domains"] else "general",
                    "intent": pattern["intents"].most_common(1)[0][0] if pattern["intents"] else "general",
                    "suggestion_type": "pattern_based"
                })
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return suggestions[:max_suggestions]
    
    def _find_successful_patterns(self, domain: str, intent: str, complexity: str) -> List[Dict[str, Any]]:
        """Find successful search patterns for given context."""
        patterns = []
        
        strategy_key = f"{intent}_{domain}_{complexity}"
        if strategy_key in self.successful_strategies:
            strategies = self.successful_strategies[strategy_key]
            
            for strategy in strategies:
                # Count successful uses of this strategy
                usage_count = sum(1 for record in self.search_history[-500:] 
                                if record["intent"] == intent 
                                and record["domain"] == domain
                                and record["complexity"] == complexity
                                and record["search_mode"] == strategy
                                and record["success"])
                
                total_count = sum(1 for record in self.search_history[-500:] 
                                if record["intent"] == intent 
                                and record["domain"] == domain
                                and record["complexity"] == complexity
                                and record["search_mode"] == strategy)
                
                if total_count > 0:
                    success_rate = usage_count / total_count
                    patterns.append({
                        "strategy": strategy,
                        "description": f"Use {strategy.replace('_', ' ')} search approach",
                        "success_rate": round(success_rate, 3),
                        "usage_count": usage_count
                    })
        
        return sorted(patterns, key=lambda x: x["success_rate"], reverse=True)[:3]
    
    def _get_refinement_suggestions(self, query: str, intent: str) -> List[Dict[str, Any]]:
        """Get query refinement suggestions based on patterns."""
        suggestions = []
        
        # Look for successful refinement patterns
        for path in self.query_refinement_paths[-50:]:  # Recent refinement paths
            if len(path) >= 2 and query.lower() in [q.lower() for q in path[:-1]]:
                # Found a refinement path that includes this query
                final_query = path[-1]
                suggestions.append({
                    "type": "refinement_suggestion",
                    "description": f"Consider refining to: '{final_query}'",
                    "confidence": 0.7,
                    "based_on": "Similar successful refinement patterns"
                })
        
        return suggestions[:2]  # Limit refinement suggestions
    
    def _generate_usage_recommendations(self, patterns: Dict[str, Any], recent_searches: List[Dict]) -> List[Dict[str, Any]]:
        """Generate recommendations based on usage patterns."""
        recommendations = []
        
        # Success rate recommendations
        if patterns["success_rate"] < 0.7:
            recommendations.append({
                "type": "query_improvement",
                "description": "Consider using more specific terminology in your queries",
                "priority": "high",
                "rationale": "Low success rate indicates queries may be too vague or broad"
            })
        
        # Domain specialization recommendations
        top_domains = patterns["most_common_domains"][:2]
        for domain, count in top_domains:
            if count > len(recent_searches) * 0.3:  # If domain represents >30% of searches
                recommendations.append({
                    "type": "domain_specialization",
                    "description": f"Consider configuring domain-specific search settings for {domain.replace('_', ' ')}",
                    "priority": "medium",
                    "rationale": f"Frequent searches in {domain.replace('_', ' ')} domain detected"
                })
        
        return recommendations
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess the quality of collected intelligence data."""
        if not self.search_history:
            return {"quality_score": 0.0, "issues": ["No search data available"]}
        
        issues = []
        quality_factors = []
        
        # Check data volume
        if len(self.search_history) < 50:
            issues.append("Insufficient search history for reliable patterns")
            quality_factors.append(0.3)
        else:
            quality_factors.append(1.0)
        
        # Check data recency
        if self.search_history:
            latest_search = datetime.fromisoformat(self.search_history[-1]["timestamp"])
            days_since_latest = (datetime.now() - latest_search).days
            
            if days_since_latest > 30:
                issues.append("Search data is outdated")
                quality_factors.append(0.5)
            else:
                quality_factors.append(1.0)
        
        # Check data diversity
        unique_queries = len(set(record["query"] for record in self.search_history[-100:]))
        recent_count = min(100, len(self.search_history))
        diversity_ratio = unique_queries / recent_count if recent_count > 0 else 0
        
        if diversity_ratio < 0.5:
            issues.append("Limited query diversity may affect recommendation quality")
            quality_factors.append(0.7)
        else:
            quality_factors.append(1.0)
        
        quality_score = sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
        
        return {
            "quality_score": round(quality_score, 2),
            "issues": issues,
            "data_points": len(self.search_history),
            "unique_patterns": len(self.query_patterns)
        }
    
    def _save_intelligence_data(self) -> None:
        """Save intelligence data to storage."""
        if not self.storage_path:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "search_history": self.search_history,
                "query_patterns": {k: dict(v) for k, v in self.query_patterns.items()},
                "domain_preferences": dict(self.domain_preferences),
                "successful_strategies": {k: list(v) for k, v in self.successful_strategies.items()},
                "query_refinement_paths": self.query_refinement_paths,
                "performance_metrics": {k: list(v) for k, v in self.performance_metrics.items()},
                "saved_at": datetime.now().isoformat()
            }
            
            # Convert Counter objects to regular dicts for JSON serialization
            for pattern in data["query_patterns"].values():
                if "domains" in pattern:
                    pattern["domains"] = dict(pattern["domains"])
                if "intents" in pattern:
                    pattern["intents"] = dict(pattern["intents"])
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved search intelligence data to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save intelligence data: {e}")
    
    def _load_intelligence_data(self) -> None:
        """Load intelligence data from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.search_history = data.get("search_history", [])
            
            # Restore query patterns with Counter objects
            patterns_data = data.get("query_patterns", {})
            self.query_patterns = defaultdict(dict)
            for k, v in patterns_data.items():
                self.query_patterns[k] = v.copy()
                if "domains" in v:
                    self.query_patterns[k]["domains"] = Counter(v["domains"])
                if "intents" in v:
                    self.query_patterns[k]["intents"] = Counter(v["intents"])
            
            self.domain_preferences = defaultdict(float, data.get("domain_preferences", {}))
            self.successful_strategies = defaultdict(list, data.get("successful_strategies", {}))
            self.query_refinement_paths = data.get("query_refinement_paths", [])
            
            # Restore performance metrics
            metrics_data = data.get("performance_metrics", {})
            self.performance_metrics = defaultdict(list)
            for k, v in metrics_data.items():
                self.performance_metrics[k] = list(v)
            
            logger.debug(f"Loaded search intelligence data from {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to load intelligence data: {e}")
    
    def save(self) -> None:
        """Explicitly save intelligence data."""
        self._save_intelligence_data()
    
    def clear_data(self) -> None:
        """Clear all intelligence data."""
        with self._lock:
            self.search_history.clear()
            self.query_patterns.clear()
            self.domain_preferences.clear()
            self.successful_strategies.clear()
            self.query_refinement_paths.clear()
            self.performance_metrics.clear()
        
        logger.info("Cleared all search intelligence data")