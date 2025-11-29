"""
Rule-based agent using heuristics.
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class RuleBasedAgent:
    """Agent that uses simple heuristics to select platforms."""
    
    def __init__(self):
        self.name = "rule_based"
        self.decision_history = []
    
    def select_platform(self, data_source: str, experiment_type: str, 
                       available_platforms: List[str], 
                       context: Dict[str, Any] = None) -> str:
        """Select platform based on rules."""
        if context is None:
            context = {}
        
        reasoning = []
        
        # Rule 1: Vector data → FAISS or Annoy
        if data_source == 'vectors':
            if 'faiss' in available_platforms:
                reasoning.append("Vector data detected → using FAISS")
                decision = 'faiss'
            elif 'annoy' in available_platforms:
                reasoning.append("Vector data detected → using Annoy")
                decision = 'annoy'
            else:
                reasoning.append("Vector data but no vector platform → using pandas")
                decision = 'pandas'
        
        # Rule 2: Text similarity → baseline or pandas
        elif experiment_type == 'text_similarity':
            if 'baseline' in available_platforms:
                reasoning.append("Text similarity → using baseline")
                decision = 'baseline'
            else:
                reasoning.append("Text similarity → using pandas")
                decision = 'pandas'
        
        # Rule 3: Large tabular data → DuckDB
        elif data_source.startswith('tabular') and context.get('size', 0) > 100000:
            if 'duckdb' in available_platforms:
                reasoning.append("Large tabular data → using DuckDB")
                decision = 'duckdb'
            else:
                reasoning.append("Large tabular data → using pandas")
                decision = 'pandas'
        
        # Rule 4: Aggregation queries → DuckDB
        elif experiment_type in ['aggregate', 'time_window']:
            if 'duckdb' in available_platforms:
                reasoning.append("Aggregation query → using DuckDB")
                decision = 'duckdb'
            else:
                reasoning.append("Aggregation query → using pandas")
                decision = 'pandas'
        
        # Rule 5: Join operations → DuckDB or SQLite
        elif experiment_type == 'join':
            if 'duckdb' in available_platforms:
                reasoning.append("Join operation → using DuckDB")
                decision = 'duckdb'
            elif 'sqlite' in available_platforms:
                reasoning.append("Join operation → using SQLite")
                decision = 'sqlite'
            else:
                reasoning.append("Join operation → using pandas")
                decision = 'pandas'
        
        # Default: pandas
        else:
            reasoning.append("Default rule → using pandas")
            decision = 'pandas'
        
        # Ensure decision is in available platforms
        if decision not in available_platforms:
            decision = available_platforms[0] if available_platforms else 'pandas'
            reasoning.append(f"Selected platform not available, using {decision}")
        
        decision_reasoning = " | ".join(reasoning)
        
        self.decision_history.append({
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': decision,
            'reasoning': decision_reasoning,
            'context': context
        })
        
        return decision
    
    def get_decision_reasoning(self) -> str:
        """Get reasoning for last decision."""
        if self.decision_history:
            return self.decision_history[-1].get('reasoning', '')
        return ""
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get decision history."""
        return self.decision_history

