"""
LangGraph agent for platform selection with stateful agentic pipelines.
Reliable state management for complex decision workflows.
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class LangGraphAgent:
    """LangGraph agent with stateful pipelines for reliable decision-making."""
    
    def __init__(self, model_name: str = "llama2", use_local: bool = True, use_ollama: bool = True):
        self.name = "langgraph"
        self.decision_history = []
        self.model_name = model_name
        self.use_local = use_local
        self.use_ollama = use_ollama
        self._llm = None
        self._model_available = False
        self._state = {}  # Stateful pipeline state
        
        # Performance tracking
        self.platform_performance = {}
        self.platform_stats = {}
        
        self._initialize_langgraph()
    
    def _initialize_langgraph(self):
        """Initialize LangGraph with stateful agent."""
        try:
            try:
                from langchain_community.llms import Ollama
                from langgraph.graph import StateGraph, END
                from typing import TypedDict
                
                if self.use_ollama:
                    try:
                        import ollama
                        models_response = ollama.list()
                        
                        models_list = []
                        if hasattr(models_response, 'models'):
                            models_list = models_response.models
                        elif isinstance(models_response, dict):
                            models_list = models_response.get('models', [])
                        
                        models = []
                        for m in models_list:
                            if hasattr(m, 'model'):
                                model_name = m.model
                            elif isinstance(m, dict):
                                model_name = m.get('name') or m.get('model') or str(m)
                            else:
                                model_name = str(m)
                            if model_name:
                                models.append(model_name)
                        
                        requested_base = self.model_name.split(':')[0].lower()
                        model_found = False
                        selected_model = None
                        
                        for model in models:
                            model_base = model.split(':')[0].lower()
                            if requested_base == model_base or requested_base in model_base:
                                model_found = True
                                selected_model = model
                                break
                        
                        if model_found:
                            self.model_name = selected_model
                        elif models:
                            self.model_name = models[0]
                        else:
                            raise ValueError("No Ollama models available")
                        
                        self._llm = Ollama(model=self.model_name)
                        self._model_available = True
                        logger.info(f"[OK] LangGraph initialized with Ollama model: {self.model_name}")
                        
                        # Create stateful graph
                        self._graph = self._create_decision_graph()
                        
                    except Exception as e:
                        logger.warning(f"LangGraph - Ollama setup failed: {e}")
                        self._llm = "simple"
                else:
                    self._llm = "simple"
                    
            except ImportError:
                logger.warning("LangGraph packages not installed. Install with: pip install langgraph langchain-community")
                self._llm = "simple"
                
        except Exception as e:
            logger.warning(f"LangGraph initialization failed: {e}")
            self._llm = "simple"
    
    def _create_decision_graph(self):
        """Create stateful decision graph."""
        try:
            from langgraph.graph import StateGraph, END
            from typing import TypedDict
            
            class DecisionState(TypedDict):
                data_source: str
                experiment_type: str
                available_platforms: List[str]
                performance_data: Dict[str, Any]
                selected_platform: Optional[str]
                reasoning: str
            
            def gather_performance_data(state: DecisionState) -> DecisionState:
                """Gather historical performance data."""
                performance_data = {}
                for platform in state['available_platforms']:
                    matching_latencies = []
                    for key, latencies in self.platform_performance.items():
                        if key[0] == platform:
                            matching_latencies.extend(latencies)
                    
                    if matching_latencies:
                        performance_data[platform] = {
                            'mean': np.mean(matching_latencies),
                            'std': np.std(matching_latencies),
                            'count': len(matching_latencies)
                        }
                    else:
                        performance_data[platform] = {'mean': None, 'std': None, 'count': 0}
                
                state['performance_data'] = performance_data
                return state
            
            def analyze_and_select(state: DecisionState) -> DecisionState:
                """Analyze data and select platform."""
                if self._llm != "simple" and self._model_available:
                    # Use LLM for decision
                    prompt = f"""Select best platform for:
- Data Source: {state['data_source']}
- Experiment Type: {state['experiment_type']}
- Available: {', '.join(state['available_platforms'])}

Performance data:
"""
                    for platform, perf in state['performance_data'].items():
                        if perf['mean'] is not None:
                            prompt += f"  {platform}: {perf['mean']:.2f}ms avg (n={perf['count']})\n"
                    
                    prompt += f"\nSelect from: {', '.join(state['available_platforms'])}"
                    
                    try:
                        response = self._llm(prompt)
                        selected = self._extract_platform(str(response), state['available_platforms'])
                        reasoning = f"LangGraph LLM: {str(response)[:200]}"
                    except:
                        selected = self._select_best_from_data(state)
                        reasoning = "LangGraph: Selected based on performance data"
                else:
                    selected = self._select_best_from_data(state)
                    reasoning = "LangGraph: Selected based on performance data"
                
                state['selected_platform'] = selected
                state['reasoning'] = reasoning
                return state
            
            def _select_best_from_data(self, state: DecisionState) -> str:
                """Select best platform from performance data."""
                best_platform = None
                best_mean = float('inf')
                
                for platform, perf in state['performance_data'].items():
                    if perf['mean'] is not None and perf['mean'] < best_mean:
                        best_mean = perf['mean']
                        best_platform = platform
                
                return best_platform or state['available_platforms'][0]
            
            # Build graph
            workflow = StateGraph(DecisionState)
            workflow.add_node("gather_data", gather_performance_data)
            workflow.add_node("analyze", analyze_and_select)
            workflow.set_entry_point("gather_data")
            workflow.add_edge("gather_data", "analyze")
            workflow.add_edge("analyze", END)
            
            return workflow.compile()
            
        except Exception as e:
            logger.warning(f"Failed to create LangGraph: {e}")
            return None
    
    def _extract_platform(self, response: str, available_platforms: List[str]) -> Optional[str]:
        """Extract platform name from response."""
        response_lower = response.lower()
        for platform in available_platforms:
            if platform.lower() in response_lower:
                return platform
        return available_platforms[0] if available_platforms else None
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str], context: Dict[str, Any] = None) -> str:
        """Select platform using LangGraph stateful pipeline."""
        if not available_platforms:
            return None
        
        context = context or {}
        
        if hasattr(self, '_graph') and self._graph is not None:
            try:
                from typing import TypedDict
                
                initial_state = {
                    'data_source': data_source,
                    'experiment_type': experiment_type,
                    'available_platforms': available_platforms,
                    'performance_data': {},
                    'selected_platform': None,
                    'reasoning': ''
                }
                
                result = self._graph.invoke(initial_state)
                selected = result.get('selected_platform') or available_platforms[0]
                reasoning = result.get('reasoning', 'LangGraph stateful decision')
                
            except Exception as e:
                logger.error(f"LangGraph execution failed: {e}")
                selected = self._simple_reasoning(data_source, experiment_type, available_platforms)
                reasoning = f"LangGraph fallback: {e}"
        else:
            selected = self._simple_reasoning(data_source, experiment_type, available_platforms)
            reasoning = "LangGraph: Simple reasoning fallback"
        
        decision = {
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': selected,
            'available_platforms': available_platforms,
            'reasoning': reasoning,
            'llm_model': self.model_name,
            'framework': 'LangGraph',
            'stateful': True
        }
        
        self.decision_history.append(decision)
        return selected
    
    def _simple_reasoning(self, data_source: str, experiment_type: str, available_platforms: List[str]) -> str:
        """Simple rule-based fallback."""
        if 'vector' in experiment_type.lower() and any(p in available_platforms for p in ['annoy', 'faiss']):
            return 'annoy' if 'annoy' in available_platforms else 'faiss'
        elif 'aggregate' in experiment_type.lower():
            if 'duckdb' in available_platforms:
                return 'duckdb'
            elif 'polars' in available_platforms:
                return 'polars'
        return available_platforms[0] if available_platforms else None
    
    def update(self, data_source: str, experiment_type: str, platform: str, 
              metrics: Dict[str, Any]):
        """Update performance history."""
        latency = metrics.get('latency_ms', 0)
        
        key = (platform, data_source, experiment_type)
        if key not in self.platform_performance:
            self.platform_performance[key] = []
        
        self.platform_performance[key].append(latency)
        
        if platform not in self.platform_stats:
            self.platform_stats[platform] = {'latencies': [], 'mean': 0, 'std': 0, 'count': 0}
        
        self.platform_stats[platform]['latencies'].append(latency)
        self.platform_stats[platform]['count'] = len(self.platform_stats[platform]['latencies'])
        self.platform_stats[platform]['mean'] = np.mean(self.platform_stats[platform]['latencies'])
        self.platform_stats[platform]['std'] = np.std(self.platform_stats[platform]['latencies'])
    
    def get_decision_reasoning(self) -> str:
        """Get the reasoning for the last decision."""
        if self.decision_history:
            last_decision = self.decision_history[-1]
            return last_decision.get('reasoning', 'No reasoning available')
        return 'No decisions made yet'
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Return decision history."""
        return self.decision_history

