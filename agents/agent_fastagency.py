"""
FastAgency agent - lightweight, tool-focused, ideal for local development.
Simple and efficient agent framework.
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class FastAgencyAgent:
    """FastAgency agent - lightweight and tool-focused."""
    
    def __init__(self, model_name: str = "llama2", use_local: bool = True, use_ollama: bool = True):
        self.name = "fastagency"
        self.decision_history = []
        self.model_name = model_name
        self.use_local = use_local
        self.use_ollama = use_ollama
        self._llm = None
        self._model_available = False
        
        # Performance tracking
        self.platform_performance = {}
        self.platform_stats = {}
        
        self._initialize_fastagency()
    
    def _initialize_fastagency(self):
        """Initialize FastAgency with LLM."""
        try:
            # FastAgency is lightweight - we'll use a simple tool-based approach
            # For now, we'll use direct Ollama calls with tool functions
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
                    
                    self._llm = "ollama"
                    self._model_available = True
                    logger.info(f"[OK] FastAgency initialized with Ollama model: {self.model_name}")
                    
                except Exception as e:
                    logger.warning(f"FastAgency - Ollama setup failed: {e}")
                    self._llm = "simple"
            else:
                self._llm = "simple"
                
        except Exception as e:
            logger.warning(f"FastAgency initialization failed: {e}")
            self._llm = "simple"
    
    def _call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool function (FastAgency-style lightweight tool execution)."""
        if tool_name == "get_performance_history":
            platform = kwargs.get('platform')
            data_source = kwargs.get('data_source')
            experiment_type = kwargs.get('experiment_type')
            
            matching_latencies = []
            for key, latencies in self.platform_performance.items():
                plat, ds, exp = key
                if plat != platform:
                    continue
                if data_source and ds != data_source:
                    continue
                if experiment_type and exp != experiment_type:
                    continue
                matching_latencies.extend(latencies)
            
            if matching_latencies:
                return {
                    'platform': platform,
                    'mean': np.mean(matching_latencies),
                    'std': np.std(matching_latencies),
                    'count': len(matching_latencies)
                }
            return {'platform': platform, 'mean': None, 'std': None, 'count': 0}
        
        elif tool_name == "compare_platforms":
            platforms = kwargs.get('platforms', [])
            comparisons = []
            
            for platform in platforms:
                matching_latencies = []
                for key, latencies in self.platform_performance.items():
                    if key[0] == platform:
                        matching_latencies.extend(latencies)
                
                if matching_latencies:
                    comparisons.append({
                        'platform': platform,
                        'mean': np.mean(matching_latencies),
                        'std': np.std(matching_latencies),
                        'count': len(matching_latencies)
                    })
            
            comparisons.sort(key=lambda x: x['mean'] if x['mean'] is not None else float('inf'))
            return comparisons
        
        elif tool_name == "get_platform_specs":
            platform = kwargs.get('platform')
            specs = {
                'pandas': 'In-memory DataFrame, general-purpose',
                'duckdb': 'Analytical SQL engine, fast aggregations',
                'polars': 'Fast DataFrame library, parallel execution',
                'sqlite': 'OLTP SQL database, ACID compliance',
                'faiss': 'Vector similarity search, GPU support',
                'annoy': 'Approximate nearest neighbors',
                'baseline': 'Naive Python implementation'
            }
            return specs.get(platform, f"{platform}: Unknown")
        
        return None
    
    def _build_fastagency_prompt(self, data_source: str, experiment_type: str,
                                 available_platforms: List[str], context: Dict[str, Any]) -> str:
        """Build prompt with tool results (FastAgency lightweight approach)."""
        # Call tools directly (lightweight, no heavy framework)
        comparisons = self._call_tool("compare_platforms", platforms=available_platforms)
        
        prompt = f"""Select the best platform for this workload.

TASK:
- Data Source: {data_source}
- Experiment Type: {experiment_type}
- Available Platforms: {', '.join(available_platforms)}

PERFORMANCE DATA (via tools):
"""
        
        if comparisons:
            for comp in comparisons:
                if comp['mean'] is not None:
                    prompt += f"  {comp['platform']}: {comp['mean']:.2f}ms avg (n={comp['count']})\n"
        else:
            prompt += "  No historical data available.\n"
        
        prompt += "\nPLATFORM SPECS:\n"
        for platform in available_platforms:
            specs = self._call_tool("get_platform_specs", platform=platform)
            prompt += f"  {platform}: {specs}\n"
        
        prompt += f"\nSelect the best platform from: {', '.join(available_platforms)}"
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM (FastAgency lightweight approach)."""
        if self._llm == "ollama" and self._model_available:
            try:
                import ollama
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.3,
                        'num_predict': 50
                    }
                )
                return response.get('response', '')
            except Exception as e:
                logger.error(f"FastAgency - Ollama call failed: {e}")
                return ""
        else:
            return self._simple_reasoning(prompt)
    
    def _simple_reasoning(self, prompt: str) -> str:
        """Simple rule-based fallback."""
        if 'vector' in prompt.lower() and any(p in prompt for p in ['annoy', 'faiss']):
            return 'annoy' if 'annoy' in prompt else 'faiss'
        elif 'aggregate' in prompt.lower():
            if 'duckdb' in prompt:
                return 'duckdb'
            elif 'polars' in prompt:
                return 'polars'
        return 'pandas'
    
    def _extract_platform(self, response: str, available_platforms: List[str]) -> Optional[str]:
        """Extract platform name from response."""
        response_lower = response.lower()
        for platform in available_platforms:
            if platform.lower() in response_lower:
                return platform
        return available_platforms[0] if available_platforms else None
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str], context: Dict[str, Any] = None) -> str:
        """Select platform using FastAgency lightweight approach."""
        if not available_platforms:
            return None
        
        context = context or {}
        
        # Build prompt with tool results
        prompt = self._build_fastagency_prompt(data_source, experiment_type, available_platforms, context)
        
        # Call LLM
        response = self._call_llm(prompt)
        
        # Extract platform
        selected = self._extract_platform(response, available_platforms)
        if not selected:
            selected = available_platforms[0]
        
        decision = {
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': selected,
            'available_platforms': available_platforms,
            'reasoning': f'FastAgency: {response[:200]}',
            'llm_model': self.model_name,
            'framework': 'FastAgency',
            'lightweight': True,
            'tools_used': ['get_performance_history', 'compare_platforms', 'get_platform_specs']
        }
        
        self.decision_history.append(decision)
        return selected
    
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

