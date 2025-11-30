"""
LangChain agent for platform selection with tool support.
Most flexible framework for building LLM applications.
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class LangChainAgent:
    """LangChain agent with tool support for enhanced decision-making."""
    
    def __init__(self, model_name: str = "llama2", use_local: bool = True, use_ollama: bool = True):
        self.name = "langchain"
        self.decision_history = []
        self.model_name = model_name
        self.use_local = use_local
        self.use_ollama = use_ollama
        self._llm = None
        self._model_available = False
        
        # Performance tracking for tools
        self.platform_performance = {}  # (platform, data_source, experiment_type) -> [latencies]
        self.platform_stats = {}  # platform -> {mean, std, count}
        
        self._initialize_langchain()
    
    def _initialize_langchain(self):
        """Initialize LangChain with LLM and tools."""
        try:
            # Try to import LangChain
            try:
                from langchain_community.llms import Ollama
                from langchain.tools import Tool
                from langchain.agents import initialize_agent, AgentType
                
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
                        
                        # Initialize Ollama LLM for LangChain
                        self._llm = Ollama(model=self.model_name)
                        self._model_available = True
                        logger.info(f"[OK] LangChain initialized with Ollama model: {self.model_name}")
                        
                        # Create tools
                        self._tools = self._create_tools()
                        
                        # Initialize agent
                        self._agent = initialize_agent(
                            tools=self._tools,
                            llm=self._llm,
                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            verbose=False
                        )
                        
                    except Exception as e:
                        logger.warning(f"LangChain - Ollama setup failed: {e}")
                        self._llm = "simple"
                else:
                    self._llm = "simple"
                    
            except ImportError:
                logger.warning("LangChain packages not installed. Install with: pip install langchain langchain-community")
                self._llm = "simple"
                
        except Exception as e:
            logger.warning(f"LangChain initialization failed: {e}")
            self._llm = "simple"
    
    def _create_tools(self) -> List:
        """Create LangChain tools for platform selection."""
        from langchain.tools import Tool
        
        def get_performance_history(platform: str, data_source: str = "", experiment_type: str = "") -> str:
            """Get historical performance for a platform."""
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
                return f"{platform}: mean={np.mean(matching_latencies):.2f}ms, std={np.std(matching_latencies):.2f}ms, n={len(matching_latencies)}"
            return f"{platform}: No historical data"
        
        def compare_platforms(platforms: str) -> str:
            """Compare historical performance of platforms. Input: comma-separated platform names."""
            platform_list = [p.strip() for p in platforms.split(',')]
            comparisons = []
            
            for platform in platform_list:
                matching_latencies = []
                for key, latencies in self.platform_performance.items():
                    if key[0] == platform:
                        matching_latencies.extend(latencies)
                
                if matching_latencies:
                    comparisons.append(f"{platform}: {np.mean(matching_latencies):.2f}ms avg")
            
            comparisons.sort()
            return "\n".join(comparisons) if comparisons else "No historical data"
        
        def get_platform_specs(platform: str) -> str:
            """Get platform specifications."""
            specs = {
                'pandas': 'In-memory DataFrame, general-purpose, easy to use, best for small-medium data',
                'duckdb': 'Analytical SQL engine, columnar storage, fast aggregations, best for OLAP',
                'polars': 'Fast DataFrame library, parallel execution, lazy evaluation, best for large datasets',
                'sqlite': 'OLTP SQL database, ACID compliance, structured queries, best for transactional workloads',
                'faiss': 'Vector similarity search, GPU support, best for vector embeddings',
                'annoy': 'Approximate nearest neighbors, fast indexing, best for large-scale similarity',
                'baseline': 'Naive Python implementation, baseline comparison only'
            }
            return specs.get(platform, f"{platform}: Unknown platform")
        
        tools = [
            Tool(
                name="get_performance_history",
                func=get_performance_history,
                description="Get historical performance metrics for a platform. Input: platform name, optional data_source and experiment_type"
            ),
            Tool(
                name="compare_platforms",
                func=compare_platforms,
                description="Compare historical performance of multiple platforms. Input: comma-separated platform names"
            ),
            Tool(
                name="get_platform_specs",
                func=get_platform_specs,
                description="Get platform specifications and characteristics. Input: platform name"
            )
        ]
        
        return tools
    
    def _build_langchain_prompt(self, data_source: str, experiment_type: str,
                                available_platforms: List[str], context: Dict[str, Any]) -> str:
        """Build prompt for LangChain agent."""
        prompt = f"""You are an expert database system selector. Select the best platform for this workload.

TASK:
- Data Source: {data_source}
- Experiment Type: {experiment_type}
- Available Platforms: {', '.join(available_platforms)}

You have access to tools:
1. get_performance_history - Get historical performance for a platform
2. compare_platforms - Compare multiple platforms
3. get_platform_specs - Get platform specifications

Use these tools to make an informed decision. Select the best platform from: {', '.join(available_platforms)}
Respond with just the platform name."""
        
        return prompt
    
    def _call_langchain_agent(self, prompt: str) -> str:
        """Call LangChain agent with tools."""
        if hasattr(self, '_agent') and self._model_available:
            try:
                response = self._agent.run(prompt)
                return str(response)
            except Exception as e:
                logger.error(f"LangChain agent call failed: {e}")
                return self._simple_reasoning(prompt)
        else:
            return self._simple_reasoning(prompt)
    
    def _simple_reasoning(self, prompt: str) -> str:
        """Simple rule-based fallback."""
        if 'vector' in prompt.lower() and any(p in prompt for p in ['annoy', 'faiss']):
            return 'annoy' if 'annoy' in prompt else 'faiss'
        elif 'aggregate' in prompt.lower() or 'group' in prompt.lower():
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
        """Select platform using LangChain agent with tools."""
        if not available_platforms:
            return None
        
        context = context or {}
        
        # Build prompt
        prompt = self._build_langchain_prompt(data_source, experiment_type, available_platforms, context)
        
        # Call LangChain agent
        response = self._call_langchain_agent(prompt)
        
        # Extract platform
        selected = self._extract_platform(response, available_platforms)
        if not selected:
            selected = available_platforms[0]
        
        decision = {
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': selected,
            'available_platforms': available_platforms,
            'reasoning': f'LangChain agent: {response[:200]}',
            'llm_model': self.model_name,
            'framework': 'LangChain',
            'tools_used': ['get_performance_history', 'compare_platforms', 'get_platform_specs']
        }
        
        self.decision_history.append(decision)
        return selected
    
    def update(self, data_source: str, experiment_type: str, platform: str, 
              metrics: Dict[str, Any]):
        """Update performance history for tools."""
        latency = metrics.get('latency_ms', 0)
        
        key = (platform, data_source, experiment_type)
        if key not in self.platform_performance:
            self.platform_performance[key] = []
        
        self.platform_performance[key].append(latency)
        
        # Update platform-level stats
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

