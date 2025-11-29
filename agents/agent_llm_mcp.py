"""
LLM agent with Model Context Protocol (MCP) integration.
Allows LLM to access tools and performance history for better decisions.
"""
import logging
from typing import Dict, Any, List, Optional
import json
import numpy as np

logger = logging.getLogger(__name__)

class LLMAgentMCP:
    """LLM agent with MCP tool access for enhanced decision-making."""
    
    def __init__(self, model_name: str = "llama2", use_local: bool = True, use_ollama: bool = True):
        self.name = "llm_mcp"
        self.decision_history = []
        self.model_name = model_name
        self.use_local = use_local
        self.use_ollama = use_ollama
        self._llm = None
        self._model_available = False
        
        # Performance tracking for MCP tools
        self.platform_performance = {}  # (platform, data_source, experiment_type) -> [latencies]
        self.platform_stats = {}  # platform -> {mean, std, count}
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM (same as base LLM agent)."""
        try:
            if self.use_local:
                if self.use_ollama:
                    try:
                        import ollama
                        try:
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
                            
                            logger.info(f"MCP Agent - Found {len(models)} Ollama model(s): {models}")
                            
                            model_found = False
                            requested_base = self.model_name.split(':')[0].lower()
                            
                            for model in models:
                                model_base = model.split(':')[0].lower()
                                if requested_base == model_base or requested_base in model_base:
                                    model_found = True
                                    self.model_name = model
                                    break
                            
                            if model_found:
                                self._llm = "ollama"
                                self._model_available = True
                                logger.info(f"[OK] MCP Agent - Ollama initialized with model: {self.model_name}")
                            elif models:
                                logger.info(f"MCP Agent - Using available model: {models[0]}")
                                self.model_name = models[0]
                                self._llm = "ollama"
                                self._model_available = True
                            else:
                                logger.warning("MCP Agent - No Ollama models found")
                                self._llm = "simple"
                        except Exception as e:
                            logger.warning(f"MCP Agent - Ollama check failed: {e}")
                            self._llm = "simple"
                    except ImportError:
                        logger.warning("MCP Agent - ollama package not installed")
                        self._llm = "simple"
                else:
                    self._llm = "simple"
            else:
                self._llm = "simple"
            
            if self._llm is None or self._llm == "simple":
                logger.warning("MCP Agent - Using simple reasoning")
                self._llm = "simple"
        except Exception as e:
            logger.warning(f"MCP Agent - LLM initialization failed: {e}")
            self._llm = "simple"
    
    def _get_performance_history_tool(self, platform: str, data_source: str = None, 
                                      experiment_type: str = None) -> Dict[str, Any]:
        """MCP Tool: Get historical performance for a platform."""
        results = {
            'platform': platform,
            'has_history': False,
            'metrics': {}
        }
        
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
            results['has_history'] = True
            results['metrics'] = {
                'mean_latency': np.mean(matching_latencies),
                'std_latency': np.std(matching_latencies),
                'min_latency': np.min(matching_latencies),
                'max_latency': np.max(matching_latencies),
                'count': len(matching_latencies)
            }
        
        return results
    
    def _compare_platforms_tool(self, platforms: List[str], data_source: str = None,
                                experiment_type: str = None) -> List[Dict[str, Any]]:
        """MCP Tool: Compare historical performance of platforms."""
        comparisons = []
        
        for platform in platforms:
            perf = self._get_performance_history_tool(platform, data_source, experiment_type)
            if perf['has_history']:
                comparisons.append({
                    'platform': platform,
                    'mean_latency': perf['metrics']['mean_latency'],
                    'std_latency': perf['metrics']['std_latency'],
                    'sample_size': perf['metrics']['count']
                })
        
        # Sort by mean latency
        comparisons.sort(key=lambda x: x['mean_latency'])
        return comparisons
    
    def _get_platform_specs_tool(self, platform: str) -> Dict[str, Any]:
        """MCP Tool: Get platform specifications and characteristics."""
        specs = {
            'pandas': {
                'type': 'in-memory DataFrame',
                'strengths': ['general-purpose', 'easy to use', 'flexible'],
                'best_for': ['small-medium data', 'mixed operations', 'prototyping'],
                'limitations': ['memory-bound', 'single-threaded']
            },
            'duckdb': {
                'type': 'analytical SQL engine',
                'strengths': ['columnar storage', 'fast aggregations', 'SQL interface'],
                'best_for': ['large datasets', 'OLAP queries', 'aggregations'],
                'limitations': ['setup overhead', 'not for OLTP']
            },
            'polars': {
                'type': 'fast DataFrame library',
                'strengths': ['parallel execution', 'lazy evaluation', 'memory efficient'],
                'best_for': ['large datasets', 'complex transformations', 'performance-critical'],
                'limitations': ['newer API', 'learning curve']
            },
            'sqlite': {
                'type': 'OLTP SQL database',
                'strengths': ['ACID compliance', 'structured queries', 'indexing'],
                'best_for': ['transactional workloads', 'small-medium data', 'multi-user'],
                'limitations': ['slower for analytics', 'single-writer']
            },
            'faiss': {
                'type': 'vector similarity search',
                'strengths': ['GPU support', 'exact/approximate search', 'high-dimensional'],
                'best_for': ['vector embeddings', 'ML similarity', 'large-scale search'],
                'limitations': ['vector-only', 'specialized use']
            },
            'annoy': {
                'type': 'approximate nearest neighbors',
                'strengths': ['fast indexing', 'memory-mapped', 'approximate'],
                'best_for': ['large-scale similarity', 'recommendation', 'clustering'],
                'limitations': ['approximate results', 'vector-only']
            },
            'baseline': {
                'type': 'naive Python implementation',
                'strengths': ['simple', 'no dependencies', 'baseline comparison'],
                'best_for': ['benchmarking', 'verification'],
                'limitations': ['very slow', 'not for production']
            }
        }
        
        return specs.get(platform, {'type': 'unknown', 'strengths': [], 'best_for': [], 'limitations': []})
    
    def _build_mcp_prompt(self, data_source: str, experiment_type: str,
                         available_platforms: List[str], context: Dict[str, Any]) -> str:
        """Build prompt with MCP tool results."""
        # Get performance history for all platforms
        comparisons = self._compare_platforms_tool(available_platforms, data_source, experiment_type)
        
        # Build prompt with tool results
        prompt = f"""You are an expert database system selector. Select the best platform for this workload.

TASK:
- Data Source: {data_source}
- Experiment Type: {experiment_type}
- Available Platforms: {', '.join(available_platforms)}

HISTORICAL PERFORMANCE DATA (via MCP tools):
"""
        
        if comparisons:
            prompt += "Platform performance for similar workloads:\n"
            for comp in comparisons:
                prompt += f"  - {comp['platform']}: {comp['mean_latency']:.2f}ms avg (±{comp['std_latency']:.2f}ms, n={comp['sample_size']})\n"
        else:
            prompt += "No historical data available for this workload combination.\n"
        
        prompt += "\nPLATFORM SPECIFICATIONS (via MCP tools):\n"
        for platform in available_platforms:
            specs = self._get_platform_specs_tool(platform)
            prompt += f"\n{platform.upper()}:\n"
            prompt += f"  Type: {specs['type']}\n"
            prompt += f"  Best for: {', '.join(specs['best_for'])}\n"
        
        prompt += f"\nBased on the historical performance data and platform specifications, "
        prompt += f"which platform should be used for {experiment_type} on {data_source}?\n"
        prompt += f"Respond with just the platform name: {', '.join(available_platforms)}"
        
        return prompt
    
    def _call_llm_with_mcp(self, prompt: str) -> str:
        """Call LLM with MCP-enhanced prompt."""
        if self._llm == "ollama" and self._model_available:
            try:
                import ollama
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.3,  # Lower temperature for more focused decisions
                        'num_predict': 50   # Short response expected
                    }
                )
                return response.get('response', '')
            except Exception as e:
                logger.error(f"MCP Agent - Ollama call failed: {e}")
                return ""
        else:
            # Simple reasoning fallback
            return self._simple_reasoning(prompt)
    
    def _simple_reasoning(self, prompt: str) -> str:
        """Simple rule-based fallback."""
        # Extract data source and experiment type from prompt
        if 'vector' in prompt.lower() and any(p in prompt for p in ['annoy', 'faiss']):
            return 'annoy' if 'annoy' in prompt else 'faiss'
        elif 'aggregate' in prompt.lower() or 'group' in prompt.lower():
            if 'duckdb' in prompt:
                return 'duckdb'
            elif 'polars' in prompt:
                return 'polars'
        return 'pandas'  # Default
    
    def _extract_platform(self, response: str, available_platforms: List[str]) -> Optional[str]:
        """Extract platform name from LLM response."""
        response_lower = response.lower()
        for platform in available_platforms:
            if platform.lower() in response_lower:
                return platform
        return available_platforms[0] if available_platforms else None
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str], context: Dict[str, Any] = None) -> str:
        """Select platform using LLM with MCP tool access."""
        if not available_platforms:
            return None
        
        context = context or {}
        
        # Build MCP-enhanced prompt with tool results
        prompt = self._build_mcp_prompt(data_source, experiment_type, available_platforms, context)
        
        # Call LLM
        response = self._call_llm_with_mcp(prompt)
        
        # Extract platform from response
        selected = self._extract_platform(response, available_platforms)
        if not selected:
            selected = available_platforms[0]
        
        # Get performance history for logging
        comparisons = self._compare_platforms_tool(available_platforms, data_source, experiment_type)
        
        decision = {
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': selected,
            'available_platforms': available_platforms,
            'reasoning': f'LLM+MCP decision: {response[:200]}',
            'llm_model': self.model_name,
            'mcp_tools_used': ['get_performance_history', 'compare_platforms', 'get_platform_specs'],
            'historical_data_available': len(comparisons) > 0,
            'mcp_enabled': True
        }
        
        self.decision_history.append(decision)
        return selected
    
    def update(self, data_source: str, experiment_type: str, platform: str, 
              metrics: Dict[str, Any]):
        """Update performance history for MCP tools."""
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
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Return decision history."""
        return self.decision_history
    
    def get_mcp_statistics(self) -> Dict[str, Any]:
        """Get statistics about MCP tool usage."""
        total_decisions = len(self.decision_history)
        decisions_with_history = sum(1 for d in self.decision_history if d.get('historical_data_available', False))
        
        return {
            'total_decisions': total_decisions,
            'decisions_with_historical_data': decisions_with_history,
            'historical_data_coverage': decisions_with_history / total_decisions if total_decisions > 0 else 0,
            'platforms_tracked': len(self.platform_stats),
            'total_performance_samples': sum(s['count'] for s in self.platform_stats.values())
        }

