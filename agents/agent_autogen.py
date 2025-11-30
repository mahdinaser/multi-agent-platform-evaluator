"""
AutoGen agent for multi-agent conversations with tools.
Best for multi-agent local conversations and collaborative decision-making.
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class AutoGenAgent:
    """AutoGen agent for multi-agent collaborative decision-making."""
    
    def __init__(self, model_name: str = "llama2", use_local: bool = True, use_ollama: bool = True):
        self.name = "autogen"
        self.decision_history = []
        self.model_name = model_name
        self.use_local = use_local
        self.use_ollama = use_ollama
        self._llm = None
        self._model_available = False
        self._agents = []  # Multi-agent system
        
        # Performance tracking
        self.platform_performance = {}
        self.platform_stats = {}
        
        self._initialize_autogen()
    
    def _initialize_autogen(self):
        """Initialize AutoGen with multi-agent system."""
        try:
            try:
                from autogen import ConversableAgent, GroupChat, GroupChatManager
                from autogen.agentchat.contrib.capabilities.teachable_agent import TeachableAgent
                
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
                        
                        # Create LLM config for AutoGen
                        # AutoGen supports Ollama via custom config
                        self._llm_config = {
                            "model": self.model_name,
                            "base_url": "http://localhost:11434",
                            "api_type": "ollama"
                        }
                        
                        self._model_available = True
                        logger.info(f"[OK] AutoGen initialized with Ollama model: {self.model_name}")
                        
                        # Create multi-agent system
                        self._create_agent_system()
                        
                    except Exception as e:
                        logger.warning(f"AutoGen - Ollama setup failed: {e}")
                        self._llm = "simple"
                else:
                    self._llm = "simple"
                    
            except ImportError:
                logger.warning("AutoGen packages not installed. Install with: pip install pyautogen")
                self._llm = "simple"
                
        except Exception as e:
            logger.warning(f"AutoGen initialization failed: {e}")
            self._llm = "simple"
    
    def _create_agent_system(self):
        """Create multi-agent system for collaborative decision-making."""
        try:
            from autogen import ConversableAgent, GroupChat, GroupChatManager
            
            # Create specialized agents
            # 1. Performance Analyst - analyzes historical data
            performance_analyst = ConversableAgent(
                name="performance_analyst",
                system_message="You are a performance analyst. Analyze historical performance data and provide insights.",
                llm_config=self._llm_config if self._model_available else None,
                human_input_mode="NEVER"
            )
            
            # 2. Platform Specialist - knows platform specs
            platform_specialist = ConversableAgent(
                name="platform_specialist",
                system_message="You are a platform specialist. You know the characteristics and best use cases for each platform.",
                llm_config=self._llm_config if self._model_available else None,
                human_input_mode="NEVER"
            )
            
            # 3. Decision Maker - makes final decision
            decision_maker = ConversableAgent(
                name="decision_maker",
                system_message="You are a decision maker. Synthesize information from other agents and select the best platform.",
                llm_config=self._llm_config if self._model_available else None,
                human_input_mode="NEVER"
            )
            
            self._agents = [performance_analyst, platform_specialist, decision_maker]
            
            # Create group chat
            groupchat = GroupChat(
                agents=self._agents,
                messages=[],
                max_round=5  # Limit conversation rounds
            )
            
            self._manager = GroupChatManager(
                groupchat=groupchat,
                llm_config=self._llm_config if self._model_available else None
            )
            
        except Exception as e:
            logger.warning(f"Failed to create AutoGen agent system: {e}")
            self._agents = []
            self._manager = None
    
    def _build_autogen_task(self, data_source: str, experiment_type: str,
                           available_platforms: List[str], context: Dict[str, Any]) -> str:
        """Build task for AutoGen multi-agent conversation."""
        # Gather performance data
        comparisons = []
        for platform in available_platforms:
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
        
        task = f"""Collaboratively select the best platform for this workload.

TASK:
- Data Source: {data_source}
- Experiment Type: {experiment_type}
- Available Platforms: {', '.join(available_platforms)}

PERFORMANCE ANALYST: Analyze this historical performance data:
"""
        
        if comparisons:
            for comp in comparisons:
                task += f"  {comp['platform']}: {comp['mean']:.2f}ms avg (n={comp['count']})\n"
        else:
            task += "  No historical data available.\n"
        
        task += f"""
PLATFORM SPECIALIST: Consider platform characteristics for {experiment_type} workloads.

DECISION MAKER: Based on the analysis and platform knowledge, select the best platform from: {', '.join(available_platforms)}
Respond with just the platform name."""
        
        return task
    
    def _call_autogen_agents(self, task: str) -> str:
        """Call AutoGen multi-agent system."""
        if hasattr(self, '_manager') and self._manager is not None and self._model_available:
            try:
                # Use decision_maker to initiate conversation
                result = self._agents[-1].initiate_chat(
                    self._manager,
                    message=task,
                    max_turns=5
                )
                
                # Extract platform from conversation
                messages = result.chat_history if hasattr(result, 'chat_history') else []
                for msg in reversed(messages):
                    content = str(msg.get('content', ''))
                    platform = self._extract_platform(content, [])
                    if platform:
                        return platform
                
                return self._simple_reasoning(task)
                
            except Exception as e:
                logger.error(f"AutoGen agent call failed: {e}")
                return self._simple_reasoning(task)
        else:
            return self._simple_reasoning(task)
    
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
        platforms = ['pandas', 'duckdb', 'polars', 'sqlite', 'faiss', 'annoy', 'baseline']
        for platform in platforms:
            if platform in response_lower:
                return platform
        return None
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str], context: Dict[str, Any] = None) -> str:
        """Select platform using AutoGen multi-agent system."""
        if not available_platforms:
            return None
        
        context = context or {}
        
        # Build task
        task = self._build_autogen_task(data_source, experiment_type, available_platforms, context)
        
        # Call AutoGen agents
        response = self._call_autogen_agents(task)
        
        # Extract platform
        selected = self._extract_platform(response, available_platforms)
        if not selected:
            # Fallback: use performance data
            best_platform = None
            best_mean = float('inf')
            
            for platform in available_platforms:
                matching_latencies = []
                for key, latencies in self.platform_performance.items():
                    if key[0] == platform:
                        matching_latencies.extend(latencies)
                
                if matching_latencies:
                    mean = np.mean(matching_latencies)
                    if mean < best_mean:
                        best_mean = mean
                        best_platform = platform
            
            selected = best_platform or available_platforms[0]
        
        decision = {
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': selected,
            'available_platforms': available_platforms,
            'reasoning': f'AutoGen multi-agent: {response[:200]}',
            'llm_model': self.model_name,
            'framework': 'AutoGen',
            'multi_agent': True,
            'agents_used': ['performance_analyst', 'platform_specialist', 'decision_maker']
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

