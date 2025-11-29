"""
Agent manager for initializing and managing agents.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class AgentManager:
    """Manages agents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.agents = {}
        self.config = config or {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents."""
        # Map agent name -> (module_path, class_name)
        agent_modules = {
            'rule_based': ('agents.agent_rule_based', 'RuleBasedAgent'),
            'bandit': ('agents.agent_bandit', 'BanditAgent'),
            'cost_model': ('agents.agent_cost_model', 'CostModelAgent'),
            'llm': ('agents.agent_llm', 'LLMAgent'),
            'llm_mcp': ('agents.agent_llm_mcp', 'LLMAgentMCP'),
            'langchain': ('agents.agent_langchain', 'LangChainAgent'),
            'langgraph': ('agents.agent_langgraph', 'LangGraphAgent'),
            'fastagency': ('agents.agent_fastagency', 'FastAgencyAgent'),
            'autogen': ('agents.agent_autogen', 'AutoGenAgent'),
            'hybrid': ('agents.agent_hybrid', 'HybridAgent'),
            'random': ('agents.agent_random', 'RandomAgent'),
            'oracle': ('agents.agent_oracle', 'OracleAgent'),
            'static_best': ('agents.agent_static_best', 'StaticBestAgent'),
            'round_robin': ('agents.agent_round_robin', 'RoundRobinAgent'),
            'linucb': ('agents.agent_linucb', 'LinucbAgent'),
            'thompson': ('agents.agent_thompson', 'ThompsonAgent')
        }
        
        # Check if multi-model testing is enabled
        llm_config = self.config.get('llm_config', {})
        enable_multi_model = llm_config.get('enable_multi_model', False)
        test_models = llm_config.get('test_models', [])
        
        # Initialize standard agents
        for name, (module_path, class_name) in agent_modules.items():
            try:
                module = __import__(module_path, fromlist=[class_name])
                agent_class = getattr(module, class_name)
                
                # Pass llm_config to LLM-based and Hybrid agents
                if name in ['llm', 'llm_mcp', 'langchain', 'langgraph', 'fastagency', 'autogen', 'hybrid']:
                    agent = agent_class(
                        model_name=llm_config.get('model_name', 'llama2'),
                        use_local=llm_config.get('use_local', True),
                        use_ollama=llm_config.get('use_ollama', True)
                    )
                else:
                    agent = agent_class()
                
                self.agents[name] = agent
                logger.info(f"Initialized agent: {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize agent {name}: {e}")
        
        # Initialize multi-model LLM agents if enabled
        if enable_multi_model and test_models:
            logger.info(f"Multi-model testing enabled: creating agents for {len(test_models)} models")
            self._initialize_multi_model_agents(test_models, llm_config)
    
    def _initialize_multi_model_agents(self, models: List[str], llm_config: Dict[str, Any]):
        """Initialize separate agent instances for each LLM model."""
        from agents.agent_llm import LLMAgent
        from agents.agent_llm_mcp import LLMAgentMCP
        
        for model_name in models:
            # Skip if this is the default model (already initialized)
            if model_name == llm_config.get('model_name', 'llama2'):
                continue
            
            # Create sanitized agent name from model name
            # e.g., "qwen3:14b" -> "llm_qwen3_14b"
            safe_name = model_name.replace(':', '_').replace('.', '_').replace('-', '_')
            
            # Initialize standard LLM agent with this model
            llm_agent_name = f"llm_{safe_name}"
            try:
                agent = LLMAgent(
                    model_name=model_name,
                    use_local=llm_config.get('use_local', True),
                    use_ollama=llm_config.get('use_ollama', True)
                )
                # Update agent name to reflect model
                agent.name = llm_agent_name
                agent.model_display_name = model_name
                self.agents[llm_agent_name] = agent
                logger.info(f"Initialized multi-model agent: {llm_agent_name} (model: {model_name})")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM agent with model {model_name}: {e}")
            
            # Initialize MCP variant with this model
            mcp_agent_name = f"llm_mcp_{safe_name}"
            try:
                agent = LLMAgentMCP(
                    model_name=model_name,
                    use_local=llm_config.get('use_local', True),
                    use_ollama=llm_config.get('use_ollama', True)
                )
                # Update agent name to reflect model
                agent.name = mcp_agent_name
                agent.model_display_name = model_name
                self.agents[mcp_agent_name] = agent
                logger.info(f"Initialized multi-model MCP agent: {mcp_agent_name} (model: {model_name})")
            except Exception as e:
                logger.warning(f"Failed to initialize MCP agent with model {model_name}: {e}")
    
    def get_agent(self, name: str):
        """Get agent by name."""
        return self.agents.get(name)
    
    def get_all_agents(self) -> Dict[str, Any]:
        """Get all initialized agents."""
        return self.agents
    
    def get_agent_names(self) -> List[str]:
        """Get list of available agent names."""
        return list(self.agents.keys())
    
    def is_available(self, name: str) -> bool:
        """Check if agent is available."""
        return name in self.agents

