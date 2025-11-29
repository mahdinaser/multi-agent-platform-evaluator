"""
LLM agent using actual language model for platform selection.
"""
import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)

class LLMAgent:
    """Agent that uses actual LLM for decision making."""
    
    def __init__(self, model_name: str = "llama2", use_local: bool = True, use_ollama: bool = True):
        self.name = "llm"
        self.decision_history = []
        self.model_name = model_name
        self.use_local = use_local
        self.use_ollama = use_ollama
        self._llm = None
        self._model_available = False
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM (Ollama, transformers, or API-based)."""
        try:
            if self.use_local:
                # Try Ollama first (easiest to use)
                if self.use_ollama:
                    try:
                        import ollama
                        # Test if Ollama is running and model exists
                        try:
                            models_response = ollama.list()
                            logger.debug(f"Ollama response type: {type(models_response)}")
                            
                            # Handle different response formats
                            models_list = []
                            # Ollama returns ListResponse object with .models attribute
                            if hasattr(models_response, 'models'):
                                models_list = models_response.models
                                logger.debug(f"Extracted models_list from ListResponse: {len(models_list)} models")
                            elif isinstance(models_response, dict):
                                models_list = models_response.get('models', [])
                                logger.debug(f"Extracted models_list from dict: {models_list}")
                            elif isinstance(models_response, list):
                                models_list = models_response
                                logger.debug(f"Response is list: {models_list}")
                            else:
                                logger.warning(f"Unexpected response format: {type(models_response)}")
                            
                            # Extract model names (handle both dict and string formats)
                            models = []
                            for m in models_list:
                                # Handle Model objects (has .model attribute)
                                if hasattr(m, 'model'):
                                    model_name = m.model
                                elif isinstance(m, dict):
                                    # Try different possible keys - Ollama uses 'name' or 'model'
                                    model_name = m.get('name') or m.get('model') or m.get('id') or str(m)
                                elif isinstance(m, str):
                                    model_name = m
                                else:
                                    model_name = str(m)
                                if model_name:
                                    models.append(model_name)
                            
                            logger.info(f"Found {len(models)} Ollama model(s): {models}")
                            
                            # Check if model exists (handle both "llama2" and "llama2:latest")
                            model_found = False
                            selected_model = None
                            requested_base = self.model_name.split(':')[0].lower()
                            
                            for model in models:
                                # Check if requested model matches (case-insensitive, handle tags)
                                model_base = model.split(':')[0].lower()
                                if requested_base == model_base or requested_base in model_base or model_base in requested_base:
                                    model_found = True
                                    selected_model = model
                                    logger.debug(f"Matched model: {model} (requested: {self.model_name})")
                                    break
                            
                            if model_found:
                                self.model_name = selected_model  # Use exact model name
                                self._llm = "ollama"
                                self._model_available = True
                                logger.info(f"[OK] Ollama initialized with model: {self.model_name}")
                            else:
                                if models:
                                    logger.warning(f"Model '{self.model_name}' not found in Ollama. Available: {models}")
                                    # Use first available model as fallback
                                    logger.info(f"  -> Using available model: {models[0]} as fallback")
                                    self.model_name = models[0]
                                    self._llm = "ollama"
                                    self._model_available = True
                                    logger.info(f"[OK] Ollama initialized with fallback model: {self.model_name}")
                                else:
                                    logger.warning(f"No models installed in Ollama. Response was: {models_response}")
                                    logger.info(f"To use LLM, run: ollama pull {self.model_name}")
                                    self._llm = None
                                    self._model_available = False
                        except Exception as e:
                            logger.warning(f"Ollama check failed: {e}", exc_info=True)
                            logger.info("Falling back to transformers. To use Ollama:")
                            logger.info("  1. Make sure Ollama server is running: ollama serve")
                            logger.info(f"  2. Pull a model: ollama pull {self.model_name}")
                            self._llm = None
                            self._model_available = False
                    except ImportError:
                        logger.warning("ollama package not installed. Install with: pip install ollama")
                        logger.info("Also make sure Ollama application is installed from https://ollama.ai/")
                        self._llm = None
                
                # Fallback to transformers if Ollama not available
                if self._llm is None:
                    try:
                        from transformers import AutoTokenizer, AutoModelForCausalLM
                        import torch
                        
                        # Use a small, efficient model for local inference
                        model_id = self.model_name if "/" in self.model_name else f"microsoft/DialoGPT-small"
                        
                        try:
                            logger.info(f"Loading local LLM model via transformers: {model_id}")
                            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                            self.model = AutoModelForCausalLM.from_pretrained(model_id)
                            self._llm = "local_transformers"
                            logger.info("Local LLM (transformers) initialized successfully")
                        except Exception as e:
                            logger.warning(f"Failed to load transformers model: {e}")
                            self._llm = None
                    except ImportError:
                        logger.warning("transformers not available")
                        self._llm = None
            else:
                # Use API-based LLM (OpenAI, etc.)
                try:
                    import openai
                    self._llm = "openai"
                    logger.info("OpenAI API available")
                except ImportError:
                    logger.warning("OpenAI not available")
                    self._llm = None
            
            # Final fallback
            if self._llm is None:
                logger.warning("No LLM available, using simple rule-based reasoning")
                self._llm = "simple"
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}, using simple reasoning")
            self._llm = "simple"
    
    def _call_llm_ollama(self, prompt: str) -> str:
        """Call Ollama LLM model."""
        if self._llm == "ollama" and self._model_available:
            try:
                import ollama
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.7,
                        'num_predict': 200  # Limit response length
                    }
                )
                return response.get('response', '')
            except Exception as e:
                logger.error(f"Ollama LLM call failed: {e}")
                # Mark model as unavailable to avoid repeated calls
                self._model_available = False
                self._llm = "simple"
                return ""
        return ""
    
    def _call_llm_local(self, prompt: str) -> str:
        """Call local LLM model via transformers."""
        if self._llm == "local_transformers":
            try:
                import torch
                # Create a simple prompt-completion task
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 50,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract only the new generated text (after prompt)
                if prompt in response:
                    response = response.split(prompt, 1)[-1].strip()
                return response
            except Exception as e:
                logger.error(f"Local LLM call failed: {e}")
                return ""
        return ""
    
    def _call_llm_api(self, prompt: str) -> str:
        """Call API-based LLM."""
        if self._llm == "openai":
            try:
                import openai
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert in database and data processing platforms."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"API LLM call failed: {e}")
                return ""
        return ""
    
    def _extract_platform_from_response(self, response: str, available_platforms: List[str]) -> Optional[str]:
        """Extract platform name from LLM response."""
        response_lower = response.lower()
        
        # Check for platform mentions
        for platform in available_platforms:
            if platform.lower() in response_lower:
                return platform
        
        # Fallback: return first available
        return available_platforms[0] if available_platforms else None
    
    def _llm_reasoning(self, data_source: str, experiment_type: str,
                      available_platforms: List[str], context: Dict[str, Any]) -> tuple:
        """Use actual LLM for reasoning."""
        # Build prompt
        platform_info = {
            'pandas': 'General-purpose in-memory DataFrame library, good for small-medium data',
            'duckdb': 'Analytical SQL engine optimized for OLAP queries and aggregations',
            'sqlite': 'OLTP SQL database engine, good for structured queries and transactions',
            'faiss': 'Facebook AI Similarity Search, specialized for high-performance vector search',
            'annoy': 'Approximate Nearest Neighbors, fast indexing for similarity search',
            'baseline': 'Naive Python implementation for baseline comparison'
        }
        
        available_info = "\n".join([f"- {p}: {platform_info.get(p, 'Available platform')}" 
                                   for p in available_platforms])
        
        prompt = f"""Given the following scenario, recommend the best data processing platform:

Data Source: {data_source}
Experiment Type: {experiment_type}
Data Size: {context.get('size', 'unknown')} records
Available Platforms:
{available_info}

Consider:
1. The data source type and characteristics
2. The operation type (scan, filter, aggregate, join, etc.)
3. Platform strengths and specializations
4. Expected data size

Recommend ONE platform from the available list and briefly explain why."""

        # Call LLM (try Ollama first, then transformers, then API, then fallback)
        if self._llm == "ollama" and self._model_available:
            response = self._call_llm_ollama(prompt)
            # If call failed, fall back to simple reasoning
            if not response:
                response = self._simple_reasoning(data_source, experiment_type, available_platforms)
        elif self._llm == "local_transformers":
            response = self._call_llm_local(prompt)
            if not response:
                response = self._simple_reasoning(data_source, experiment_type, available_platforms)
        elif self._llm == "openai":
            response = self._call_llm_api(prompt)
            if not response:
                response = self._simple_reasoning(data_source, experiment_type, available_platforms)
        else:
            # Fallback to simple reasoning
            response = self._simple_reasoning(data_source, experiment_type, available_platforms)
        
        # Extract platform decision
        decision = self._extract_platform_from_response(response, available_platforms)
        if not decision:
            decision = available_platforms[0] if available_platforms else "pandas"
        
        reasoning = f"LLM Analysis:\nPrompt: {prompt[:200]}...\n\nResponse: {response[:500]}\n\nSelected: {decision}"
        
        return decision, reasoning
    
    def _simple_reasoning(self, data_source: str, experiment_type: str, 
                         available_platforms: List[str]) -> str:
        """Simple fallback reasoning when LLM is not available."""
        if data_source == 'vectors':
            if 'faiss' in available_platforms:
                return "For vector data, FAISS is the specialized choice for high-performance similarity search."
            elif 'annoy' in available_platforms:
                return "For vector data, Annoy provides fast approximate nearest neighbor search."
        elif experiment_type == 'aggregate':
            if 'duckdb' in available_platforms:
                return "For aggregation queries, DuckDB is optimized for analytical workloads."
        elif experiment_type == 'join':
            if 'duckdb' in available_platforms:
                return "For join operations, DuckDB handles analytical joins efficiently."
        
        return f"Selecting {available_platforms[0]} as a general-purpose option."
    
    def select_platform(self, data_source: str, experiment_type: str,
                       available_platforms: List[str],
                       context: Dict[str, Any] = None) -> str:
        """Select platform using actual LLM reasoning."""
        if context is None:
            context = {}
        
        decision, reasoning = self._llm_reasoning(data_source, experiment_type, available_platforms, context)
        
        self.decision_history.append({
            'data_source': data_source,
            'experiment_type': experiment_type,
            'selected_platform': decision,
            'reasoning': reasoning,
            'context': context,
            'llm_type': self._llm
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

