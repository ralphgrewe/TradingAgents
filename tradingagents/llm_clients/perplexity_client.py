"""Perplexity Agent API client for TradingAgents.

This client provides access to Perplexity's Agent API, which allows
running agent-based workflows with built-in tools like web_search and fetch_url.
"""

import os
from typing import Any, Optional

try:
    import perplexity
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False

from .base_client import BaseLLMClient


class PerplexityAgentClient:
    """Wrapper for Perplexity Agent API.
    
    This class provides a simplified interface to the Perplexity Agent API,
    specifically for running news analysis agents with web search capabilities.
    """
    
    def __init__(self, model: str = "sonar-pro", api_key: Optional[str] = None):
        """Initialize the Perplexity Agent client.
        
        Args:
            model: Perplexity model to use (default: "sonar-pro")
            api_key: Perplexity API key. If None, will try environment variable.
        """
        if not PERPLEXITY_AVAILABLE:
            raise ImportError(
                "Perplexity SDK is not installed. Install with: pip install perplexity"
            )
        
        self.model = model
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Perplexity API key is required. Set PERPLEXITY_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = perplexity.Client(api_key=self.api_key)
    
    def run_agent(
        self,
        messages: list,
        tools: Optional[list] = None,
        response_format: Optional[dict] = None,
        **kwargs
    ) -> Any:
        """Run a Perplexity agent with the given configuration.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: List of tool configurations for the agent
            response_format: Optional response format specification
            **kwargs: Additional arguments passed to client.agents.run()
            
        Returns:
            Agent response (typically a message with content and tool calls)
        """
        run_kwargs = {
            "model": self.model,
            "messages": messages,
        }
        
        if tools:
            run_kwargs["tools"] = tools
        
        if response_format:
            run_kwargs["response_format"] = response_format
        
        # Merge with additional kwargs
        run_kwargs.update(kwargs)
        
        response = self.client.agents.run(**run_kwargs)
        return response


class PerplexityClient(BaseLLMClient):
    """LLM client for Perplexity chat completions and agent API.
    
    This client provides both traditional chat completions and access to
    Perplexity's Agent API for more complex workflows.
    """
    
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Perplexity client.
        
        Args:
            model: Perplexity model name
            base_url: Optional base URL (not typically used for Perplexity)
            api_key: Perplexity API key
            **kwargs: Additional arguments
        """
        super().__init__(model, base_url, **kwargs)
        self.provider = "perplexity"
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        
        if not PERPLEXITY_AVAILABLE:
            raise ImportError(
                "Perplexity SDK is not installed. Install with: pip install perplexity"
            )
    
    def get_llm(self) -> Any:
        """Return a LangChain-compatible LLM instance.
        
        Note: For Agent API usage, use get_agent_client() instead.
        """
        from langchain_community.chat_models import ChatPerplexity
        
        self.warn_if_unknown_model()
        
        api_key = self.api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("Perplexity API key is required")
        
        return ChatPerplexity(
            model=self.model,
            api_key=api_key,
            temperature=self.kwargs.get("temperature"),
        )
    
    def get_agent_client(self) -> PerplexityAgentClient:
        """Return a Perplexity Agent API client.
        
        This provides access to the Agent API with tools like web_search.
        """
        return PerplexityAgentClient(
            model=self.model,
            api_key=self.api_key,
        )
    
    def validate_model(self) -> bool:
        """Validate that the model is supported by Perplexity.
        
        Note: Perplexity doesn't publish a comprehensive model list,
        so we validate against known models.
        """
        # List of known Perplexity models
        known_models = {
            "sonar",
            "sonar-pro",
            "sonar-small",
            "sonar-small-chat",
            "sonar-small-online",
            "sonar-chat",
            "sonar-online",
            "llama-3.1-sonar-large-128k-chat",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-70b-instruct",
            "llama-3-70b-instruct",
            "llama-3-8b-instruct",
            "mistral-large",
            "mixtral-8x7b-instruct",
        }
        return self.model.lower() in known_models
