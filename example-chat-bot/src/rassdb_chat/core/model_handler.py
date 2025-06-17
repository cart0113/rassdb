"""Model handler for both local (Ollama) and API (Anthropic) models."""

import os
import json
import logging
from typing import AsyncIterator, Optional, Dict, Any
from abc import ABC, abstractmethod

import httpx
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


class ModelHandler(ABC):
    """Abstract base class for model handlers."""
    
    @abstractmethod
    async def generate(self, prompt: str, context: Optional[str] = None) -> AsyncIterator[str]:
        """Generate response from the model."""
        pass
    
    @abstractmethod
    async def check_status(self) -> Dict[str, Any]:
        """Check model status."""
        pass


class OllamaHandler(ModelHandler):
    """Handler for local Ollama models."""
    
    def __init__(self, model_name: str = "qwen2.5-coder:3b-instruct"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate(self, prompt: str, context: Optional[str] = None) -> AsyncIterator[str]:
        """Generate response using Ollama API."""
        if context:
            full_prompt = f"Context:\n{context}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = prompt
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse Ollama response: {line}")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            yield f"Error: Failed to generate response - {str(e)}"
    
    async def check_status(self) -> Dict[str, Any]:
        """Check Ollama status and available models."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return {
                "status": "connected",
                "models": [model["name"] for model in data.get("models", [])]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()


class AnthropicHandler(ModelHandler):
    """Handler for Anthropic API models."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.model_name = model_name
        self.client = AsyncAnthropic(api_key=self.api_key)
    
    async def generate(self, prompt: str, context: Optional[str] = None) -> AsyncIterator[str]:
        """Generate response using Anthropic API."""
        messages = []
        
        if context:
            messages.append({
                "role": "user",
                "content": f"Context:\n{context}\n\nUser: {prompt}"
            })
        else:
            messages.append({
                "role": "user",
                "content": prompt
            })
        
        try:
            stream = await self.client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=4096,
                stream=True
            )
            
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield event.delta.text
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            yield f"Error: Failed to generate response - {str(e)}"
    
    async def check_status(self) -> Dict[str, Any]:
        """Check Anthropic API status."""
        try:
            # Simple test to verify API key works
            response = await self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return {
                "status": "connected",
                "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()


def create_model_handler(model_type: str, model_name: Optional[str] = None, api_key: Optional[str] = None) -> ModelHandler:
    """Factory function to create appropriate model handler."""
    if model_type == "ollama":
        return OllamaHandler(model_name or "qwen2.5-coder:3b-instruct")
    elif model_type == "anthropic":
        return AnthropicHandler(api_key=api_key, model_name=model_name or "claude-3-5-sonnet-20241022")
    else:
        raise ValueError(f"Unknown model type: {model_type}")