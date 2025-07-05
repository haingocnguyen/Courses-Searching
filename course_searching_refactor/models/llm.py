import time
import asyncio
import logging
import streamlit as st
from typing import List, Dict, Generator, AsyncGenerator
import ollama
from config import LLM_MODEL

logger = logging.getLogger(__name__)

class OllamaLLM:
    """Ollama LLM wrapper with streaming and async support"""
    
    def __init__(self, model=LLM_MODEL, small=True):
        self.model = model
        self.opts = {'temperature': 0.1 if small else 0.3}
        
    async def invoke_async(self, messages: List[Dict[str, str]]) -> str:
        """Async invocation for non-blocking operations"""
        start_time = time.perf_counter()
        prompt = "\n".join(m["content"] for m in messages)
        
        response = await asyncio.to_thread(ollama.generate, 
                        model=self.model, 
                        prompt=prompt, 
                        options=self.opts)
            
        logger.debug(f"LLM invocation took {time.perf_counter() - start_time:.2f} seconds")
        return response['response']
    
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """Synchronous invocation with error handling"""
        prompt = "\n".join(m["content"] for m in messages)
        try:
            response = ollama.generate(model=self.model, prompt=prompt, options=self.opts)
            return response['response']
        except Exception as e:
            logger.error(f"LLM invoke failed: {e}")
            return "Sorry, I encountered an error. Please try again."

    async def stream_async(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Async streaming for real-time responses"""
        prompt = "\n".join(m["content"] for m in messages)
        stream = ollama.generate(model=self.model, prompt=prompt, options=self.opts, stream=True)
        for chunk in stream:
            if chunk.get('response'):
                yield chunk['response']

    def stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Synchronous streaming"""
        prompt = "\n".join(m["content"] for m in messages)
        stream = ollama.generate(model=self.model, prompt=prompt, options=self.opts, stream=True)
        for chunk in stream:
            if chunk.get('response'):
                yield chunk['response']
                
    def stream_with_patience(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Patient streaming with status updates but no interruption"""
        prompt = "\n".join(m["content"] for m in messages)
        
        try:
            stream = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={**self.opts, 'stream': True},
                stream=True
            )
            
            start_time = time.time()
            last_chunk_time = start_time
            has_content = False
            in_think_tag = False
            total_chunks = 0
            
            for chunk in stream:
                current_time = time.time()
                
                if chunk.get('response'):
                    content = chunk['response']
                    total_chunks += 1
                    
                    # Handle thinking tags
                    if '<think>' in content:
                        in_think_tag = True
                        content = content.split('<think>')[0]
                    if '</think>' in content:
                        in_think_tag = False
                        content = content.split('</think>')[-1]
                    if in_think_tag:
                        continue
                    
                    # First content check
                    if not has_content and content.strip():
                        initial_wait = current_time - start_time
                        if initial_wait > 20:
                            yield f"⏳ LLM took {initial_wait:.1f}s to start. Continuing...\n\n"
                        has_content = True
                    
                    # Long gap warning but don't interrupt
                    if has_content:
                        gap = current_time - last_chunk_time
                        if gap > 5:
                            yield f"⏳ Processing... "
                    
                    # Yield actual content
                    if content:
                        yield content
                        last_chunk_time = current_time
                        
        except Exception as e:
            yield f"\n\n❌ Stream error: {e}"

@st.cache_resource
def get_llm(small=True):
    """Cached LLM instance"""
    return OllamaLLM(small=small)