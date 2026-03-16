import os
import sys
from typing import Any, List, Optional
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from gigachat import GigaChat
from gigachat.models import Chat, Messages

def setup_terminal_encoding():
    """Ensure UTF-8 for terminal output (helps with Cyrillic on Windows)"""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class GigaChatLLM(CustomLLM):
    context_window: int = 8192
    num_output: int = 4096
    model_name: str = "GigaChat"
    credentials: str = ""
    scope: str = "GIGACHAT_API_PERS"
    _client: Any = None

    def __init__(self, credentials: str, scope: str = "GIGACHAT_API_PERS", model_name: str = "GigaChat"):
        super().__init__(credentials=credentials, scope=scope, model_name=model_name)
        self._client = GigaChat(credentials=credentials, scope=scope, verify_ssl_certs=False)

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        payload = Chat(
            messages=[Messages(role="user", content=prompt)],
            model=self.model_name,
        )
        response = self._client.chat(payload)
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Streaming not implemented for GigaChatLLM")
