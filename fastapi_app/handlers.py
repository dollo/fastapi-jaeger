
from typing import Any
from typing import AsyncIterator
from typing import Dict
from typing import List
from typing import Literal
from typing import Sequence
from typing import Union
from typing import cast

import opentelemetry

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction
from langchain.schema import AgentFinish
from langchain.schema import BaseMessage
from langchain.schema import LLMResult
from langchain.schema.document import Document

from langchain.callbacks.base import BaseCallbackHandler
from loguru import logger
import opentelemetry
from opentelemetry.trace import Tracer
from opentelemetry.trace.span import Span


class TracingCallbackHandler(BaseCallbackHandler):
    """Handle callbacks from langchain that activates opentelemetry spans and logging"""

    def __init__(self, tracer: Tracer) -> None:
        """Initialize callback handler."""
        if tracer is None:
            raise Exception(
                "No tracer defined. Please initialized the callback with an opentelemetry tracer."
            )
        self.tracer = tracer
        self.llm_span: Span = None
        self.retriever_span: Span = None

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        logger.info("on_llm_start")
        self.llm_span = self.tracer.start_span("llm_inference")
        with opentelemetry.trace.use_span(self.llm_span, end_on_exit=False):
            prompts_len = sum([len(prompt) for prompt in prompts])
            self.llm_span.set_attribute("num_processed_prompts", len(prompts))
            self.llm_span.set_attribute("prompts_len", prompts_len)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        logger.info("on_chat_model_start")
        self.llm_span = self.tracer.start_span("llm_inference")
        with opentelemetry.trace.use_span(self.llm_span, end_on_exit=False):
            self.llm_span.set_attribute("messages", messages)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        logger.info("on_llm_end")
        with opentelemetry.trace.use_span(self.llm_span, end_on_exit=True):
            token_usage = response.llm_output["token_usage"]
            for k, v in token_usage.items():
                self.llm_span.set_attribute(k, v)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        logger.info("on_chain_end")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    def on_retriever_start(
        self, serialized: Dict[str, Any], query: str, **kwargs: Any
    ) -> Any:
        """Run when retriever starts."""
        logger.info("on_retriever_start")
        self.retriever_span = self.tracer.start_span("vectorstore_retrieval")
        with opentelemetry.trace.use_span(self.retriever_span, end_on_exit=False):
            self.retriever_span.set_attribute("query", query)
            for k, v in serialized.items():
                self.retriever_span.set_attribute(k, v)

    def on_retriever_end(self, documents: Sequence[Document], **kwargs: Any) -> Any:
        """Run when retriever ends running."""
        logger.info("on_retriever_end")
        with opentelemetry.trace.use_span(self.retriever_span, end_on_exit=True):
            n_documents = len(documents)
            self.retriever_span.set_attribute("n_documents", n_documents)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
