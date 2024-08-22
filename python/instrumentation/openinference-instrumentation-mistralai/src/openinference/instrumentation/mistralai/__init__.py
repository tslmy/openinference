import logging
from typing import (
    Any,
    Collection,
)

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.mistralai._chat_wrapper import (
    _AsyncChatWrapper,
    _AsyncStreamChatWrapper,
    _SyncChatWrapper,
)
from openinference.instrumentation.mistralai.package import _instruments
from openinference.instrumentation.mistralai.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_MODULE = "mistralai"


class MistralAIInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for mistralai
    """

    __slots__ = (
        "_tracer",
        "_original_sync_chat_method",
        "_original_sync_stream_chat_method",
        "_original_async_chat_method",
        "_original_async_stream_chat_method",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        try:
            import mistralai
            from mistralai.chat import Chat
        except ImportError as err:
            raise Exception(
                "Could not import mistralai. Please install with `pip install mistralai`."
            ) from err

        wrap_function_wrapper(
            module="mistralai.chat",
            name="Chat.complete",
            wrapper=_SyncChatWrapper(self._tracer, mistralai),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        pass