import asyncio
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

if TYPE_CHECKING:
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

    from handlers.story_handlers import ConversationMemory


class TestInputTransport(BaseInputTransport):
    """A transport that emits a single user utterance and then ends."""

    def __init__(self, utterance: str, **kwargs):
        super().__init__(
            params=TransportParams(audio_in_enabled=False, audio_out_enabled=False), **kwargs
        )
        self._utterance = utterance
        self._sent_utterance = False

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Small delay to ensure pipeline is ready
        await asyncio.sleep(0.1)
        # Push the utterance as a transcription frame
        await self.push_frame(
            TranscriptionFrame(
                text=self._utterance,
                user_id="user",
                timestamp=str(asyncio.get_event_loop().time()),
            )
        )
        self._sent_utterance = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Don't send EndFrame until we're told to by the output transport
        await super().process_frame(frame, direction)

    async def end_input(self):
        """Called by the output transport when it's ready to end"""
        if self._sent_utterance:
            await self.push_frame(EndFrame())


class TestOutputTransport(BaseOutputTransport):
    """A transport that captures the bot's final state and resolves a future."""

    def __init__(
        self,
        future: asyncio.Future,
        memory: "ConversationMemory",
        context: "OpenAILLMContext",
        input_transport: TestInputTransport,
        **kwargs,
    ):
        super().__init__(
            params=TransportParams(audio_in_enabled=False, audio_out_enabled=False), **kwargs
        )
        self._future = future
        self._memory = memory
        self._context = context
        self._input_transport = input_transport
        self._buffer: list[str] = []
        self._response_received = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            self._buffer.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            # LLM has finished generating the complete response
            self._response_received = True
            # Signal input transport to end
            await self._input_transport.end_input()
        elif isinstance(frame, EndFrame):
            if self._response_received:
                # Only resolve when we've received a complete LLM response
                state = self._memory.to_dict()
                state["messages"] = self._context.messages
                if not self._future.done():
                    self._future.set_result({"state": state})
            else:
                # If we haven't received a response, resolve with current state
                state = self._memory.to_dict()
                state["messages"] = self._context.messages
                if not self._future.done():
                    self._future.set_result({"state": state})


class TestTransport(BaseTransport):
    """A combined transport for headless testing."""

    def __init__(
        self,
        utterance: str,
        future: asyncio.Future,
        memory: "ConversationMemory",
        context: "OpenAILLMContext",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input = TestInputTransport(utterance=utterance)
        self._output = TestOutputTransport(
            future=future, memory=memory, context=context, input_transport=self._input
        )

    def input(self):
        return self._input

    def output(self):
        return self._output
