import asyncio
from typing import TYPE_CHECKING, Union

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
    """A transport that can receive multiple utterances over time."""

    def __init__(self, initial_utterance: Union[str, None] = None, **kwargs):
        super().__init__(
            params=TransportParams(audio_in_enabled=False, audio_out_enabled=False), **kwargs
        )
        self._utterance_queue: asyncio.Queue = asyncio.Queue()
        self._sent_utterance = False
        if initial_utterance:
            # For backward compatibility - add initial utterance to queue
            self._utterance_queue.put_nowait(initial_utterance)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Small delay to ensure pipeline is ready
        await asyncio.sleep(0.1)

        # Start processing utterances from queue
        asyncio.create_task(self._process_utterances())

    async def _process_utterances(self):
        """Process utterances from the queue continuously."""
        while True:
            try:
                # Wait for next utterance
                utterance = await self._utterance_queue.get()

                # DEBUG: Log utterance being sent
                print(f"DEBUG: TestInputTransport sending utterance: '{utterance}'")

                # Push the utterance as a transcription frame
                await self.push_frame(
                    TranscriptionFrame(
                        text=utterance,
                        user_id="user",
                        timestamp=str(asyncio.get_event_loop().time()),
                    )
                )

                # DEBUG: Log utterance sent
                print("DEBUG: TestInputTransport utterance sent successfully")

                # Mark task as done
                self._utterance_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"DEBUG: Error in TestInputTransport: {e}")
                break

    async def send_utterance(self, utterance: str):
        """Send a new utterance to the pipeline."""
        print(f"DEBUG: TestInputTransport queueing utterance: '{utterance}'")
        await self._utterance_queue.put(utterance)

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
        memory: Union["ConversationMemory", None],
        context: Union["OpenAILLMContext", None],
        input_transport: TestInputTransport,
        **kwargs,
    ):
        super().__init__(
            params=TransportParams(audio_in_enabled=False, audio_out_enabled=False), **kwargs
        )
        self._memory = memory
        self._context = context
        self._input_transport = input_transport
        self._buffer: list[str] = []
        self._response_received = False
        self._pending_futures: list = []  # Track multiple response futures
        self._llm_response_end_count = 0  # Track LLMFullResponseEndFrame count per turn

    async def add_response_future(self, future: asyncio.Future):
        """Add a future that should be resolved when next response completes."""
        print("DEBUG: TestOutputTransport adding response future")
        self._pending_futures.append(future)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # DEBUG: Log all frames being processed
        print(f"DEBUG: TestOutputTransport processing frame: {type(frame).__name__}")

        if isinstance(frame, TextFrame):
            self._buffer.append(frame.text)
            # Don't resolve immediately - wait for second LLMFullResponseEndFrame
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._llm_response_end_count += 1
            count = self._llm_response_end_count
            print(f"DEBUG: TestOutputTransport received LLMFullResponseEndFrame #{count}")

            if self._llm_response_end_count == 1:
                # First LLMFullResponseEndFrame = tool calls done, text response starting
                print("DEBUG: Tool calls complete, waiting for text response...")
                self._response_received = True
                # Signal input transport to end
                await self._input_transport.end_input()
            elif self._llm_response_end_count == 2:
                # Second LLMFullResponseEndFrame = text response done, ready to resolve
                print("DEBUG: Text response complete, resolving futures!")
                if self._pending_futures and self._memory and self._context:
                    self._resolve_pending_futures()
        elif isinstance(frame, EndFrame):
            resp_received = self._response_received
            print(f"DEBUG: TestOutputTransport EndFrame, response_received={resp_received}")
            # EndFrame handling is now primarily for cleanup/logging

    def _resolve_pending_futures(self):
        """Resolve all pending futures with current state."""
        if not (self._pending_futures and self._memory and self._context):
            return

        state = self._memory.to_dict()
        state["messages"] = self._context.messages
        msg_count = len(state['messages'])
        future_count = len(self._pending_futures)
        print(f"DEBUG: Resolving {future_count} futures with {msg_count} messages")

        # Resolve all pending futures
        for future in self._pending_futures:
            if not future.done():
                future.set_result({"state": state})

        # Clear resolved futures and reset state for next turn
        self._pending_futures.clear()
        self._response_received = False
        self._buffer.clear()  # Clear text buffer for next turn
        self._llm_response_end_count = 0  # Reset counter for next turn


class TestTransport(BaseTransport):
    """A combined transport for headless testing."""

    def __init__(
        self,
        initial_utterance: Union[str, None] = None,
        memory: Union["ConversationMemory", None] = None,
        context: Union["OpenAILLMContext", None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input = TestInputTransport(initial_utterance=initial_utterance)
        self._output = TestOutputTransport(
            memory=memory, context=context, input_transport=self._input
        )

    def input(self):
        return self._input

    def output(self):
        return self._output
