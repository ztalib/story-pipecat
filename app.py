import asyncio
import json
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pydantic import BaseModel

from config.config import config
from handlers.story_handlers import (
    BaseHandler,
    ConversationMemory,
    StoryCaptureHandler,
    StoryFlowHandler,
)
from tests.test_transport import TestTransport


# NEW: Pydantic model for the test endpoint request
class TestChatRequest(BaseModel):
    utterance: str
    state: dict | None = None


load_dotenv()

# Suppress noisy pipecat warnings in test mode
class PipecatWarningFilter(logging.Filter):
    def filter(self, record):
        # Suppress "destination [None] not registered" warnings
        if "destination [None] not registered" in record.getMessage():
            return False
        return True

pipecat_logger = logging.getLogger("pipecat.transports.base_output")
pipecat_logger.addFilter(PipecatWarningFilter())

# -----------------
# WebSocket Connection Manager
# -----------------


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()

# -----------------
# Custom Frame Processor for UI Updates
# -----------------


# Removed verbose DebugFrameProcessor - kept key diagnostic logging in test endpoint


class SendTranscriptionToClient(FrameProcessor):
    """A simple class to send transcriptions to the client."""

    def __init__(self, speaker: str, log_filename: str):
        super().__init__()
        self._speaker = speaker
        self._log_filename = log_filename
        self._buffer: list[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (TranscriptionFrame, TextFrame)):
            text = frame.text
            if text.strip():
                if self._speaker == "USER" and isinstance(frame, TranscriptionFrame):
                    print(f"{self._speaker} SAID: {text}")
                    with open(self._log_filename, "a") as f:
                        f.write(f"{self._speaker}: {text}\n")
                        f.flush()
                elif self._speaker == "BOT" and isinstance(frame, TextFrame):
                    self._buffer.append(frame.text)

                message = {"type": "transcript", "speaker": self._speaker.lower(), "text": text}
                await manager.broadcast(json.dumps(message))
        elif isinstance(frame, LLMFullResponseEndFrame):
            full_response = "".join(self._buffer).strip()
            if full_response:
                print(f"BOT SAID: {full_response}")
                with open(self._log_filename, "a") as f:
                    f.write(f"{self._speaker}: {full_response}\n")
                    f.flush()
            self._buffer = []

        await self.push_frame(frame, direction)


# -----------------
# Bot Design: Tools, Handlers, and Pipeline
# -----------------


def create_bot_logic(
    convo_dir: str, initial_state: dict | None = None
) -> tuple[OpenAILLMService, OpenAILLMContext, ConversationMemory]:
    """Create and configure the core bot components (memory, handlers, LLM)."""
    data_filepath = os.path.join(convo_dir, "extracted_data.json")
    messages = initial_state.get("messages", []) if initial_state else []

    # 1. Create a single, shared memory object for this conversation
    memory = ConversationMemory()
    if initial_state:
        memory.extracted_elements = initial_state.get("extracted_elements", {})
        memory.conversation_state = initial_state.get("conversation_state", "opening")

    # 2. Create instances of our handlers, passing in the SAME memory object
    capture_handler = StoryCaptureHandler(data_filepath=data_filepath, memory=memory)
    flow_handler = StoryFlowHandler(data_filepath=data_filepath, memory=memory)
    all_handlers: list[BaseHandler] = [capture_handler, flow_handler]

    # 3. Create the ToolsSchema by getting the schema from each handler class
    tools = ToolsSchema(standard_tools=[handler.get_schema() for handler in all_handlers])

    # 4. Create the LLM context, configured with a system prompt and our tools
    # If we have a message history, use it. Otherwise, start with the system prompt.
    if not messages:
        messages = [{"role": "system", "content": config.system_prompt}]

    context = OpenAILLMContext(messages=messages, tools=tools)

    # 5. Create the LLM Service
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model=config.llm_model)

    # 6. Register the handler methods by name
    for handler in all_handlers:
        llm.register_function(handler.NAME, handler.handle_request)

    return llm, context, memory


# --- Main Bot Logic ---


async def run_bot(transport, convo_dir: str):
    log_filename = os.path.join(convo_dir, "transcript.log")
    llm, context, _ = create_bot_logic(convo_dir)

    # Create other services and the context aggregator
    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))
    tts = OpenAITTSService(api_key=os.getenv("OPENAI_API_KEY"))
    context_aggregator = llm.create_context_aggregator(context)
    send_user_transcripts = SendTranscriptionToClient(speaker="USER", log_filename=log_filename)
    send_bot_transcripts = SendTranscriptionToClient(speaker="BOT", log_filename=log_filename)

    # 8. Assemble the final pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            send_user_transcripts,
            context_aggregator.user(),
            llm,
            send_bot_transcripts,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        pass  # The bot will wait for the user to speak first.

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        # The memory object is now managed by the ConversationMemory class instance
        # and will be garbage collected automatically.
        # We still need to cancel the task to gracefully shut down the pipeline.
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)


# -----------------
# FastAPI server
# -----------------

# --- FastAPI App ---
app = FastAPI()
app.mount("/ui", StaticFiles(directory="ui"), name="ui")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/start-bot")
async def start_bot(request: dict, background_tasks: BackgroundTasks):
    """Start bot in a Daily room"""
    room_url = request.get("room_url")
    token = request.get("token")

    if not room_url:
        return {"error": "room_url is required"}

    # Create a unique directory for this conversation
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")
    convo_dir = f"conversations/{date_str}"
    if not os.path.exists(convo_dir):
        os.makedirs(convo_dir)

    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="Voice AI Bot",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    background_tasks.add_task(run_bot, transport, convo_dir=convo_dir)

    return {"status": "Bot started", "room_url": room_url}


@app.post("/test/chat")
async def test_chat(request: TestChatRequest):
    """A text-only endpoint for testing the bot's logic."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    # 1. Create a temporary directory for this test run.
    # We still need this for the handler to write its JSON file.
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S_%f")
    convo_dir = f"conversations/test_{date_str}"
    if not os.path.exists(convo_dir):
        os.makedirs(convo_dir)

    # 2. Get the bot's core components, initialized with the provided state.
    # The memory object is the key here, as it's shared with the transport.
    llm, context, memory = create_bot_logic(convo_dir, initial_state=request.state)

    # 3. Create our headless test transport.
    transport = TestTransport(
        utterance=request.utterance,
        future=future,
        memory=memory,
        context=context,
    )

    # 4. Assemble the full pipeline, but without STT and TTS.
    log_filename = os.path.join(convo_dir, "transcript.log")
    send_user_transcripts = SendTranscriptionToClient(speaker="USER", log_filename=log_filename)
    send_bot_transcripts = SendTranscriptionToClient(speaker="BOT", log_filename=log_filename)
    context_aggregator = llm.create_context_aggregator(context)

    # Keep basic diagnostic info for test troubleshooting
    tools_count = len(context.tools) if context.tools else 0
    api_status = '✓' if os.getenv('OPENAI_API_KEY') else '✗'
    print(f"Test setup: Messages={len(context.messages)}, Tools={tools_count}, API={api_status}")

    pipeline = Pipeline(
        [
            transport.input(),
            send_user_transcripts,
            context_aggregator.user(),
            llm,
            send_bot_transcripts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    # 5. Run the pipeline and wait for completion.
    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    # Run both the pipeline and wait for the result concurrently
    pipeline_task = asyncio.create_task(runner.run(task))

    try:
        # Wait for the future to be resolved by the transport (with timeout)
        result = await asyncio.wait_for(future, timeout=30.0)
        return result["state"]
    except asyncio.TimeoutError:
        return {"error": "Test timed out after 30 seconds"}
    finally:
        # Ensure pipeline is properly cancelled
        if not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass


@app.post("/create-room")
async def create_daily_room():
    """Create a Daily room for the conversation"""
    import aiohttp

    daily_api_key = os.getenv("DAILY_API_KEY")
    if not daily_api_key:
        return {"error": "DAILY_API_KEY not configured"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.daily.co/v1/rooms",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {daily_api_key}",
                },
                json={
                    "properties": {
                        "exp": int(datetime.now().timestamp()) + 30 * 60,
                        "enable_chat": False,
                        "enable_screenshare": False,
                        "start_audio_off": False,
                        "start_video_off": True,
                        "owner_only_broadcast": False,
                    }
                },
            ) as resp:
                if resp.status in [200, 201]:
                    return await resp.json()
                else:
                    response_text = await resp.text()
                    return {"error": f"Failed to create room: {response_text}"}
    except Exception as e:
        return {"error": f"Exception creating room: {str(e)}"}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("ui/index.html") as f:
        return HTMLResponse(content=f.read())
