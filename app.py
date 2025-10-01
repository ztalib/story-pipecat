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
    utterance: str | None = None  # Single-turn mode
    utterances: list[str] | None = None  # Multi-turn batch mode
    state: dict | None = None
    session_id: str | None = None  # Only used for single-turn mode


load_dotenv()

# Global storage for active test sessions
active_test_sessions: dict[str, dict] = {}

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
        
        # USER processor only handles TranscriptionFrames (user speech)
        if self._speaker == "USER" and isinstance(frame, TranscriptionFrame):
            text = frame.text
            if text.strip():
                print(f"{self._speaker} SAID: {text}")
                with open(self._log_filename, "a") as f:
                    f.write(f"{self._speaker}: {text}\n")
                    f.flush()
                message = {"type": "transcript", "speaker": self._speaker.lower(), "text": text}
                await manager.broadcast(json.dumps(message))
        
        # BOT processor only handles TextFrames (bot responses)
        elif self._speaker == "BOT" and isinstance(frame, TextFrame):
            text = frame.text
            if text.strip():
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
        # Have Clio greet the user first with a welcome message
        welcome_message = "Hi, I'm Clio, what story would you like to share today?"
        await task.queue_frame(TextFrame(welcome_message))

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        # The memory object is now managed by the ConversationMemory class instance
        # and will be garbage collected automatically.
        # We still need to cancel the task to gracefully shut down the pipeline.
        await task.cancel()

    # Simple timeout implementation
    timeout_seconds = config.session_timeout_seconds
    print(f"Session timeout set to {timeout_seconds} seconds")
    
    runner = PipelineRunner()
    
    try:
        # Run the pipeline with timeout
        await asyncio.wait_for(runner.run(task), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print(f"Session timed out after {timeout_seconds} seconds")
        await task.cancel()
        await transport.close()


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


@app.get("/config")
async def get_config():
    """Get application configuration for the client"""
    return {
        "session_timeout_seconds": config.session_timeout_seconds
    }

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


async def process_multi_turn_batch(
    utterances: list[str], initial_state: dict | None = None
) -> dict:
    """Process multiple utterances in sequence through a single persistent pipeline."""
    if initial_state is None:
        initial_state = {}

    # 1. Create temporary directory for this batch test
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S_%f")
    convo_dir = f"conversations/test_{date_str}"
    if not os.path.exists(convo_dir):
        os.makedirs(convo_dir)

    # 2. Get the bot's core components
    llm, context, memory = create_bot_logic(convo_dir, initial_state=initial_state)

    # 3. Create transport and pipeline
    transport = TestTransport(memory=memory, context=context)

    # 4. Setup pipeline components
    log_filename = os.path.join(convo_dir, "transcript.log")
    send_user_transcripts = SendTranscriptionToClient(speaker="USER", log_filename=log_filename)
    send_bot_transcripts = SendTranscriptionToClient(speaker="BOT", log_filename=log_filename)
    context_aggregator = llm.create_context_aggregator(context)

    # 5. Assemble the pipeline
    pipeline = Pipeline([
        transport.input(),
        send_user_transcripts,
        context_aggregator.user(),
        llm,
        send_bot_transcripts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    pipeline_task = asyncio.create_task(runner.run(task))

    turn_results = []
    current_state = initial_state

    try:
        print(f"DEBUG: Starting batch processing of {len(utterances)} utterances")

        for i, utterance in enumerate(utterances):
            print(f"DEBUG: Processing turn {i+1}: '{utterance}'")

            # Create future for this turn's response
            loop = asyncio.get_running_loop()
            future = loop.create_future()

            # Add future to output transport
            await transport.output().add_response_future(future)

            # Send utterance to pipeline
            await transport.input().send_utterance(utterance)

            # Wait for response (with reasonable timeout)
            try:
                result = await asyncio.wait_for(future, timeout=30.0)
                current_state = result["state"]

                # Store this turn's result
                turn_results.append({
                    "turn": i + 1,
                    "utterance": utterance,
                    "state": current_state
                })

                print(f"DEBUG: Turn {i+1} completed successfully")

                # Add small delay between turns to ensure proper separation
                if i < len(utterances) - 1:  # Don't delay after the last utterance
                    await asyncio.sleep(0.1)

            except asyncio.TimeoutError:
                error_msg = f"Turn {i+1} timed out after 30 seconds"
                print(f"DEBUG: {error_msg}")
                return {"error": error_msg, "completed_turns": turn_results}

    except Exception as e:
        error_msg = f"Batch processing failed: {str(e)}"
        print(f"DEBUG: {error_msg}")
        return {"error": error_msg, "completed_turns": turn_results}

    finally:
        # Clean up pipeline
        if not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass

    # Return results with intermediate states for testing
    return {
        "turn_results": turn_results,
        "final_state": current_state,
        "total_turns": len(utterances)
    }


@app.post("/test/chat")
async def test_chat(request: TestChatRequest):
    """A text-only endpoint for testing the bot's logic with batch and session support."""

    # Validate request - must have either utterance or utterances
    if not request.utterance and not request.utterances:
        return {
            "error": (
                "Must provide either 'utterance' (single-turn) or 'utterances' (multi-turn batch)"
            )
        }

    if request.utterance and request.utterances:
        return {"error": "Cannot provide both 'utterance' and 'utterances' - choose one mode"}

    # Multi-turn batch mode
    if request.utterances:
        print(f"DEBUG: test_chat called in BATCH mode with {len(request.utterances)} utterances")
        return await process_multi_turn_batch(request.utterances, request.state)

    # Single-turn mode (existing logic)
    session_id = request.session_id
    print(f"DEBUG: test_chat called in SINGLE-TURN mode with session_id={session_id}")
    print(f"DEBUG: Active sessions: {list(active_test_sessions.keys())}")

    if session_id and session_id in active_test_sessions:
        # Continue existing session
        print(f"DEBUG: Continuing session {session_id}")
        session_data = active_test_sessions[session_id]
        context = session_data["context"]
        memory = session_data["memory"]
        transport = session_data["transport"]
        convo_dir = session_data["convo_dir"]

        # DEBUG: Log context state before pipeline processes the message
        print(f"DEBUG: Context messages before: {len(context.messages)}")

        # Create future for this response
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Add future to output transport so it gets resolved when response comes
        await transport.output().add_response_future(future)

        # Send new utterance to existing pipeline
        await transport.input().send_utterance(request.utterance)

        # DEBUG: Log utterance sent to existing pipeline
        print(f"DEBUG: Sent utterance to existing pipeline: '{request.utterance}'")

        # Get the pipeline task from session for cleanup
        pipeline_task = session_data["pipeline_task"]

    else:
        # Create new session (or single-turn test)
        print(f"DEBUG: Creating new session (session_id={session_id})")
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # 1. Create a temporary directory for this test run.
        now = datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S_%f")
        convo_dir = f"conversations/test_{date_str}"
        if not os.path.exists(convo_dir):
            os.makedirs(convo_dir)

        # 2. Get the bot's core components, initialized with the provided state.
        llm, context, memory = create_bot_logic(convo_dir, initial_state=request.state)

        # 3. Create our headless test transport.
        transport = TestTransport(
            initial_utterance=request.utterance,
            memory=memory,
            context=context,
        )

        # 4. Setup pipeline components
        log_filename = os.path.join(convo_dir, "transcript.log")
        send_user_transcripts = SendTranscriptionToClient(speaker="USER", log_filename=log_filename)
        send_bot_transcripts = SendTranscriptionToClient(speaker="BOT", log_filename=log_filename)
        context_aggregator = llm.create_context_aggregator(context)

        # Keep basic diagnostic info for test troubleshooting
        tools_count = len(context.tools) if context.tools else 0
        api_status = '✓' if os.getenv('OPENAI_API_KEY') else '✗'
        msg_count = len(context.messages)
        print(f"Test setup: Messages={msg_count}, Tools={tools_count}, API={api_status}")

        # Create future for first response
        future = loop.create_future()
        await transport.output().add_response_future(future)

        # Assemble the pipeline
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

        task = PipelineTask(pipeline)
        runner = PipelineRunner()
        pipeline_task = asyncio.create_task(runner.run(task))

        # Store session if session_id provided
        if session_id:
            active_test_sessions[session_id] = {
                "context": context,
                "memory": memory,
                "llm": llm,
                "convo_dir": convo_dir,
                "transport": transport,
                "pipeline": pipeline,
                "pipeline_task": pipeline_task,
                "created_at": now
            }

    # Pipeline execution - this section is shared for both new and continuing sessions

    try:
        # DEBUG: Log pipeline execution start
        print(f"DEBUG: Starting pipeline execution for session {session_id}")

        # Wait for the future to be resolved by the transport (with timeout)
        result = await asyncio.wait_for(future, timeout=30.0)

        # DEBUG: Log pipeline completion
        print(f"DEBUG: Pipeline completed for session {session_id}")
        print(f"DEBUG: Result keys: {list(result.keys()) if result else 'None'}")

        return result["state"]
    except asyncio.TimeoutError:
        return {"error": "Test timed out after 30 seconds"}
    finally:
        # Only cancel pipeline for single-turn tests (no session_id)
        # For multi-turn tests, keep the pipeline alive across turns
        if not session_id and not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass


@app.post("/test/cleanup-session")
async def cleanup_session(session_id: str):
    """Clean up a test session."""
    if session_id in active_test_sessions:
        del active_test_sessions[session_id]
        return {"status": "Session cleaned up", "session_id": session_id}
    else:
        return {"status": "Session not found", "session_id": session_id}


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
