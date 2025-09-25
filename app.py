import os
import logging
from typing import List
import json
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMFullResponseEndFrame, TranscriptionFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from config.config import config
from handlers.story_handlers import StoryCaptureHandler, StoryFlowHandler, ConversationMemory, BaseHandler

load_dotenv()

# -----------------
# WebSocket Connection Manager
# -----------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

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

class SendTranscriptionToClient(FrameProcessor):
    """A simple class to send transcriptions to the client."""
    def __init__(self, speaker: str, log_filename: str):
        super().__init__()
        self._speaker = speaker
        self._log_filename = log_filename
        self._buffer: List[str] = []

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

                message = {
                    "type": "transcript",
                    "speaker": self._speaker.lower(),
                    "text": text
                }
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

# --- Main Bot Logic ---

async def run_bot(transport, convo_dir: str):
    log_filename = os.path.join(convo_dir, "transcript.log")
    data_filepath = os.path.join(convo_dir, "extracted_data.json")

    # 1. Create a single, shared memory object for this conversation
    memory = ConversationMemory()

    # 2. Create instances of our handlers, passing in the SAME memory object
    capture_handler = StoryCaptureHandler(data_filepath=data_filepath, memory=memory)
    flow_handler = StoryFlowHandler(data_filepath=data_filepath, memory=memory)
    all_handlers: List[BaseHandler] = [capture_handler, flow_handler]

    # 3. Create the ToolsSchema by getting the schema from each handler class
    tools = ToolsSchema(
        standard_tools=[handler.get_schema() for handler in all_handlers]
    )

    # 4. Create the LLM context, configured with a system prompt and our tools
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": config.system_prompt}], 
        tools=tools
    )

    # 5. Create the LLM Service
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"), 
        model=config.llm_model
    )

    # 6. Register the handler methods by name
    for handler in all_handlers:
        llm.register_function(handler.NAME, handler.handle_request)

    # 7. Create other services and the context aggregator
    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))
    tts = OpenAITTSService(api_key=os.getenv("OPENAI_API_KEY"))
    context_aggregator = llm.create_context_aggregator(context)
    send_user_transcripts = SendTranscriptionToClient(speaker="USER", log_filename=log_filename)
    send_bot_transcripts = SendTranscriptionToClient(speaker="BOT", log_filename=log_filename)

    # 8. Assemble the final pipeline
    pipeline = Pipeline([
        transport.input(),
        stt,
        send_user_transcripts,
        context_aggregator.user(),
        llm,
        send_bot_transcripts,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        pass # The bot will wait for the user to speak first.

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
                    "Authorization": f"Bearer {daily_api_key}"
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
                }
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
    with open("ui/index.html", "r") as f:
        return HTMLResponse(content=f.read())
