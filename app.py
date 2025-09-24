import asyncio
import os
import logging
from typing import Dict, List
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMRunFrame, TranscriptionFrame, TextFrame, DataFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipecat")

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
# Custom Frame Processor
# -----------------

# Was Working - don't Delete - for reference
# class SendTranscriptionToClient(FrameProcessor):
#     def __init__(self, speaker: str):
#         super().__init__()
#         self._speaker = speaker

#     async def process_frame(self, frame, direction):
#         await super().process_frame(frame, direction)
#         if isinstance(frame, (TranscriptionFrame, TextFrame)):
#             print(f"{self._speaker} SAID: {frame.text}")
#         await self.push_frame(frame, direction)  # Keep this - it's needed!


class SendTranscriptionToClient(FrameProcessor):
    def __init__(self, speaker: str, log_filename: str):
        super().__init__()
        self._speaker = speaker
        self._log_filename = log_filename
        self._buffer = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (TranscriptionFrame, TextFrame)):
            # print(f"{self._speaker} SAID: {frame.text}")
            text = frame.text
            if text.strip():
                # 1. Log to console for debugging / 2. Store to file
                if self._speaker == "USER" and isinstance(frame, TranscriptionFrame):       
                    print(f"{self._speaker} SAID: {text}")
                    with open(self._log_filename, "a") as f:
                        f.write(f"{self._speaker}: {text}\n")
                        f.flush()
                elif self._speaker == "BOT" and isinstance(frame, TextFrame):
                    self._buffer.append(frame.text)

                # 3. Broadcast to UI
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
# Bot logic
# -----------------

async def run_bot(transport, log_filename: str):
    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))
    tts = OpenAITTSService(api_key=os.getenv("OPENAI_API_KEY"))
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant. Respond naturally and keep your answers conversational.",
        },
    ]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    send_user_transcripts = SendTranscriptionToClient(speaker="USER", log_filename=log_filename)
    send_bot_transcripts = SendTranscriptionToClient(speaker="BOT", log_filename=log_filename)

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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)

async def bot(runner_args: RunnerArguments, log_filename: str):
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        transport = SmallWebRTCTransport(
            webrtc_connection=runner_args.webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                data_channel_enabled=False,
            ),
        )
        await run_bot(transport, log_filename)


# -----------------
# FastAPI server
# -----------------

app = FastAPI()
pcs_map: Dict[str, SmallWebRTCConnection] = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/webrtc")
async def webrtc_offer(request: dict, background_tasks: BackgroundTasks):
    # Create a unique filename for this conversation
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")
    if not os.path.exists("conversations"):
        os.makedirs("conversations")
    log_filename = f"conversations/convo_{date_str}.log"

    pipecat_connection = SmallWebRTCConnection()
    await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

    @pipecat_connection.event_handler("closed")
    async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
        logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
        pcs_map.pop(webrtc_connection.pc_id, None)

    runner_args = SmallWebRTCRunnerArguments(webrtc_connection=pipecat_connection)
    background_tasks.add_task(bot, runner_args, log_filename=log_filename)

    answer = pipecat_connection.get_answer()
    pcs_map[answer["pc_id"]] = pipecat_connection
    return answer

# Serve the custom UI
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("ui/index.html", "r") as f:
        return HTMLResponse(content=f.read())
