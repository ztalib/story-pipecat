# Voice AI Framework

A simple voice AI application built with Pipecat, featuring real-time voice conversation with clean separation between voice processing and UI updates.

## Architecture

```
User → WebRTC (Audio) → Pipecat Pipeline → WebSocket (Transcripts) → UI
```

### Core Components

- **Pipecat Pipeline**: STT → LLM → TTS voice processing
- **WebRTC Transport**: Real-time audio streaming
- **WebSocket Communication**: Real-time transcript updates to UI
- **Conversation Logging**: Timestamped conversation files

## Quick Start

### Prerequisites
```bash
# Set up environment
source venv/bin/activate
pip install -r requirements.txt
ruff check . --fix 

# Set your OpenAI API key
export OPENAI_API_KEY="your-key"
```

### Run the Application
```bash
python -m uvicorn app:app
```

Open http://localhost:8000 in your browser

## Features

- **Real-time Voice Conversation**: Full duplex audio streaming via WebRTC
- **Live Transcript Display**: Real-time text streaming to the UI via WebSocket
- **Conversation Logging**: Automatic logging to timestamped files in `conversations/` directory
- **Clean Architecture**: Voice pipeline separated from UI concerns
- **Responsive UI**: Modern web interface with real-time updates

## Configuration

### Environment Variables
- `OPENAI_API_KEY` - Required for OpenAI STT, LLM, and TTS services

## Usage

1. **Start the server**: Run `python -m uvicorn app:app`
2. **Open the UI**: Navigate to http://localhost:8000
3. **Start conversation**: Click "Start Conversation" and allow microphone access
4. **Speak naturally**: The bot will respond with voice and display transcripts in real-time
5. **View logs**: Check the `conversations/` directory for timestamped conversation files

## Directory Structure

```
story-legacy-pipecat/
├── app.py                     # Main FastAPI application
├── ui/                        # Web UI files
│   ├── index.html            # Main web interface
│   └── config.js             # UI configuration
├── conversations/             # Conversation log files
│   └── convo_YYYYMMDD_HHMM.log
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Technical Details

- **Backend**: FastAPI with Pipecat pipeline
- **Frontend**: Vanilla JavaScript with WebRTC and WebSocket
- **Voice Processing**: OpenAI STT, GPT, and TTS
- **Real-time Communication**: WebRTC for audio, WebSocket for transcripts
- **Logging**: Automatic conversation persistence with file flushing

## Development

The application uses a clean separation of concerns:
- Voice processing happens in the Pipecat pipeline
- UI updates are handled via WebSocket broadcasts
- Conversation logging is managed separately from the voice pipeline
- Each conversation session gets its own timestamped log file