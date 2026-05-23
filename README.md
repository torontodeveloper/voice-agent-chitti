# Voice Agent

A real-time voice AI agent you can call on the phone. Built with Pipecat, Twilio, Deepgram, Cartesia, and OpenAI.

## How it works

```
Twilio (phone call) → Deepgram (STT) → Silero VAD → GPT-4.1 (LLM) → Cartesia (TTS) → Twilio (back to caller)
```

1. Twilio receives the incoming call and streams audio over WebSocket as Mulaw 8kHz
2. Deepgram transcribes the audio stream in real time
3. Silero VAD detects end-of-speech so the agent knows when to respond
4. GPT-4.1 generates a response, streaming tokens as they arrive
5. Cartesia synthesizes speech from the token stream with low latency
6. Audio streams back through the WebSocket to the caller

End-to-end latency: under 1 second.

## Stack

| Component | Purpose |
|---|---|
| [Pipecat](https://github.com/pipecat-ai/pipecat) | Real-time audio pipeline orchestration |
| [Twilio](https://www.twilio.com/) | Telephony — inbound call handling, WebSocket audio streaming |
| [Deepgram](https://deepgram.com/) | Speech-to-text (streaming) |
| [Cartesia](https://cartesia.ai/) | Text-to-speech (low latency) |
| [OpenAI GPT-4.1](https://openai.com/) | LLM |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Voice activity detection |
| FastAPI | WebSocket server |

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Twilio account with a phone number
- Deepgram API key
- Cartesia API key
- OpenAI API key

### Install

```bash
uv sync
```

### Configure

Copy `.env.example` to `.env` and fill in your keys:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
CARTESIA_VOICE_ID=71a7ad14-091c-4e8e-a314-022ece01c121
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
```

### Run

```bash
uv run pipecat bot.py
```

### Twilio webhook

Set your Twilio phone number's "A call comes in" webhook to:

```
https://your-domain/ws
```

The server handles the WebSocket handshake and routes the call to the bot pipeline.

## Project structure

```
bot.py       # Pipeline definition — STT, VAD, LLM, TTS, transport
main.py      # Entry point
pyproject.toml
```
