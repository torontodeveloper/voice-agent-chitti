import logging
import os

from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import RTVIObserverParams
from pipecat.runner.run import main
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat_whisker import WhiskerObserver
from pypdf import PdfReader

load_dotenv()


transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


cv_details = PdfReader("KevinKakolla_SeniorAIEngineer.pdf")
linkedin_details = PdfReader("Profile.pdf")
cv_details_pages = len(cv_details.pages)
linkedin_details_pages = len(linkedin_details.pages)
summary = """You are Kevin Kakolla's AI assistant representing him to recruiters.
  Answer questions about his background confidently and accurately based
  on his CV and LinkedIn. Keep answers concise since this is a voice call.
  Do not use bullet points, markdown, or formatting. If asked about
  availability or salary, say Kevin is open to discussing details directly"""
cv_pages = ""
for item in range(cv_details_pages):
    page = cv_details.pages[item]
    text = page.extract_text()
    cv_pages += text
linkedin_pages = ""
for item in range(linkedin_details_pages):
    page = linkedin_details.pages[item]
    text = page.extract_text()
    linkedin_pages += text

summary += cv_pages + linkedin_pages
# print(f"CV is {summary}")

# print(f"summary is {summary}")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice=os.getenv(
                "CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"
            ),
        ),
    )
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model="gpt-5.4-mini",
            system_instruction=f"{summary}",
        ),
    )

    # llm = OpenRouterLLMService(
    #     api_key=os.environ["OPENROUTER_API_KEY"],
    #     model="meta-llama/llama-3.3-70b-instruct:free",
    #     settings=OpenRouterLLMService.Settings(
    #         system_instruction=summary,
    #     ),
    # )

    # llm = GroqLLMService(
    #     api_key=os.environ["GROQ_API_KEY"],
    #     settings=GroqLLMService.Settings(
    #         system_instruction=summary,
    #     ),
    #     model="llama-3.3-70b-versatile",
    # )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logging.info("Client connected*******")
        print("Client connected*******")
        # Kick off the conversation.
        context.add_message(
            {"role": "user", "content": "Please introduce yourself to the user."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        print("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        rtvi_observer_params=RTVIObserverParams(
            bot_llm_enabled=False,
            metrics_enabled=False,
        ),
    )
    task.add_observer(WhiskerObserver(task.pipeline))
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point called by the development runner."""

    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logging.info(f"Auto-detected transport: {transport_type}")

    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )
    # Create your transport based on the runner arguments
    # transport = WebSocketRunnerArguments(
    #     params=TransportParams(
    #         audio_in_enabled=True,
    #         audio_out_enabled=True,
    #     ),
    #     webrtc_connection=runner_args.webrtc_connection,
    # )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    # Run your bot logic
    await run_bot(transport, transport_params)


if __name__ == "__main__":
    main()
