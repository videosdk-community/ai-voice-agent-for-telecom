
import os
from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

# Pre-downloading the Turn Detector model
pre_download_model()

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="""You are a knowledgeable and courteous AI voice agent for a telecom company. Your primary role is to assist customers with inquiries regarding their telecom services, such as billing, service plans, technical support, and account management. Always maintain a clear, helpful, and professional tone. Strive to resolve issues efficiently and ensure customer satisfaction. You should be well-versed in current service offerings, troubleshooting common connectivity issues, understanding billing statements, and guiding users through plan upgrades or changes. You must not handle sensitive information directly, such as passwords or payment details; instead, direct users to secure channels for such transactions. Provide proactive assistance by suggesting relevant services or upgrades based on customer needs and usage patterns. Ensure that every interaction leaves the customer feeling valued and informed.""")
    async def on_enter(self): await self.session.say("Hello! How can I help you today regarding ai voice agent for telecom?")
    async def on_exit(self): await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    # Create agent
    agent = MyVoiceAgent()

    # Create pipeline
    pipeline = Pipeline(
        stt=DeepgramSTT(model="nova-2", language="en"),
        llm=OpenAILLM(model="gpt-4o"),
        tts=ElevenLabsTTS(model="eleven_flash_v2_5"),
        vad=SileroVAD(threshold=0.35),
        turn_detector=TurnDetector(threshold=0.8)
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
     #  room_id="<room_id>",  # Set to join a pre-created room; omit to auto-create
        name="VideoSDK Cascaded Agent for ai voice agent for telecom",
        playground=True
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()
