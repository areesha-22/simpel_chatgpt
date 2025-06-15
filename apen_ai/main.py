import os
import asyncio
from dotenv import load_dotenv
from agents import AsyncOpenAI, Agent, Runner, OpenAIChatCompletionsModel,RunConfig

load_dotenv()

async def main():
    MODEL_NAME = "gemini-2.0-flash"
    GEMINI_API = os.getenv("GEMINI_API")

    external_client = AsyncOpenAI(
        api_key=GEMINI_API,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    model = OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=external_client
    )
    config=RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )
    assistant = Agent(
        name="assistant",
        instructions="Your job is to solve the queries.",
        model=model
    )

    result = await Runner.run(assistant, "tell me about how AI is important?",run_config=config)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
