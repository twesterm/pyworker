from vastai import Serverless
import asyncio

ENDPOINT_NAME = "my-tgi-endpoint" # Change this to match your endpoint name
MAX_TOKENS = 1024
PROMPT = "Think step by step: Tell me about the Python programming language."

async def call_generate(client: Serverless) -> None:
    endpoint = await client.get_endpoint(name=ENDPOINT_NAME)

    payload = {
        "inputs": PROMPT,
        "parameters": {
            "max_new_tokens": MAX_TOKENS,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    resp = await endpoint.request("/generate", payload, cost=MAX_TOKENS)

    print(resp["response"]["generated_text"])


async def call_generate_stream(client: Serverless) -> None:
    endpoint = await client.get_endpoint(name=ENDPOINT_NAME)

    payload = {
        "inputs": PROMPT,
        "parameters": {
            "max_new_tokens": MAX_TOKENS,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False,
        }
    }

    resp = await endpoint.request(
        "/generate_stream",
        payload,
        cost=MAX_TOKENS,
        stream=True,
    )
    stream = resp["response"]

    printed_answer = False
    async for event in stream:
        tok = (event.get("token") or {}).get("text")
        if tok:
            if not printed_answer:
                printed_answer = True
                print("Answer:\n", end="", flush=True)
            print(tok, end="", flush=True)

async def main():
    async with Serverless() as client:
        await call_generate(client)
        await call_generate_stream(client)

if __name__ == "__main__":
    asyncio.run(main())
