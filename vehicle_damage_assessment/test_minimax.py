import asyncio
import json
import os
from agents.minimax_client import call_minimax

async def main():
    messages = [
        {"role": "system", "content": "你是一个 helpful assistant。请只输出 JSON。"},
        {"role": "user", "content": "请输出 {'hello': 'world'} 这个 JSON。"}
    ]
    try:
        resp = await call_minimax(messages, temperature=0.1, max_tokens=100)
        print("Response:")
        print(resp)
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
