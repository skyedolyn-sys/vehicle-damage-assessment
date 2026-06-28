import asyncio
import aiohttp
import ssl
import json

async def main():
    api_key = "sk-cp-Udb3wiU4tiqMJYNSWnstJftzFDV4DNYVAs7WytDRv9-Gz3ILlk7iKbKshUrAkpp-EZD-QWAh8lY_AGsFUnHMRz9teyfSzCJ9rOa0TfIWzloHOPu-Q_nhk-U"
    url = "https://api.minimaxi.com/v1/chat/completions"
    
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    
    payload = {
        "model": "MiniMax-M2.7",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in JSON format."}
        ],
        "temperature": 0.1,
        "max_tokens": 100,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            ssl=ssl_ctx,
        ) as resp:
            print(f"Status: {resp.status}")
            text = await resp.text()
            print(f"Raw response: {text[:2000]}")

asyncio.run(main())
