import asyncio
import json
import sys
sys.path.insert(0, '/Users/sky/Downloads/vehicle_damage_assessment')
from agents.minimax_client import call_minimax, build_image_content

async def main():
    vehicle_prior = {
        "vehicle": "蔚来 ES8 2019款",
        "topology": {"front": "X-bar 前脸", "rear": "贯穿尾灯"},
        "key_anchors": {"front": ["NIO logo"], "rear": ["贯穿尾灯"]}
    }
    photos = [
        {"id": "02", "path": "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_02.png"},
    ]
    
    system_prompt = f"""判断照片拍摄的是车辆哪个区域。输出 JSON 数组。车型：蔚来 ES8"""
    content = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": "照片编号: 02"},
        build_image_content(photos[0]["path"]),
    ]
    messages = [{"role": "user", "content": content}]
    
    raw = await call_minimax(messages, temperature=0.1, max_tokens=1000)
    print("RAW OUTPUT:")
    print(repr(raw))
    print("\nCLEANED OUTPUT:")
    print(raw)

asyncio.run(main())
