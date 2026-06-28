import asyncio
import sys
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents.minimax_client import call_minimax, build_image_content
from agents import vehicle_prior_agent

async def main():
    vehicle_info = {"brand": "蔚来", "model": "ES8", "year": "2019"}
    vehicle_prior = await vehicle_prior_agent(vehicle_info)

    folder = "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868"
    batch_files = ['蔚来ES8_25.png', '蔚来ES8_26.png', '蔚来ES8_27.png']
    photos = [{"id": f, "path": f"{folder}/{f}"} for f in batch_files]

    topology = __import__('json').dumps(vehicle_prior.get("topology", {}), ensure_ascii=False, indent=2)
    anchors = __import__('json').dumps(vehicle_prior.get("key_anchors", {}), ensure_ascii=False, indent=2)

    system_prompt = f"""你是汽车照片视角识别专家。给定 蔚来 ES8 的车型先验信息，判断每张照片拍摄的是车辆的哪个区域。

车型先验：
{topology}

关键锚点：
{anchors}

对每张照片，输出一个 JSON 对象：
{{
  "photo_id": "照片编号",
  "location": "车头/车尾/左侧/右侧/车顶/内饰/无法定位",
  "location_detail": "更具体的描述，如车头左前45度、车尾正后方等",
  "primary_anchor": "用于定位的主要锚点",
  "confidence": "high/medium/low",
  "reason": "判断理由",
  "visible_parts": ["可见部件1", "可见部件2"]
}}

必须返回一个 JSON 数组，每个元素对应一张照片，顺序与输入照片一致。
"""

    content = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": "以下是待识别的照片，请逐张分析："},
    ]
    for photo in photos:
        content.append({"type": "text", "text": f"照片编号: {photo['id']}"})
        content.append(build_image_content(photo["path"]))

    raw = await call_minimax([{"role": "user", "content": content}], temperature=0.1, max_tokens=3000)
    print("="*80)
    print("RAW OUTPUT:")
    print("="*80)
    print(raw)
    print("="*80)
    print("REPR:")
    print(repr(raw))

asyncio.run(main())
