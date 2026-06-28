import asyncio
import json
import sys
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents.minimax_client import call_minimax, build_image_content, extract_json
from config import PARTS_CATALOG

async def test_with_photos(paths: list[str], count_label: str):
    print(f"\n{'='*60}")
    print(f"TEST: {count_label} ({len(paths)} photos)")
    print(f"{'='*60}")

    vehicle_name = "蔚来 ES8 2019款"
    parts_list = json.dumps(PARTS_CATALOG, ensure_ascii=False, indent=2)

    system_prompt = f"""你是汽车外部损伤评估专家。给定 {vehicle_name} 的车型先验和多张已定位的照片，请对车辆外部每个部件进行损伤识别，输出结构化 JSON。

必须检查的部件清单（每个部件都必须有结论）：
{parts_list}

输出 JSON 格式：
{{
  "parts": [
    {{
      "part_id": "hood",
      "part_name": "引擎盖",
      "status": "intact|damaged|uncertain",
      "damage_level": "none|light|moderate|severe",
      "damage_type": ["scratch", "dent", "crack", "tear", "deformation", "missing", "other"],
      "confidence": "high|medium|low",
      "evidence_photo": ["photo_01"],
      "notes": "补充说明"
    }}
  ],
  "uncertain_items": []
}}

只输出 JSON，不要额外文字。"""

    content = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": "以下是已定位的关键照片："},
    ]
    for i, path in enumerate(paths):
        content.append({"type": "text", "text": f"照片 {i+1}: {path.split('/')[-1]}"})
        content.append(build_image_content(path))

    try:
        raw = await call_minimax([{"role": "user", "content": content}], temperature=0.1, max_tokens=4000)
        result = extract_json(raw)
        print(f"OK - result type: {type(result)}")
        if isinstance(result, dict) and "parts" in result:
            print(f"Parts count: {len(result['parts'])}")
    except Exception as e:
        print(f"FAILED: {e}")

async def main():
    folder = "/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868"
    await test_with_photos([f"{folder}/蔚来ES8_02.png"], "1 photo")
    await test_with_photos([f"{folder}/蔚来ES8_02.png", f"{folder}/蔚来ES8_03.png"], "2 photos")
    await test_with_photos([f"{folder}/蔚来ES8_02.png", f"{folder}/蔚来ES8_03.png", f"{folder}/蔚来ES8_04.png"], "3 photos")

asyncio.run(main())
