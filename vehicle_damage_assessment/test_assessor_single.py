import asyncio
import json
import sys
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents.minimax_client import call_minimax, build_image_content, extract_json
from config import PARTS_CATALOG

async def main():
    vehicle_name = "蔚来 ES8 2019款"
    parts_list = json.dumps(PARTS_CATALOG, ensure_ascii=False, indent=2)

    system_prompt = f"""你是汽车外部损伤评估专家。给定 {vehicle_name} 的车型先验和一张已定位的照片，请对车辆外部每个部件进行损伤识别，输出结构化 JSON。

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
        {"type": "text", "text": "照片 蔚来ES8_02.png - 车头正前偏上视角，可见引擎盖大面积损毁"},
        build_image_content("/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_02.png"),
    ]

    raw = await call_minimax([{"role": "user", "content": content}], temperature=0.1, max_tokens=4000)
    print("RAW:", repr(raw[:1000]))
    result = extract_json(raw)
    print("TYPE:", type(result))
    print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])

asyncio.run(main())
