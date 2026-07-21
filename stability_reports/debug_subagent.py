import asyncio, sys, json
sys.path.insert(0, '/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment')
from agents.minimax_client import call_minimax, extract_json
from agents.vision_subagent import _build_covered_regions_text
from agents.topology_builder import build_vehicle_topology
from agents.image_utils import compress_image_to_base64
from agents.view_mapping import get_display_name
from config import IMAGE_MAX_WIDTH

photos = [{'id': '167111-03.png', 'path': '/Users/sky/Downloads/车顶闸调试样本_20260622/lead_167111/167111-03.png', 'name': '167111-03.png'}]
vehicle_prior = {'vehicle': '2019 蔚来 ES8'}
topology = build_vehicle_topology({'vehicle_id':'test'}, {**vehicle_prior, 'vehicle_specs': {'body_style':'suv','doors':5,'has_sunroof':True,'has_roof_rack':False,'rear_door_type':'tailgate'}})

view_id = 'front_right'
view_display_name = get_display_name(view_id)
covered_regions_text = _build_covered_regions_text(view_id, topology)

with open('/Users/sky/vehicle_damage_assessment/vehicle_damage_assessment/agents/vision_subagent.py') as f:
    code = f.read()

# Extract prompt template
import re
m = re.search(r'_SYSTEM_PROMPT_TEMPLATE = """(.*?)"""', code, re.DOTALL)
template = m.group(1) if m else ""

system_prompt = template.format(
    view_id=view_id,
    view_display_name=view_display_name,
    covered_regions_text=covered_regions_text,
    vehicle_name=vehicle_prior['vehicle'],
)

content = [
    {'type': 'text', 'text': system_prompt},
    {'type': 'text', 'text': f'以下是 1 张{view_display_name}照片，请联合分析：'},
]
for photo in photos:
    content.append({'type': 'text', 'text': f"照片编号: {photo['id']}"})
    data_uri, _ = compress_image_to_base64(photo['path'], max_width=IMAGE_MAX_WIDTH)
    content.append({'type': 'image_url', 'image_url': {'url': data_uri}})

async def main():
    messages = [{'role': 'user', 'content': content}]
    raw = await call_minimax(messages, temperature=0.1, max_tokens=3000)
    print('RAW LENGTH:', len(raw))
    print('RAW FIRST 1000:', raw[:1000])
    result = extract_json(raw)
    print('RESULT PARTS COUNT:', len(result.get('parts', [])) if result else 'NONE')

asyncio.run(main())
