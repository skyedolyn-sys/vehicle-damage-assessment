import asyncio
import json
import sys
sys.path.insert(0, '/Users/sky/Downloads/vehicle_damage_assessment')
from agents.damage_assessor import damage_assessor_agent
from agents.minimax_client import call_minimax, build_image_content

async def main():
    vehicle_prior = {
        "vehicle": "蔚来 ES8 2019款",
        "topology": {"front": "X-bar 前脸", "rear": "贯穿尾灯", "roof": "全景天窗"},
        "key_anchors": {"front": ["NIO logo"], "rear": ["贯穿尾灯"]}
    }
    
    # 模拟已定位的关键照片
    all_photos = []
    locations = []
    selected_ids = ["02", "04", "06", "11", "16", "27"]
    for pid in selected_ids:
        all_photos.append({
            "id": f"蔚来ES8_{pid}.png",
            "path": f"/Users/sky/Downloads/车顶闸样本_蔚来ES8_176868/蔚来ES8_{pid}.png",
            "url": f"http://localhost:8000/uploads/test/蔚来ES8_{pid}.png"
        })
        locations.append({
            "photo_id": f"蔚来ES8_{pid}.png",
            "location": "车头" if pid in ["02", "04", "06", "11", "16", "27"] else "车尾",
            "location_detail": "车头严重碰撞区域",
            "confidence": "high"
        })
    
    try:
        result = await damage_assessor_agent(all_photos, locations, vehicle_prior)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:3000])
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(main())
