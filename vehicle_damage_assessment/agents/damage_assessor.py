import asyncio
from typing import List, Dict, Any, Optional
from agents.regional_worker import regional_damage_worker
from agents.synthesizer import synthesizer_agent
from config import MAX_CONCURRENT_API_CALLS
from models.topology import VehicleTopology


# Cross-region sharing rules: primary location -> secondary locations.
# These are intentionally broad so that a photo captured from an angle that
# may cover adjacent regions is available to the corresponding workers.
_SECONDARY_LOCATIONS = {
    "车头": ["左侧", "右侧"],
    "车尾": ["左侧", "右侧"],
    "左侧": ["车头", "车尾"],
    "右侧": ["车头", "车尾"],
    "车顶": [],
}


async def damage_assessor_agent(
    all_photos: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
    topology: Optional[VehicleTopology] = None,
) -> Dict[str, Any]:
    """
    第三层：区域并行损伤评估 + Synthesizer 汇总
    """
    # 按位置分组照片；一张照片可能同时属于多个区域
    photos_by_location: Dict[str, List[Dict[str, Any]]] = {}

    for loc in locations:
        loc_key = loc.get("location", "无法定位")
        loc_detail = loc.get("location_detail", "")
        photo_id = loc.get("photo_id")
        photo = next((p for p in all_photos if p["id"] == photo_id), None)
        if photo is None:
            continue

        photo_entry = {
            "id": photo_id,
            "path": photo["path"],
            "url": photo.get("url") or photo["path"],
            "detail": loc_detail,
            "confidence": loc.get("confidence", "low"),
        }

        # Collect all locations this photo belongs to
        all_locations = {loc_key}
        secondary = loc.get("secondary_locations", [])
        if isinstance(secondary, list):
            all_locations.update(secondary)
        # Always include the configured cross-region shares for the primary location
        for shared in _SECONDARY_LOCATIONS.get(loc_key, []):
            all_locations.add(shared)

        for location in all_locations:
            if location not in ["无法定位", "辅助信息", "内饰"]:
                photos_by_location.setdefault(location, []).append(photo_entry)

    # 排除无效位置
    valid_locations = {
        loc_key: photos
        for loc_key, photos in photos_by_location.items()
        if loc_key not in ["无法定位", "辅助信息", "内饰"] and photos
    }

    if not valid_locations:
        return {"parts": [], "uncertain_items": []}

    # 并行启动区域 Worker
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

    async def run_worker(location: str, photos: List[Dict[str, Any]]):
        async with semaphore:
            try:
                return await regional_damage_worker(location, photos, vehicle_prior, topology)
            except Exception as e:
                # 单个区域失败不影响其他区域
                return {"region": location, "parts": [], "uncertain_items": [], "error": str(e)}

    tasks = [
        asyncio.create_task(run_worker(location, photos))
        for location, photos in valid_locations.items()
    ]

    region_results = await asyncio.gather(*tasks)

    # 过滤空结果
    region_results = [r for r in region_results if r.get("parts")]

    if not region_results:
        return {"parts": [], "uncertain_items": []}

    # Synthesizer 汇总（确定性合并，非 LLM）
    final_result = synthesizer_agent(region_results, vehicle_prior, topology)
    return final_result
