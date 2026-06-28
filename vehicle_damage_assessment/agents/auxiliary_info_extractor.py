"""Extract vehicle info from auxiliary photos such as license and VIN labels."""

import re
from typing import Any, Dict, List, Optional

from agents.minimax_client import call_minimax, build_image_content, extract_json


_SYSTEM_PROMPT = """你是车辆信息识别专家。给定一张车辆证件照片或车辆铭牌/VIN照片，请提取以下字段：

输出 JSON 格式：
{
  "brand": "品牌中文名或英文名，如保时捷、保时捷Panamera、奔驰、BMW",
  "model": "车系/型号，如Panamera、C级、3系",
  "year": "年款，如2019、2020；若无法识别则为空字符串",
  "vin": "VIN码，17位字母数字组合",
  "confidence": "high/medium/low",
  "reason": "判断理由"
}

规则：
1. 行驶证照片：重点关注"品牌型号"字段，例如"保时捷WP0AJ2Z97"应提取 brand=保时捷、model=Panamera
2. VIN标签照片：提取17位VIN码，若VIN前三位能推断品牌则填写 brand
3. 若某字段无法识别，使用空字符串""
4. 不要编造信息，不确定就为空字符串
"""


async def extract_vehicle_info_from_auxiliary_photos(
    photos: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Extract brand/model/year from auxiliary photos.

    Parameters
    ----------
    photos:
        List of dicts with ``id`` and ``path`` keys, already filtered to
        auxiliary photos (driver's license, VIN label, etc.).

    Returns
    -------
    dict
        ``{"brand": "", "model": "", "year": "", "vin": ""}`` with whatever
        could be inferred. Empty strings for unknown fields.
    """
    result = {"brand": "", "model": "", "year": "", "vin": ""}
    if not photos:
        return result

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": _SYSTEM_PROMPT},
        {"type": "text", "text": f"共 {len(photos)} 张辅助信息照片，请逐张分析并汇总最可能的车辆信息："},
    ]
    for photo in photos:
        content.append({"type": "text", "text": f"照片编号: {photo['id']}"})
        content.append(build_image_content(photo["path"]))

    messages = [{"role": "user", "content": content}]

    raw = await call_minimax(messages, temperature=0.1, max_tokens=1500)
    extracted = extract_json(raw)
    if not isinstance(extracted, dict):
        return result

    for key in result:
        value = extracted.get(key)
        if isinstance(value, str):
            result[key] = value.strip()

    # Clean obvious noise
    if result["year"] and not re.match(r"^\d{4}$", result["year"]):
        result["year"] = ""

    if result["vin"]:
        result["vin"] = result["vin"].upper().strip()
        if len(result["vin"]) != 17:
            result["vin"] = ""

    return result
