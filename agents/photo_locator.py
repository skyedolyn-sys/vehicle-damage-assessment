import json
from typing import List, Dict, Any
from agents.minimax_client import call_minimax, build_image_content, extract_json


async def photo_locator_agent(
    photo_batch: List[Dict[str, Any]],
    vehicle_prior: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    第二层：锚点定位
    对每批照片，判断每张图拍摄的是车辆的哪个区域
    """
    vehicle_name = vehicle_prior.get("vehicle", "该车")
    topology = json.dumps(vehicle_prior.get("topology", {}), ensure_ascii=False, indent=2)
    anchors = json.dumps(vehicle_prior.get("key_anchors", {}), ensure_ascii=False, indent=2)

    system_prompt = f"""你是汽车照片视角识别专家。给定 {vehicle_name} 的车型先验信息，判断每张照片拍摄的是车辆的哪个区域。

车型先验：
{topology}

关键锚点：
{anchors}

对每张照片，输出一个 JSON 对象：
{{
  "photo_id": "照片编号",
  "location": "主要区域：车头/车尾/左侧/右侧/车顶/内饰/辅助信息/无法定位",
  "secondary_locations": ["可选的次要区域，如左侧、车尾"],
  "location_detail": "更具体的描述，如车头左前45度、车尾左后45度等",
  "primary_anchor": "用于定位的主要锚点",
  "confidence": "high/medium/low",
  "reason": "判断理由",
  "visible_parts": ["可见部件1", "可见部件2"]
}}

必须返回一个 JSON 数组，每个元素对应一张照片，顺序与输入照片一致。只输出 JSON 数组，不要额外文字。

规则：
1. 优先使用强锚点：品牌logo、方向盘位置、车门框架、VIN位置、充电口/油箱盖位置
2. 车辆严重变形时，如果无法可靠区分车头/车尾，confidence设为low
3. 内饰照片标记为"内饰"
4. 行驶证/VIN/铭牌等辅助照片标记为"辅助信息"
5. 不确定就标记"无法定位"
6. 左右侧判断（重要）：
   - 中国大陆车辆为左舵（方向盘在左侧），但左右侧判断以车身实际侧为准，与驾驶位无关
   - 判断方法（按可靠性排序）：
     a) 若能看到加油口/充电口盖板，它在哪一侧后翼子板，就是车辆的哪一侧
     b) 看画面中完整露出的后车门：完整看到右后车门→右侧；完整看到左后车门→左侧
     c) 车尾侧后方视角：车尾朝向画面一侧，车身露出的另一侧即为车辆侧
7. secondary_locations 填写规则（一张照片可能同时属于多个区域）：
   - 车头左前45度：location="车头"，secondary_locations=["左侧"]
   - 车头右前45度：location="车头"，secondary_locations=["右侧"]
   - 车尾左后45度：location="车尾"，secondary_locations=["左侧"]
   - 车尾右后45度：location="车尾"，secondary_locations=["右侧"]
   - 左侧90度（从前到后看到左前门、左后门、左后翼子板）：location="左侧"，secondary_locations=["车头","车尾"]
   - 右侧90度：location="右侧"，secondary_locations=["车头","车尾"]
   - 纯车头正前方/正后方/车顶俯视/纯内饰：secondary_locations=[]
8. 辅助信息照片（行驶证、VIN铭牌）除了标记 location=辅助信息，还请在 visible_parts 中尽量提取品牌、型号、VIN 等车辆信息
"""

    content = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": "以下是待识别的照片，请逐张分析："},
    ]
    for photo in photo_batch:
        content.append({"type": "text", "text": f"照片编号: {photo['id']}"})
        # 使用本地文件路径（会自动转为 base64），避免 localhost URL 被 API 拒绝
        image_url = photo["path"]
        content.append(build_image_content(image_url))

    messages = [
        {"role": "user", "content": content},
    ]

    raw = await call_minimax(messages, temperature=0.1, max_tokens=4000)
    result = extract_json(raw)
    if result is None:
        # 模型未返回可解析 JSON，给该批次每张照片 fallback 结果
        return [
            {
                "photo_id": photo["id"],
                "location": "无法定位",
                "secondary_locations": [],
                "location_detail": "模型输出无法解析",
                "primary_anchor": "无",
                "confidence": "low",
                "reason": f"模型返回内容无法解析为 JSON: {raw[:200] if raw else '(empty)'}",
                "visible_parts": [],
            }
            for photo in photo_batch
        ]
    if isinstance(result, dict) and "results" in result:
        result = result["results"]
    if isinstance(result, dict):
        result = [result]

    # Normalize secondary_locations to a list
    for item in result:
        if isinstance(item, dict):
            secondary = item.get("secondary_locations", [])
            if isinstance(secondary, str):
                secondary = [s.strip() for s in secondary.split(",") if s.strip()]
            elif not isinstance(secondary, list):
                secondary = []
            item["secondary_locations"] = secondary

    return result
