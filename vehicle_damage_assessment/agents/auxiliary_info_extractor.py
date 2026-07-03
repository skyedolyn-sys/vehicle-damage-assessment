"""Extract vehicle info from auxiliary photos — deterministic.

DAMAGE_RECOGNITION_POLICY §1.6 / 步骤 5: 把辅助信息(行驶证/VIN/铭牌)的提取
从 LLM 改为纯确定性。VIN 用 17 位正则匹配,WMI(前 3 位)用字典查表得到 brand。

完全无 LLM 调用、无 OCR(后续 PR 可加 tesseract),仅靠文件名启发式 + VIN 字典。
"""

import re
from typing import Any, Dict, List


# VIN 17 位: 大写字母 + 数字,排除 I/O/Q
_VIN_REGEX = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")


# WMI → (中文 brand, 英文 brand) 字典。覆盖最常见的几个主流品牌。
# 后续 PR 可以扩充到 ~50 个主流 WMI。本次先列 8 个。
_WMI_TO_BRAND: Dict[str, tuple[str, str]] = {
    "WBA": ("宝马", "BMW"),
    "WBS": ("宝马 M", "BMW M"),
    "WDB": ("奔驰", "Mercedes-Benz"),
    "WDC": ("奔驰", "Mercedes-Benz"),
    "WDD": ("奔驰", "Mercedes-Benz"),
    "WDF": ("奔驰", "Mercedes-Benz"),
    "WVW": ("大众", "Volkswagen"),
    "WVG": ("大众", "Volkswagen"),
    "LFV": ("一汽大众", "FAW-VW"),
    "LSV": ("上汽大众", "SAIC-VW"),
    "LGB": ("东风日产", "Dongfeng-Nissan"),
    "LSG": ("上汽通用", "SAIC-GM"),
    "LHG": ("广汽本田", "GAC-Honda"),
    "LVH": ("东风本田", "Dongfeng-Honda"),
    "LFM": ("一汽丰田", "FAW-Toyota"),
    "LTV": ("丰田", "Toyota"),
}


async def extract_vehicle_info_from_auxiliary_photos(
    photos: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Extract brand/model/year/vin from auxiliary photos.

    DAMAGE_RECOGNITION_POLICY §1.6 (步骤 5): 纯确定性,无 LLM。

    算法:
    1. 从每张照片的文件名/id 里找 17 位 VIN(用正则,排除 I/O/Q)
    2. 用 VIN 前 3 位(WMI)查 _WMI_TO_BRAND 得到 brand
    3. model / year 留空(后续 PR 用 OCR 补)

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
    result: Dict[str, str] = {"brand": "", "model": "", "year": "", "vin": ""}
    if not photos:
        return result

    # 1. 在 filename/id 里找 17 位 VIN
    vin = ""
    for photo in photos:
        for hint in [photo.get("id", ""), photo.get("name", "") or photo.get("path", "")]:
            match = _VIN_REGEX.search(hint)
            if match:
                vin = match.group(0).upper()
                break
        if vin:
            break

    # 2. 用 WMI 查 brand
    brand_cn = brand_en = ""
    if len(vin) >= 3:
        wmi = vin[:3]
        if wmi in _WMI_TO_BRAND:
            brand_cn, brand_en = _WMI_TO_BRAND[wmi]

    # 3. 用文件名启发式补 brand(若 VIN 没匹配)
    if not brand_cn:
        lowered = " ".join(
            str(photo.get("id", "")) + " " + str(photo.get("name", ""))
            for photo in photos
        ).lower()
        # 中文品牌关键词
        brand_keywords = {
            "宝马": ("宝马", "BMW"),
            "奔驰": ("奔驰", "Mercedes-Benz"),
            "大众": ("大众", "Volkswagen"),
            "丰田": ("丰田", "Toyota"),
            "本田": ("本田", "Honda"),
            "日产": ("日产", "Nissan"),
            "奥迪": ("奥迪", "Audi"),
            "保时捷": ("保时捷", "Porsche"),
            "蔚来": ("蔚来", "NIO"),
            "特斯拉": ("特斯拉", "Tesla"),
            "tesla": ("特斯拉", "Tesla"),
            "bmw": ("宝马", "BMW"),
            "mercedes": ("奔驰", "Mercedes-Benz"),
        }
        for keyword, (cn, en) in brand_keywords.items():
            if keyword in lowered:
                brand_cn, brand_en = cn, en
                break

    # 4. 用文件名启发式补 year(若 VIN 没匹配)
    year = ""
    for photo in photos:
        for hint in [photo.get("id", ""), photo.get("name", "") or photo.get("path", "")]:
            # 找 4 位年份(1990-2030)
            year_match = re.search(r"\b(199\d|20[0-3]\d)\b", hint)
            if year_match:
                year = year_match.group(0)
                break
        if year:
            break

    result["vin"] = vin
    result["brand"] = brand_cn
    result["model"] = ""  # 当前无 OCR,留空(后续 PR 补)
    result["year"] = year

    return result