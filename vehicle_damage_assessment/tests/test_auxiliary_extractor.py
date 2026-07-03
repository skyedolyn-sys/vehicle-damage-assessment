"""Tests for auxiliary_info_extractor deterministic VIN/brand extraction.

DAMAGE_RECOGNITION_POLICY §1.6 / 步骤 5: 验证 100% 确定性替代 LLM 调用。
"""

import asyncio

import pytest

from agents.auxiliary_info_extractor import (
    _VIN_REGEX,
    _WMI_TO_BRAND,
    extract_vehicle_info_from_auxiliary_photos,
)


class TestExtractVehicleInfoFromAuxiliaryPhotos:
    """DAMAGE_RECOGNITION_POLICY §1.6 / 步骤 5: 确定性 VIN 解析。"""

    @pytest.mark.asyncio
    async def test_vin_in_filename(self):
        """VIN 出现在文件名 → 提取 + WMI 查 brand"""
        r = await extract_vehicle_info_from_auxiliary_photos([
            {"id": "172852-vin-WBA3A5C50CF256789.png", "path": "/tmp/x.png"},
        ])
        assert r["vin"] == "WBA3A5C50CF256789"
        assert r["brand"] == "宝马"

    @pytest.mark.asyncio
    async def test_mercedes_vin(self):
        r = await extract_vehicle_info_from_auxiliary_photos([
            {"id": "WDBUF56X48B123456-vin.png", "path": "/tmp/x.png"},
        ])
        assert r["vin"] == "WDBUF56X48B123456"
        assert r["brand"] == "奔驰"

    @pytest.mark.asyncio
    async def test_filename_keyword_brand(self):
        """文件名含中文品牌关键词 → 提取 brand"""
        r = await extract_vehicle_info_from_auxiliary_photos([
            {"id": "行驶证-蔚来ES8.png", "path": "/tmp/x.png"},
        ])
        assert r["brand"] == "蔚来"

    @pytest.mark.asyncio
    async def test_filename_year_extraction(self):
        """文件名含 4 位年份 → 提取 year"""
        r = await extract_vehicle_info_from_auxiliary_photos([
            {"id": "行驶证-宝马X5-2019.png", "path": "/tmp/x.png"},
        ])
        assert r["year"] == "2019"

    @pytest.mark.asyncio
    async def test_empty_photos(self):
        r = await extract_vehicle_info_from_auxiliary_photos([])
        assert r == {"brand": "", "model": "", "year": "", "vin": ""}

    @pytest.mark.asyncio
    async def test_invalid_vin_excluded(self):
        """含 I/O/Q 的 17 位字符串不应被识别为 VIN"""
        # 注意 VIN 排除 I/O/Q,所以"INVALIDI123456789"长度 17 但含 I,正则不匹配
        assert _VIN_REGEX.search("INVALIDI123456789") is None
        assert _VIN_REGEX.search("1O23456789012345A") is None

    @pytest.mark.asyncio
    async def test_wmi_table_coverage(self):
        """WMI 字典覆盖至少 6 个主流品牌。"""
        assert len(_WMI_TO_BRAND) >= 6
        assert "WBA" in _WMI_TO_BRAND
        assert "WDB" in _WMI_TO_BRAND
        assert "WVW" in _WMI_TO_BRAND

    @pytest.mark.asyncio
    async def test_multiple_photos_first_vin_wins(self):
        """多张照片时,找到第一个 VIN 即可"""
        r = await extract_vehicle_info_from_auxiliary_photos([
            {"id": "行驶证.png", "path": "/tmp/x.png"},
            {"id": "vin-LSVAB12345678901Y.png", "path": "/tmp/x.png"},
        ])
        assert r["vin"] == "LSVAB12345678901Y"

    @pytest.mark.asyncio
    async def test_no_llm_call(self):
        """DAMAGE_RECOGNITION_POLICY §1.6: extract_auxiliary_info 绝不应调 LLM。

        通过验证模块没有 import call_minimax 来确保(模块级别静态保证)。
        """
        import agents.auxiliary_info_extractor as mod
        assert not hasattr(mod, "call_minimax"), (
            "auxiliary_info_extractor 不应再 import call_minimax"
        )