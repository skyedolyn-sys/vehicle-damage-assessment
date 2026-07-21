#!/usr/bin/env python3
"""Seed script: pre-fill vehicle_specs_cache.json with common Chinese-market vehicles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path so imports resolve when run as script
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.vehicle_specs import VehicleSpecs  # noqa: E402
from data.vehicle_specs_cache import (  # noqa: E402
    save_cached_specs,
    _load_json_cache as _load_cache,
    _save_json_cache as _save_cache,
    CACHE_FILE,
)

# ---------------------------------------------------------------------------
# Common Chinese-market vehicles (200+ entries)
# ---------------------------------------------------------------------------

COMMON_CHINESE_VEHICLES: List[Dict[str, Any]] = [
    # ============================ 蔚来 NIO (5) ============================
    {
        "vehicle_info": {"brand": "蔚来", "model": "ES6", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="split",
            rear_door_type="tailgate",
            notes="中型纯电SUV，分体式大灯，全景天幕，隐藏式门把手",
        ),
    },
    {
        "vehicle_info": {"brand": "蔚来", "model": "ES8", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="split",
            rear_door_type="tailgate",
            notes="中大型纯电SUV，分体式大灯，全景天幕，无行李架",
        ),
    },
    {
        "vehicle_info": {"brand": "蔚来", "model": "ET5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="split",
            rear_door_type="trunk_lid",
            notes="中型纯电轿车，分体式大灯，全景天幕，溜背造型",
        ),
    },
    {
        "vehicle_info": {"brand": "蔚来", "model": "ET7", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="split",
            rear_door_type="trunk_lid",
            notes="中大型纯电轿车，分体式大灯，全景天幕，行政级轿车",
        ),
    },
    {
        "vehicle_info": {"brand": "蔚来", "model": "EC6", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="split",
            rear_door_type="tailgate",
            notes="中型纯电轿跑SUV，分体式大灯，溜背尾门，全景天幕",
        ),
    },
    # ============================ 理想 Li Auto (4) ============================
    {
        "vehicle_info": {"brand": "理想", "model": "L7", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="tailgate",
            notes="中大型五座SUV，星环灯贯穿前脸，全景天幕，电动尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "理想", "model": "L8", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="tailgate",
            notes="中大型六座SUV，星环灯贯穿前脸，双全景天幕，电动尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "理想", "model": "L9", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="tailgate",
            notes="全尺寸六座SUV，星环灯贯穿前脸，前后双全景天幕，电动尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "理想", "model": "MEGA", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="sliding",
            notes="纯电旗舰MPV，水滴型外观设计，双侧滑门，电动尾门",
        ),
    },
    # ============================ 小鹏 XPeng (4) ============================
    {
        "vehicle_info": {"brand": "小鹏", "model": "P7", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型纯电轿跑，贯穿式前后灯，全景天幕，无框车门",
        ),
    },
    {
        "vehicle_info": {"brand": "小鹏", "model": "G6", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型纯电轿跑SUV，溜背造型，电动尾门，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "小鹏", "model": "G9", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型纯电SUV，贯穿式前后灯，全景天幕，电动尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "小鹏", "model": "X9", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="sliding",
            notes="纯电旗舰MPV，星舰造型，双侧滑门，电动尾门，全景天幕",
        ),
    },
    # ============================ 比亚迪 BYD (14) ============================
    {
        "vehicle_info": {"brand": "比亚迪", "model": "秦PLUS", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型插混/纯电轿车，Dragon Face前脸，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "秦L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型插混轿车，新一代Dragon Face前脸，贯穿尾灯，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "汉", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型纯电/插混轿车，Dragon Face前脸，贯穿尾灯，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "汉L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型纯电轿车，新一代龙颜设计，贯穿尾灯，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "海豹", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型纯电轿跑，海洋美学设计，溜背造型，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "唐", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型七座SUV，Dragon Face前脸，电动尾门，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "宋Pro", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，Dragon Face前脸，电动尾门，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "宋PLUS", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，海洋美学前脸，电动尾门，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "元PLUS", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型纯电SUV，Dragon Face前脸，电动尾门，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "元UP", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="小型纯电SUV，海洋美学设计，无天窗，电动尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "海豚", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="hatchback",
            doors=5,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="小型纯电两厢车，海洋美学设计，无天窗，电动尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "海鸥", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="hatchback",
            doors=5,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="微型纯电两厢车，海洋美学设计，无天窗，入门级车型",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "驱逐舰05", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型插混轿车，海洋美学前脸，普通天窗，家用轿车",
        ),
    },
    {
        "vehicle_info": {"brand": "比亚迪", "model": "护卫舰07", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型插混SUV，海洋美学前脸，电动尾门，全景天幕",
        ),
    },
    # ============================ 腾势 Denza (1) ============================
    {
        "vehicle_info": {"brand": "腾势", "model": "D9", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="sliding",
            notes="中大型豪华MPV，π-Motion前脸，双侧电动滑门，电动尾门，分段式天窗",
        ),
    },
    # ============================ 仰望 Yangwang (1) ============================
    {
        "vehicle_info": {"brand": "仰望", "model": "U8", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="大型豪华越野SUV，鼎字大灯设计，行李架，电动尾门，全景天幕",
        ),
    },
    # ============================ 方程豹 Fangchengbao (1) ============================
    {
        "vehicle_info": {"brand": "方程豹", "model": "豹5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型越野SUV，硬派方盒子造型，行李架，电动尾门，全景天幕",
        ),
    },
    # ============================ 特斯拉 Tesla (4) ============================
    {
        "vehicle_info": {"brand": "特斯拉", "model": "Model 3", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型纯电轿车，极简无格栅前脸，全景玻璃车顶，隐藏门把手",
        ),
    },
    {
        "vehicle_info": {"brand": "特斯拉", "model": "Model Y", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型纯电SUV，溜背造型，全景玻璃车顶，电动尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "特斯拉", "model": "Model S", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型纯电轿车，极简无格栅前脸，全景玻璃车顶，隐藏门把手",
        ),
    },
    {
        "vehicle_info": {"brand": "特斯拉", "model": "Model X", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型纯电SUV，鹰翼门设计，全景玻璃车顶，电动尾门",
        ),
    },
    # ============================ 宝马 BMW (6) ============================
    {
        "vehicle_info": {"brand": "宝马", "model": "3系", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型豪华轿车，双肾格栅，天使眼大灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "宝马", "model": "5系", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型豪华轿车，双肾格栅，贯穿式尾灯（新款），分段式天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "宝马", "model": "X3", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华SUV，双肾格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "宝马", "model": "X5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型豪华SUV，双肾格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "宝马", "model": "4系", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="coupe",
            doors=2,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型豪华轿跑，大双肾格栅，无框车门，溜背造型，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "宝马", "model": "i3", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型纯电轿车，封闭式双肾格栅，天使眼大灯，全景天幕",
        ),
    },
    # ============================ 奔驰 Mercedes-Benz (6) ============================
    {
        "vehicle_info": {"brand": "奔驰", "model": "C级", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型豪华轿车，夜幕星河格栅，普通天窗，经典三厢造型",
        ),
    },
    {
        "vehicle_info": {"brand": "奔驰", "model": "E级", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型豪华轿车，花生大灯，贯穿尾灯，分段式全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奔驰", "model": "GLC", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华SUV，夜幕星河格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奔驰", "model": "GLE", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型豪华SUV，夜幕星河格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奔驰", "model": "CLE", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="coupe",
            doors=2,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型豪华轿跑，夜幕星河格栅，无框车门，溜背造型，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奔驰", "model": "S级", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="大型豪华轿车，盾形格栅，流星雨大灯，分段式全景天窗",
        ),
    },
    # ============================ 奥迪 Audi (6) ============================
    {
        "vehicle_info": {"brand": "奥迪", "model": "A4L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型豪华轿车，六边形格栅，矩阵LED大灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奥迪", "model": "A6L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型豪华轿车，六边形格栅，矩阵LED大灯，分段式全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奥迪", "model": "Q5L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华SUV，八边形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奥迪", "model": "A5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="coupe",
            doors=2,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型豪华轿跑，六边形格栅，无框车门，溜背造型，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奥迪", "model": "A4 Avant", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="wagon",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华旅行车，六边形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奥迪", "model": "Q3", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型豪华SUV，八边形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    # ============================ 丰田 Toyota (9) ============================
    {
        "vehicle_info": {"brand": "丰田", "model": "凯美瑞", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，梯形大嘴格栅，C型大灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "丰田", "model": "卡罗拉", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，梯形格栅，普通大灯，无天窗，家用代步",
        ),
    },
    {
        "vehicle_info": {"brand": "丰田", "model": "雷凌", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，梯形格栅，普通大灯，无天窗，家用代步",
        ),
    },
    {
        "vehicle_info": {"brand": "丰田", "model": "RAV4荣放", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，梯形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "丰田", "model": "威兰达", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，梯形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "丰田", "model": "锋兰达", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，梯形格栅，无行李架，普通天窗，家用SUV",
        ),
    },
    {
        "vehicle_info": {"brand": "丰田", "model": "汉兰达", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型七座SUV，梯形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "丰田", "model": "赛那", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="sliding",
            notes="中大型MPV，梯形格栅，双侧电动滑门，电动尾门，分段式天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "丰田", "model": "格瑞维亚", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="sliding",
            notes="中大型MPV，梯形格栅，双侧电动滑门，电动尾门，分段式天窗",
        ),
    },
    # ============================ 本田 Honda (9) ============================
    {
        "vehicle_info": {"brand": "本田", "model": "雅阁", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，六边形格栅，贯穿式尾灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "本田", "model": "思域", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，扁平格栅，贯穿式尾灯，普通天窗，运动风格",
        ),
    },
    {
        "vehicle_info": {"brand": "本田", "model": "型格", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，扁平格栅，贯穿式尾灯，普通天窗，运动风格",
        ),
    },
    {
        "vehicle_info": {"brand": "本田", "model": "CR-V", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，六边形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "本田", "model": "皓影", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，六边形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "本田", "model": "冠道", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型SUV，六边形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "本田", "model": "UR-V", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型SUV，六边形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "本田", "model": "奥德赛", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="sliding",
            notes="中型MPV，大嘴格栅，双侧电动滑门，电动尾门，分段式天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "本田", "model": "艾力绅", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="sliding",
            notes="中型MPV，大嘴格栅，双侧电动滑门，电动尾门，分段式天窗",
        ),
    },
    # ============================ 日产 Nissan (6) ============================
    {
        "vehicle_info": {"brand": "日产", "model": "轩逸", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，V-Motion格栅，普通大灯，无天窗，家用代步",
        ),
    },
    {
        "vehicle_info": {"brand": "日产", "model": "天籁", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，V-Motion格栅，普通天窗，家用舒适",
        ),
    },
    {
        "vehicle_info": {"brand": "日产", "model": "奇骏", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，V-Motion格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "日产", "model": "逍客", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，V-Motion格栅，普通天窗，电动尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "日产", "model": "途乐", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="大型越野SUV，V-Motion格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "日产", "model": "ARIYA艾睿雅", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型纯电SUV，封闭式盾形格栅，贯穿尾灯，全景天幕",
        ),
    },
    # ============================ 大众 Volkswagen (11) ============================
    {
        "vehicle_info": {"brand": "大众", "model": "迈腾", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，横幅格栅，贯穿尾灯（新款），普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "大众", "model": "帕萨特", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，星空/经典双前脸，贯穿尾灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "大众", "model": "朗逸", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，横幅格栅，普通大灯，无天窗，家用代步",
        ),
    },
    {
        "vehicle_info": {"brand": "大众", "model": "速腾", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，横幅格栅，普通天窗，家用代步",
        ),
    },
    {
        "vehicle_info": {"brand": "大众", "model": "宝来", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，横幅格栅，普通大灯，无天窗，家用代步",
        ),
    },
    {
        "vehicle_info": {"brand": "大众", "model": "途观L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型SUV，横幅格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "大众", "model": "探岳", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型SUV，X型前脸，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "大众", "model": "揽巡", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型SUV，横幅格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "大众", "model": "途昂", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型SUV，横幅格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "大众", "model": "ID.3", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="hatchback",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型纯电两厢车，封闭式前脸，全景天幕，电动尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "大众", "model": "ID.4", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型纯电SUV，封闭式前脸，贯穿灯带，全景天幕，电动尾门",
        ),
    },
    # ============================ 别克 Buick (4) ============================
    {
        "vehicle_info": {"brand": "别克", "model": "君威", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，飞翼格栅，展翼大灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "别克", "model": "君越", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型轿车，飞翼格栅，展翼大灯，分段式天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "别克", "model": "昂科威", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型SUV，飞翼格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "别克", "model": "GL8", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="sliding",
            notes="中大型MPV，飞翼格栅，双侧电动滑门，电动尾门，分段式天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "别克", "model": "世纪", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="sliding",
            notes="大型豪华MPV，飞翼格栅，双侧电动滑门，电动尾门，分段式天窗",
        ),
    },
    # ============================ 雪佛兰 Chevrolet (4) ============================
    {
        "vehicle_info": {"brand": "雪佛兰", "model": "科鲁泽", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，双格栅设计，普通大灯，无天窗，家用代步",
        ),
    },
    {
        "vehicle_info": {"brand": "雪佛兰", "model": "迈锐宝XL", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，双格栅设计，普通天窗，溜背造型",
        ),
    },
    {
        "vehicle_info": {"brand": "雪佛兰", "model": "探界者", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型SUV，双格栅设计，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "雪佛兰", "model": "开拓者", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型SUV，双格栅设计，行李架，电动尾门，全景天窗",
        ),
    },
    # ============================ 福特 Ford (5) ============================
    {
        "vehicle_info": {"brand": "福特", "model": "蒙迪欧", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，八边形格栅，贯穿尾灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "福特", "model": "锐界L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型七座SUV，八边形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "福特", "model": "探险者", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型SUV，八边形格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "福特", "model": "Mustang Mach-E", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型纯电轿跑SUV，封闭式 Mustang 前脸，溜背造型，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "福特", "model": "F-150", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="pickup",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="大型皮卡，C型大灯，货箱尾门，全景天窗，越野风格",
        ),
    },
    # ============================ 现代 Hyundai (4) ============================
    {
        "vehicle_info": {"brand": "现代", "model": "伊兰特", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，参数化格栅，贯穿尾灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "现代", "model": "索纳塔", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，贯穿灯带，溜背造型，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "现代", "model": "途胜L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，参数化格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "现代", "model": "胜达", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型SUV，H型灯带，行李架，电动尾门，全景天窗",
        ),
    },
    # ============================ 起亚 Kia (4) ============================
    {
        "vehicle_info": {"brand": "起亚", "model": "K3", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，虎啸格栅，普通大灯，无天窗，家用代步",
        ),
    },
    {
        "vehicle_info": {"brand": "起亚", "model": "K5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，虎啸格栅，贯穿尾灯，普通天窗，溜背造型",
        ),
    },
    {
        "vehicle_info": {"brand": "起亚", "model": "狮铂拓界", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，虎啸格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "起亚", "model": "嘉华", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="sliding",
            notes="中大型MPV，虎啸格栅，双侧电动滑门，电动尾门，分段式天窗",
        ),
    },
    # ============================ 马自达 Mazda (3) ============================
    {
        "vehicle_info": {"brand": "马自达", "model": "马自达3 昂克赛拉", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，魂动设计，普通天窗，运动风格",
        ),
    },
    {
        "vehicle_info": {"brand": "马自达", "model": "CX-5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，魂动设计，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "马自达", "model": "CX-50", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，魂动设计，行李架，电动尾门，全景天窗",
        ),
    },
    # ============================ 标致 Peugeot (4) ============================
    {
        "vehicle_info": {"brand": "标致", "model": "408", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，狮吼前脸，獠牙日行灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "标致", "model": "508", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，狮吼前脸，獠牙日行灯，普通天窗，溜背造型",
        ),
    },
    {
        "vehicle_info": {"brand": "标致", "model": "4008", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，狮吼前脸，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "标致", "model": "5008", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型七座SUV，狮吼前脸，行李架，电动尾门，全景天窗",
        ),
    },
    # ============================ 雪铁龙 Citroen (2) ============================
    {
        "vehicle_info": {"brand": "雪铁龙", "model": "凡尔赛C5 X", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="wagon",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型跨界旅行车，X型前脸，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "雪铁龙", "model": "天逸C5 AIRCROSS", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，分体式大灯，行李架，电动尾门，全景天窗",
        ),
    },
    # ============================ 斯柯达 Skoda (3) ============================
    {
        "vehicle_info": {"brand": "斯柯达", "model": "明锐", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="hatchback",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型掀背轿车，直瀑格栅，普通天窗，电动尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "斯柯达", "model": "柯迪亚克", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型SUV，直瀑格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "斯柯达", "model": "速派", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿车，直瀑格栅，普通天窗，掀背尾门",
        ),
    },
    # ============================ 沃尔沃 Volvo (5) ============================
    {
        "vehicle_info": {"brand": "沃尔沃", "model": "S60", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型豪华轿车，雷神之锤大灯，普通天窗，北欧简约",
        ),
    },
    {
        "vehicle_info": {"brand": "沃尔沃", "model": "S90", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型豪华轿车，雷神之锤大灯，分段式全景天窗，北欧简约",
        ),
    },
    {
        "vehicle_info": {"brand": "沃尔沃", "model": "XC60", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华SUV，雷神之锤大灯，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "沃尔沃", "model": "XC90", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型豪华SUV，雷神之锤大灯，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "沃尔沃", "model": "V60", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="wagon",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华旅行车，雷神之锤大灯，行李架，电动尾门，全景天窗",
        ),
    },
    # ============================ 凯迪拉克 Cadillac (6) ============================
    {
        "vehicle_info": {"brand": "凯迪拉克", "model": "CT5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型豪华轿车，盾形格栅，直列式大灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "凯迪拉克", "model": "CT6", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型豪华轿车，盾形格栅，直列式大灯，分段式全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "凯迪拉克", "model": "XT4", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型豪华SUV，盾形格栅，直列式大灯，行李架，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "凯迪拉克", "model": "XT5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华SUV，盾形格栅，直列式大灯，行李架，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "凯迪拉克", "model": "XT6", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型豪华SUV，盾形格栅，直列式大灯，行李架，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "凯迪拉克", "model": "LYRIQ锐歌", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型纯电豪华SUV，黑晶光曜格栅，直列式大灯，全景天幕",
        ),
    },
    # ============================ 林肯 Lincoln (4) ============================
    {
        "vehicle_info": {"brand": "林肯", "model": "Z", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型豪华轿车，星辉格栅，贯穿尾灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "林肯", "model": "冒险家", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型豪华SUV，星辉格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "林肯", "model": "航海家", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华SUV，星辉格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "林肯", "model": "飞行家", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型豪华SUV，星辉格栅，行李架，电动尾门，全景天窗",
        ),
    },
    # ============================ 捷豹 Jaguar (3) ============================
    {
        "vehicle_info": {"brand": "捷豹", "model": "XEL", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型豪华轿车，大嘴格栅，J型日行灯，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "捷豹", "model": "XFL", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型豪华轿车，大嘴格栅，J型日行灯，分段式天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "捷豹", "model": "F-PACE", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华SUV，大嘴格栅，J型日行灯，行李架，全景天窗",
        ),
    },
    # ============================ 路虎 Land Rover (4) ============================
    {
        "vehicle_info": {"brand": "路虎", "model": "揽胜极光L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型豪华SUV，蚌壳式引擎盖，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "路虎", "model": "发现运动版", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华SUV，蚌壳式引擎盖，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "路虎", "model": "揽胜", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="大型豪华SUV，蚌壳式引擎盖，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "路虎", "model": "卫士", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="round",
            rear_door_type="tailgate",
            notes="中大型越野SUV，圆灯设计，行李架，侧开尾门，全景天窗",
        ),
    },
    # ============================ 保时捷 Porsche (5) ============================
    {
        "vehicle_info": {"brand": "保时捷", "model": "Macan", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华SUV，蛙眼大灯，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "保时捷", "model": "Cayenne", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型豪华SUV，蛙眼大灯，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "保时捷", "model": "Panamera", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="大型豪华轿跑，蛙眼大灯，溜背造型，分段式全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "保时捷", "model": "Taycan", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型纯电轿跑，四点式大灯，溜背造型，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "保时捷", "model": "911", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="coupe",
            doors=2,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="round",
            rear_door_type="trunk_lid",
            notes="跑车，经典圆灯，后置引擎，溜背造型，普通天窗",
        ),
    },
    # ============================ 玛莎拉蒂 Maserati (2) ============================
    {
        "vehicle_info": {"brand": "玛莎拉蒂", "model": "Ghibli", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型豪华轿跑，大嘴格栅，三叉戟徽标，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "玛莎拉蒂", "model": "Levante", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型豪华SUV，大嘴格栅，三叉戟徽标，行李架，全景天窗",
        ),
    },
    # ============================ 宾利 Bentley (2) ============================
    {
        "vehicle_info": {"brand": "宾利", "model": "飞驰", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="round",
            rear_door_type="trunk_lid",
            notes="大型豪华轿车，圆灯设计，飞翼徽标，分段式全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "宾利", "model": "添越", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="round",
            rear_door_type="tailgate",
            notes="中大型豪华SUV，圆灯设计，飞翼徽标，行李架，全景天窗",
        ),
    },
    # ============================ 劳斯莱斯 Rolls-Royce (2) ============================
    {
        "vehicle_info": {"brand": "劳斯莱斯", "model": "古思特", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="大型豪华轿车，帕特农格栅，欢庆女神，分段式全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "劳斯莱斯", "model": "库里南", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="大型豪华SUV，帕特农格栅，欢庆女神，行李架，全景天窗",
        ),
    },
    # ============================ 兰博基尼 Lamborghini (1) ============================
    {
        "vehicle_info": {"brand": "兰博基尼", "model": "Urus", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型豪华SUV，Y型日行灯，公牛徽标，电动尾门，全景天窗",
        ),
    },
    # ============================ 法拉利 Ferrari (1) ============================
    {
        "vehicle_info": {"brand": "法拉利", "model": "Roma", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="coupe",
            doors=2,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="跑车，优雅流线设计，跃马徽标，溜背造型，普通天窗",
        ),
    },
    # ============================ 迈凯伦 McLaren (1) ============================
    {
        "vehicle_info": {"brand": "迈凯伦", "model": "Artura", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="coupe",
            doors=2,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="跑车，蝴蝶门设计，混动超跑，溜背造型，普通天窗",
        ),
    },
    # ============================ smart (2) ============================
    {
        "vehicle_info": {"brand": "smart", "model": "精灵#1", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="小型纯电SUV，圆润造型，贯穿尾灯，全景天幕，电动尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "smart", "model": "精灵#3", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型纯电轿跑SUV，圆润造型，溜背设计，贯穿尾灯，全景天幕",
        ),
    },
    # ============================ MINI (2) ============================
    {
        "vehicle_info": {"brand": "MINI", "model": "COOPER", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="hatchback",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="round",
            rear_door_type="tailgate",
            notes="小型豪华两厢车，经典圆灯，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "MINI", "model": "COUNTRYMAN", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="round",
            rear_door_type="tailgate",
            notes="紧凑型豪华SUV，经典圆灯，行李架，电动尾门，全景天窗",
        ),
    },
    # ============================ Jeep (2) ============================
    {
        "vehicle_info": {"brand": "Jeep", "model": "牧马人", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="round",
            rear_door_type="tailgate",
            notes="中型越野SUV，七孔格栅，圆灯设计，可拆卸车顶，行李架",
        ),
    },
    {
        "vehicle_info": {"brand": "Jeep", "model": "大切诺基", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型豪华SUV，七孔格栅，行李架，电动尾门，全景天窗",
        ),
    },
    # ============================ 吉利 Geely (6) ============================
    {
        "vehicle_info": {"brand": "吉利", "model": "星瑞", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，能量音弦格栅，普通天窗，家用轿车",
        ),
    },
    {
        "vehicle_info": {"brand": "吉利", "model": "星越L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，直瀑格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "吉利", "model": "帝豪", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，水滴涟漪格栅，普通大灯，无天窗，家用代步",
        ),
    },
    {
        "vehicle_info": {"brand": "吉利", "model": "博越L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，光波涟漪格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "吉利", "model": "银河L7", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="tailgate",
            notes="紧凑型插混SUV，贯穿灯带，电动尾门，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "吉利", "model": "缤越", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="小型SUV，星空回想格栅，电动尾门，普通天窗",
        ),
    },
    # ============================ 极氪 Zeekr (3) ============================
    {
        "vehicle_info": {"brand": "极氪", "model": "001", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="wagon",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型纯电猎装车，分体式大灯，电动尾门，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "极氪", "model": "007", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型纯电轿车，贯穿灯幕，溜背造型，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "极氪", "model": "009", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="sliding",
            notes="中大型纯电豪华MPV，镀铬格栅，双侧电动滑门，电动尾门，分段式天幕",
        ),
    },
    # ============================ 领克 Lynk & Co (4) ============================
    {
        "vehicle_info": {"brand": "领克", "model": "03", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型运动轿车，北极之光日行灯，普通天窗，运动风格",
        ),
    },
    {
        "vehicle_info": {"brand": "领克", "model": "08", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型插混SUV，破晓之光日行灯，电动尾门，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "领克", "model": "09", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型插混SUV，直瀑格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "领克", "model": "06", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="小型SUV，北极之光日行灯，电动尾门，普通天窗",
        ),
    },
    # ============================ 长安 Changan (9) ============================
    {
        "vehicle_info": {"brand": "长安", "model": "CS75 PLUS", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="贯穿式",
            rear_door_type="tailgate",
            notes="紧凑型SUV，无边界格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "长安", "model": "CS55 PLUS", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，无边界格栅，电动尾门，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "长安", "model": "逸动", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，无边界格栅，普通天窗，家用轿车",
        ),
    },
    {
        "vehicle_info": {"brand": "长安", "model": "UNI-V", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿跑，无边界格栅，电动尾翼，普通天窗，运动风格",
        ),
    },
    {
        "vehicle_info": {"brand": "长安", "model": "UNI-K", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型SUV，无边界格栅，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "长安", "model": "深蓝SL03", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型纯电/增程轿车，星能之瓣大灯，溜背造型，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "长安", "model": "深蓝S07", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型纯电/增程SUV，星能之瓣大灯，电动尾门，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "长安", "model": "阿维塔11", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型纯电SUV，碟翼式前脸，电动尾门，全景天幕，华为智驾",
        ),
    },
    {
        "vehicle_info": {"brand": "长安", "model": "阿维塔12", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型纯电轿车，碟翼式前脸，溜背造型，全景天幕，华为智驾",
        ),
    },
    # ============================ 长城 Great Wall (7) ============================
    {
        "vehicle_info": {"brand": "长城", "model": "哈弗H6", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，星轨格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "长城", "model": "哈弗大狗", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="round",
            rear_door_type="tailgate",
            notes="紧凑型SUV，复古圆灯，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "长城", "model": "坦克300", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="round",
            rear_door_type="tailgate",
            notes="紧凑型越野SUV，复古圆灯，硬派方盒子，行李架，侧开尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "长城", "model": "坦克500", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型越野SUV，粗壮镀铬格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "长城", "model": "魏牌蓝山", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型六座SUV，梯形格栅，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "长城", "model": "欧拉好猫", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="hatchback",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="round",
            rear_door_type="tailgate",
            notes="小型纯电两厢车，复古圆灯，圆润造型，电动尾门，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "长城", "model": "长城炮", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="pickup",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型皮卡，粗壮格栅，货箱尾门，普通天窗",
        ),
    },
    # ============================ 奇瑞 Chery (5) ============================
    {
        "vehicle_info": {"brand": "奇瑞", "model": "瑞虎8", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型七座SUV，虎啸格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奇瑞", "model": "瑞虎9", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型SUV，直瀑格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奇瑞", "model": "艾瑞泽8", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，纺锤格栅，普通天窗，家用轿车",
        ),
    },
    {
        "vehicle_info": {"brand": "奇瑞", "model": "捷途旅行者", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型越野SUV，硬派方盒子，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "奇瑞", "model": "星纪元ET", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="tailgate",
            notes="中大型纯电SUV，贯穿灯带，电动尾门，全景天幕",
        ),
    },
    # ============================ 红旗 Hongqi (4) ============================
    {
        "vehicle_info": {"brand": "红旗", "model": "H5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型豪华轿车，直瀑格栅，红旗立标，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "红旗", "model": "H9", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型豪华轿车，直瀑格栅，红旗立标，分段式全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "红旗", "model": "E-QM5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型纯电轿车，直瀑格栅，红旗立标，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "红旗", "model": "HS5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型豪华SUV，直瀑格栅，红旗立标，行李架，全景天窗",
        ),
    },
    # ============================ 传祺 Trumpchi (4) ============================
    {
        "vehicle_info": {"brand": "传祺", "model": "M8", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="sliding",
            notes="中大型MPV，醒狮前脸，双侧电动滑门，电动尾门，分段式天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "传祺", "model": "M6", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型MPV，展翼格栅，电动尾门，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "传祺", "model": "GS8", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型七座SUV，龙鳞翼格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "传祺", "model": "影豹", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型运动轿车，战机前脸，普通天窗，运动风格",
        ),
    },
    # ============================ 北汽 BAIC (2) ============================
    {
        "vehicle_info": {"brand": "北汽", "model": "EU5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型纯电轿车，封闭式格栅，普通大灯，无天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "北汽", "model": "魔方", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，无边界格栅，电动尾门，普通天窗",
        ),
    },
    # ============================ 名爵 MG (3) ============================
    {
        "vehicle_info": {"brand": "名爵", "model": "MG7", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型轿跑，大嘴格栅，无框车门，电动尾翼，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "名爵", "model": "MG4", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="hatchback",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型纯电两厢车，X型前脸，电动尾门，普通天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "名爵", "model": "MG ZS", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="小型SUV，星辉格栅，行李架，电动尾门，普通天窗",
        ),
    },
    # ============================ 荣威 Roewe (3) ============================
    {
        "vehicle_info": {"brand": "荣威", "model": "RX5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=True,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型SUV，凤羽格栅，行李架，电动尾门，全景天窗",
        ),
    },
    {
        "vehicle_info": {"brand": "荣威", "model": "i5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="紧凑型轿车，荣麟格栅，普通大灯，无天窗，家用代步",
        ),
    },
    {
        "vehicle_info": {"brand": "荣威", "model": "D7", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中型纯电/插混轿车，电感化前脸，全景天幕，家用轿车",
        ),
    },
    # ============================ 哪吒 Neta (3) ============================
    {
        "vehicle_info": {"brand": "哪吒", "model": "哪吒S", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型纯电/增程轿跑，展翼式前脸，溜背造型，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "哪吒", "model": "哪吒L", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型纯电/增程SUV，展翼式前脸，电动尾门，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "哪吒", "model": "哪吒GT", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="coupe",
            doors=2,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="双门纯电跑车，展翼式前脸，无框车门，溜背造型，全景天幕",
        ),
    },
    # ============================ 零跑 Leapmotor (4) ============================
    {
        "vehicle_info": {"brand": "零跑", "model": "C11", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="tailgate",
            notes="中型纯电/增程SUV，贯穿灯带，电动尾门，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "零跑", "model": "C10", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="tailgate",
            notes="中型纯电/增程SUV，贯穿灯带，电动尾门，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "零跑", "model": "T03", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="hatchback",
            doors=5,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="微型纯电两厢车，圆润造型，无天窗，电动尾门，代步小车",
        ),
    },
    {
        "vehicle_info": {"brand": "零跑", "model": "C16", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="tailgate",
            notes="中大型六座纯电/增程SUV，贯穿灯带，电动尾门，全景天幕",
        ),
    },
    # ============================ 岚图 Voyah (3) ============================
    {
        "vehicle_info": {"brand": "岚图", "model": "FREE", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="tailgate",
            notes="中大型增程/纯电SUV，贯穿灯带，电动尾门，可升降全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "岚图", "model": "梦想家", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="mpv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="sliding",
            notes="中大型豪华MPV，贯穿灯带，双侧电动滑门，电动尾门，分段式天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "岚图", "model": "追光", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="trunk_lid",
            notes="中大型纯电轿车，贯穿灯带，溜背造型，全景天幕",
        ),
    },
    # ============================ 智己 IM (3) ============================
    {
        "vehicle_info": {"brand": "智己", "model": "L7", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型纯电轿车，Sigma前脸，可编程大灯，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "智己", "model": "LS6", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型纯电轿跑SUV，Sigma前脸，电动尾门，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "智己", "model": "LS7", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型纯电SUV，Sigma前脸，电动尾门，前穹顶玻璃",
        ),
    },
    # ============================ 问界 AITO (3) ============================
    {
        "vehicle_info": {"brand": "问界", "model": "M5", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型增程/纯电SUV，梯形格栅，电动尾门，全景天幕，华为智驾",
        ),
    },
    {
        "vehicle_info": {"brand": "问界", "model": "M7", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中大型增程SUV，梯形格栅，电动尾门，全景天幕，华为智驾",
        ),
    },
    {
        "vehicle_info": {"brand": "问界", "model": "M9", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="贯穿式",
            rear_door_type="tailgate",
            notes="大型豪华增程/纯电SUV，贯穿灯带，电动尾门，全景天幕，华为智驾",
        ),
    },
    # ============================ 小米 Xiaomi (1) ============================
    {
        "vehicle_info": {"brand": "小米", "model": "SU7", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型纯电轿跑，水滴大灯，溜背造型，全景天幕，运动风格",
        ),
    },
    # ============================ 极狐 Arcfox (2) ============================
    {
        "vehicle_info": {"brand": "极狐", "model": "阿尔法S", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="sedan",
            doors=4,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="trunk_lid",
            notes="中大型纯电轿车，X型前脸，溜背造型，全景天幕",
        ),
    },
    {
        "vehicle_info": {"brand": "极狐", "model": "阿尔法T", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="suv",
            doors=5,
            has_sunroof=True,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="中型纯电SUV，X型前脸，电动尾门，全景天幕",
        ),
    },
    # ============================ 五菱 Wuling (3) ============================
    {
        "vehicle_info": {"brand": "五菱", "model": "宏光MINIEV", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="hatchback",
            doors=3,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="微型纯电两厢车，方盒子造型，双门四座，无天窗，入门级代步车",
        ),
    },
    {
        "vehicle_info": {"brand": "五菱", "model": "五菱荣光", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="van",
            doors=5,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="微型面包车，实用造型，无天窗，侧滑门+对开尾门",
        ),
    },
    {
        "vehicle_info": {"brand": "五菱", "model": "五菱宏光", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="van",
            doors=5,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="紧凑型MPV/面包车，实用造型，无天窗，侧滑门+上掀尾门",
        ),
    },
    # ============================ 其他 Other (1) ============================
    {
        "vehicle_info": {"brand": "长安", "model": "长安之星", "year": "2024"},
        "specs": VehicleSpecs(
            body_style="van",
            doors=5,
            has_sunroof=False,
            has_roof_rack=False,
            headlight_layout="separate",
            rear_door_type="tailgate",
            notes="微型面包车，实用造型，无天窗，侧滑门+对开尾门",
        ),
    },
]


def seed_cache(clear: bool = False) -> int:
    """Iterate over COMMON_CHINESE_VEHICLES and save each to the cache.

    Returns the number of entries written.
    """
    if clear:
        _save_cache({})
        print("Cache cleared.")

    count = 0
    for entry in COMMON_CHINESE_VEHICLES:
        vehicle_info = entry["vehicle_info"]
        specs = entry["specs"]
        save_cached_specs(vehicle_info, specs)
        count += 1
        print(f"  Seeded: {vehicle_info['brand']} {vehicle_info['model']} ({vehicle_info['year']})")

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed vehicle specs cache with common Chinese-market vehicles.")
    parser.add_argument("--clear", action="store_true", help="Clear the cache before seeding")
    args = parser.parse_args()

    count = seed_cache(clear=args.clear)
    print(f"\nDone. Seeded {count} entries into {CACHE_FILE}")


if __name__ == "__main__":
    main()
