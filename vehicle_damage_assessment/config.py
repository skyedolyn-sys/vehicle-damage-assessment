import os
from dotenv import load_dotenv

load_dotenv()

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1/chat/completions")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M3")

if not MINIMAX_API_KEY:
    raise ValueError("MINIMAX_API_KEY environment variable is required")

# 业务配置
PHOTO_LOCATOR_BATCH_SIZE = int(os.getenv("PHOTO_LOCATOR_BATCH_SIZE", "2"))
MAX_CONCURRENT_API_CALLS = int(os.getenv("MAX_CONCURRENT_API_CALLS", "2"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
IMAGE_MAX_WIDTH = int(os.getenv("IMAGE_MAX_WIDTH", "512"))
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "85"))

# 部件清单
PARTS_CATALOG = [
    # 前部
    {"part_id": "hood", "part_name": "引擎盖", "part_category": "front", "side": "center"},
    {"part_id": "bumper_front", "part_name": "前保险杠", "part_category": "front", "side": "center"},
    {"part_id": "headlight_front_left", "part_name": "左前大灯", "part_category": "front", "side": "front_left"},
    {"part_id": "headlight_front_right", "part_name": "右前大灯", "part_category": "front", "side": "front_right"},
    {"part_id": "grille_front", "part_name": "前格栅", "part_category": "front", "side": "center"},
    {"part_id": "fender_front_left", "part_name": "左前翼子板", "part_category": "front", "side": "front_left"},
    {"part_id": "fender_front_right", "part_name": "右前翼子板", "part_category": "front", "side": "front_right"},
    {"part_id": "windshield_front", "part_name": "前挡风玻璃", "part_category": "front", "side": "center"},
    # 后部
    {"part_id": "trunk_lid", "part_name": "后备箱盖", "part_category": "rear", "side": "center"},
    {"part_id": "tailgate", "part_name": "尾门", "part_category": "rear", "side": "center"},
    {"part_id": "bumper_rear", "part_name": "后保险杠", "part_category": "rear", "side": "center"},
    {"part_id": "taillight_rear_left", "part_name": "左后尾灯", "part_category": "rear", "side": "rear_left"},
    {"part_id": "taillight_rear_right", "part_name": "右后尾灯", "part_category": "rear", "side": "rear_right"},
    {"part_id": "windshield_rear", "part_name": "后挡风玻璃", "part_category": "rear", "side": "center"},
    # 左侧
    {"part_id": "door_front_left", "part_name": "左前门", "part_category": "left", "side": "front_left"},
    {"part_id": "door_rear_left", "part_name": "左后门", "part_category": "left", "side": "rear_left"},
    {"part_id": "mirror_left", "part_name": "左后视镜", "part_category": "left", "side": "front_left"},
    {"part_id": "fender_rear_left", "part_name": "左后翼子板", "part_category": "left", "side": "rear_left"},
    {"part_id": "pillar_a_left", "part_name": "左A柱", "part_category": "left", "side": "front_left"},
    {"part_id": "pillar_b_left", "part_name": "左B柱", "part_category": "left", "side": "center"},
    {"part_id": "pillar_c_left", "part_name": "左C柱", "part_category": "left", "side": "rear_left"},
    # 右侧
    {"part_id": "door_front_right", "part_name": "右前门", "part_category": "right", "side": "front_right"},
    {"part_id": "door_rear_right", "part_name": "右后门", "part_category": "right", "side": "rear_right"},
    {"part_id": "mirror_right", "part_name": "右后视镜", "part_category": "right", "side": "front_right"},
    {"part_id": "fender_rear_right", "part_name": "右后翼子板", "part_category": "right", "side": "rear_right"},
    {"part_id": "pillar_a_right", "part_name": "右A柱", "part_category": "right", "side": "front_right"},
    {"part_id": "pillar_b_right", "part_name": "右B柱", "part_category": "right", "side": "center"},
    {"part_id": "pillar_c_right", "part_name": "右C柱", "part_category": "right", "side": "rear_right"},
    # 车顶
    {"part_id": "roof_front", "part_name": "车顶前部", "part_category": "roof", "side": "center"},
    {"part_id": "roof_middle", "part_name": "车顶中部", "part_category": "roof", "side": "center"},
    {"part_id": "roof_rear", "part_name": "车顶后部", "part_category": "roof", "side": "center"},
    {"part_id": "sunroof_glass", "part_name": "天窗玻璃", "part_category": "roof", "side": "center"},
    {"part_id": "roof_rack", "part_name": "车顶行李架", "part_category": "roof", "side": "center"},
]

PARTS_BY_ID = {p["part_id"]: p for p in PARTS_CATALOG}

# Topology configuration for all 32 parts in PARTS_CATALOG.
# Defines adjacency (real vehicle geometry), node classification,
# and visibility angles for each part.
PARTS_TOPOLOGY = {
    "adjacency": {
        # ---- front ----
        "hood": ["grille_front", "fender_front_left", "fender_front_right", "windshield_front"],
        "bumper_front": ["grille_front", "fender_front_left", "fender_front_right", "headlight_front_left", "headlight_front_right"],
        "headlight_front_left": ["bumper_front", "fender_front_left", "grille_front"],
        "headlight_front_right": ["bumper_front", "fender_front_right", "grille_front"],
        "grille_front": ["hood", "bumper_front", "headlight_front_left", "headlight_front_right", "windshield_front"],
        "fender_front_left": ["hood", "bumper_front", "headlight_front_left", "door_front_left", "mirror_left"],
        "fender_front_right": ["hood", "bumper_front", "headlight_front_right", "door_front_right", "mirror_right"],
        "windshield_front": ["hood", "grille_front", "roof_front"],
        # ---- rear ----
        "trunk_lid": ["bumper_rear", "taillight_rear_left", "taillight_rear_right", "windshield_rear", "roof_rear"],
        "tailgate": ["bumper_rear", "windshield_rear", "roof_rear"],
        "bumper_rear": ["trunk_lid", "tailgate", "taillight_rear_left", "taillight_rear_right", "fender_rear_left", "fender_rear_right"],
        "taillight_rear_left": ["bumper_rear", "trunk_lid", "tailgate", "fender_rear_left"],
        "taillight_rear_right": ["bumper_rear", "trunk_lid", "tailgate", "fender_rear_right"],
        "windshield_rear": ["trunk_lid", "tailgate", "roof_rear"],
        # ---- left ----
        "door_front_left": ["fender_front_left", "door_rear_left", "mirror_left", "pillar_a_left"],
        "door_rear_left": ["door_front_left", "fender_rear_left", "pillar_b_left", "pillar_c_left"],
        "mirror_left": ["fender_front_left", "door_front_left", "pillar_a_left"],
        "fender_rear_left": ["door_rear_left", "bumper_rear", "taillight_rear_left", "pillar_c_left"],
        "pillar_a_left": ["fender_front_left", "door_front_left", "mirror_left", "roof_front", "windshield_front"],
        "pillar_b_left": ["door_front_left", "door_rear_left", "roof_middle", "pillar_a_left", "pillar_c_left"],
        "pillar_c_left": ["door_rear_left", "fender_rear_left", "roof_rear", "windshield_rear", "pillar_b_left"],
        # ---- right ----
        "door_front_right": ["fender_front_right", "door_rear_right", "mirror_right", "pillar_a_right"],
        "door_rear_right": ["door_front_right", "fender_rear_right", "pillar_b_right", "pillar_c_right"],
        "mirror_right": ["fender_front_right", "door_front_right", "pillar_a_right"],
        "fender_rear_right": ["door_rear_right", "bumper_rear", "taillight_rear_right", "pillar_c_right"],
        "pillar_a_right": ["fender_front_right", "door_front_right", "mirror_right", "roof_front", "windshield_front"],
        "pillar_b_right": ["door_front_right", "door_rear_right", "roof_middle", "pillar_a_right", "pillar_c_right"],
        "pillar_c_right": ["door_rear_right", "fender_rear_right", "roof_rear", "windshield_rear", "pillar_b_right"],
        # ---- roof ----
        "roof_front": ["windshield_front", "roof_middle", "sunroof_glass", "pillar_a_left", "pillar_a_right"],
        "roof_middle": ["roof_front", "roof_rear", "sunroof_glass", "roof_rack", "pillar_b_left", "pillar_b_right"],
        "roof_rear": ["roof_middle", "windshield_rear", "trunk_lid", "tailgate", "pillar_c_left", "pillar_c_right"],
        "sunroof_glass": ["roof_front", "roof_middle"],
        "roof_rack": ["roof_middle", "roof_front", "roof_rear"],
    },
    "node_types": {
        # ---- front ----
        "hood": "panel",
        "bumper_front": "panel",
        "headlight_front_left": "light",
        "headlight_front_right": "light",
        "grille_front": "trim",
        "fender_front_left": "panel",
        "fender_front_right": "panel",
        "windshield_front": "glass",
        # ---- rear ----
        "trunk_lid": "panel",
        "tailgate": "panel",
        "bumper_rear": "panel",
        "taillight_rear_left": "light",
        "taillight_rear_right": "light",
        "windshield_rear": "glass",
        # ---- left ----
        "door_front_left": "panel",
        "door_rear_left": "panel",
        "mirror_left": "glass",
        "fender_rear_left": "panel",
        "pillar_a_left": "structural",
        "pillar_b_left": "structural",
        "pillar_c_left": "structural",
        # ---- right ----
        "door_front_right": "panel",
        "door_rear_right": "panel",
        "mirror_right": "glass",
        "fender_rear_right": "panel",
        "pillar_a_right": "structural",
        "pillar_b_right": "structural",
        "pillar_c_right": "structural",
        # ---- roof ----
        "roof_front": "structural",
        "roof_middle": "structural",
        "roof_rear": "structural",
        "sunroof_glass": "glass",
        "roof_rack": "trim",
    },
    "visibility": {
        # ---- front ----
        "hood": ["front", "front_left", "front_right"],
        "bumper_front": ["front", "front_left", "front_right"],
        "headlight_front_left": ["front", "front_left", "left"],
        "headlight_front_right": ["front", "front_right", "right"],
        "grille_front": ["front", "front_left", "front_right"],
        "fender_front_left": ["front", "front_left", "left"],
        "fender_front_right": ["front", "front_right", "right"],
        "windshield_front": ["front", "front_left", "front_right", "top"],
        # ---- rear ----
        "trunk_lid": ["rear", "rear_left", "rear_right"],
        "tailgate": ["rear", "rear_left", "rear_right"],
        "bumper_rear": ["rear", "rear_left", "rear_right"],
        "taillight_rear_left": ["rear", "rear_left", "left"],
        "taillight_rear_right": ["rear", "rear_right", "right"],
        "windshield_rear": ["rear", "rear_left", "rear_right", "top"],
        # ---- left ----
        "door_front_left": ["left", "front_left", "rear_left"],
        "door_rear_left": ["left", "front_left", "rear_left"],
        "mirror_left": ["left", "front_left"],
        "fender_rear_left": ["left", "rear_left"],
        "pillar_a_left": ["left", "front_left", "top"],
        "pillar_b_left": ["left", "front_left", "rear_left", "top"],
        "pillar_c_left": ["left", "rear_left", "top"],
        # ---- right ----
        "door_front_right": ["right", "front_right", "rear_right"],
        "door_rear_right": ["right", "front_right", "rear_right"],
        "mirror_right": ["right", "front_right"],
        "fender_rear_right": ["right", "rear_right"],
        "pillar_a_right": ["right", "front_right", "top"],
        "pillar_b_right": ["right", "front_right", "rear_right", "top"],
        "pillar_c_right": ["right", "rear_right", "top"],
        # ---- roof ----
        "roof_front": ["front", "front_left", "front_right", "top"],
        "roof_middle": ["left", "right", "top"],
        "roof_rear": ["rear", "rear_left", "rear_right", "top"],
        "sunroof_glass": ["front", "front_left", "front_right", "left", "right", "top"],
        "roof_rack": ["front", "rear", "left", "right", "top"],
    },
}
