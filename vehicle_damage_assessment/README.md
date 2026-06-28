# 车辆损伤识别系统

基于 MiniMax 多模态模型 + Rubric 三层架构的车辆外部损伤识别 Demo。

## 技术栈

- **后端**: Django 4.2 + Django REST framework + django-cors-headers
- **Agent 工作流**: vehicle_prior → photo_locator → damage_assessor → output_validator
- **LLM**: MiniMax-M3 多模态模型
- **数据库**: SQLite（默认，生产可切 PostgreSQL）

## 项目结构

```
vehicle_damage_assessment/
├── core/                       # Django 配置
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── manage.py                   # Django 入口
├── api/                        # API 应用
│   ├── models.py               # UploadedTask, UploadedPhoto, VehicleSpec
│   ├── views.py                # /api/upload, /api/assess/<task_id>
│   └── management/commands/    # seed_vehicle_specs 等命令
├── agents/                     # 子代理（框架无关）
├── models/                     # dataclass 模型
├── data/                       # 缓存与数据文件
├── scripts/                    # 独立脚本
├── tests/                      # pytest 测试
└── uploads/                    # 上传照片存储目录（MEDIA_ROOT）
```

## 快速启动

### 1. 安装依赖

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export MINIMAX_API_KEY="你的 MiniMax API Key"
```

可选：

```bash
export MINIMAX_MODEL="MiniMax-M3"  # 默认
export MINIMAX_BASE_URL="https://api.minimaxi.com/v1/chat/completions"
export DJANGO_SECRET_KEY="your-secret-key-for-production"
export DEBUG="True"
```

### 3. 初始化数据库与缓存

```bash
python manage.py migrate
python manage.py seed_vehicle_specs   # 导入 236 款常见车型数据
```

### 4. 启动服务

开发环境（WSGI，默认端口 8000）：

```bash
python manage.py runserver 0.0.0.0:8000
```

生产环境建议使用 ASGI：

```bash
daphne -b 0.0.0.0 -p 8000 core.asgi:application
```

## API 接口

### 上传照片

```bash
POST /api/upload
Content-Type: multipart/form-data

fields:
  - files: 多张图片（.png .jpg .jpeg .webp .gif .bmp）
  - brand: 品牌
  - model: 车系
  - year: 年款
```

返回：

```json
{
  "task_id": "uuid",
  "uploaded_count": 12,
  "vehicle_info": {"brand": "蔚来", "model": "ES8", "year": "2024"}
}
```

### 识别评估（SSE 流式）

```bash
GET /api/assess/<task_id>?brand=蔚来&model=ES8&year=2024
Accept: text/event-stream
```

SSE 事件类型：

- `step`: 识别进度
- `vehicle_prior`: 车型先验结果
- `topology`: 标况拓扑模型
- `locations`: 照片定位结果
- `result`: 最终损伤评估结果
- `complete`: 识别完成
- `error`: 错误信息

## 测试

```bash
pytest tests/ -q
```

## 模型调用次数

假设上传 27 张照片：

| Agent | 调用次数 | 说明 |
|-------|----------|------|
| VehiclePriorAgent | 1 次 | 文本调用 |
| PhotoLocatorAgent | 4-5 次 | 每批 6-7 张照片，并行执行 |
| DamageAssessorAgent | 1 次 | 综合推理 |
| **总计** | **约 6-7 次** | 全流程 |

## 输出格式

详见 `models/assessment.py` 和 `agents/output_validator.py`，输出包含：

- `vehicle_info`: 车辆信息
- `assessment_summary`: 整体评估摘要
- `parts`: 每个部件的损伤状态
- `uncertain_items`: 不确定项和复核建议
- `structural_damage_reasoning`: 结构性事故触发原因
- `topology_model`: 标况拓扑模型
- `structural_patterns`: 结构性损伤模式

## 注意事项

1. 首次使用前请确认 MiniMax API Key 有效，且有 vision 模型调用权限。
2. 照片数量较多时，识别时间约 20-40 秒，请耐心等待。
3. 当前为 Demo 版本，主要用于验证 Rubric 和工作流，生产环境需要增加错误重试、限流、日志等。
4. 本项目为纯后端 API，前端需独立部署并通过 CORS 调用接口。
