# 车辆损伤识别系统

基于 MiniMax 多模态模型 + Rubric 三层架构的车辆外部损伤识别 Demo。

## 技术栈

- **后端**: Django 4.2 + Django REST framework + django-cors-headers
- **Agent 工作流**: master_assessment_agent = face_profiler → face_mapping → view_agent (face path, 默认) → synthesizer → topology_comparator;legacy view_agent 仍可通过 `?face_path=false` 切换
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
├── data/                       # 缓存与数据文件（vehicle_specs_cache.json 由 app 写入，已 gitignore）
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

把 `.env.example` 复制成 `.env` 后填入真实值（最少要填 `MINIMAX_API_KEY` 和 `DJANGO_SECRET_KEY`）：

```bash
cp .env.example .env
$EDITOR .env
```

所有可调参数（`IMAGE_MAX_WIDTH`、`MAX_CONCURRENT_API_CALLS`、`PHOTO_LOCATOR_BATCH_SIZE` 等）在 `.env.example` 里有说明。

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

假设上传 27 张照片，使用默认 face path：

| Agent | 调用次数 | 说明 |
|-------|----------|------|
| VehiclePriorAgent | 1 次 | 文本调用 |
| PlannerAgent (vision classify) | 2-4 次 | 每批 8 张照片，按外/内/辅分类 |
| FaceProfilerAgent | 2-3 次 | 每批 8 张照片，识别朝向/侧 |
| ViewAgent (×27 外饰照片) | 27 次 | 每张照片一次（face path 锁住候选集） |
| Synthesizer/Reviewer | 0 次 | 纯本地规则 |
| **总计** | **约 32-35 次** | 全流程 |

legacy view path（`?face_path=false`）仍可用，但会产生 6-7 次 LLM 调用 + 更少的稳定性。

## 输出格式

详见 `models/assessment.py` 和 `agents/output_validator.py`，输出包含：

- `vehicle_info`: 车辆信息
- `assessment_summary`: 整体评估摘要
- `parts`: 每个部件的损伤状态
- `uncertain_items`: 不确定项和复核建议
- `structural_damage_reasoning`: 结构性事故触发原因
- `topology_model`: 标况拓扑模型
- `structural_patterns`: 结构性损伤模式

## 切换 LLM Provider

UI 顶部"LLM 接入"卡片可以临时切换到任意 OpenAI 兼容或 Anthropic 兼容端点，无需重启服务或修改 `.env`。值只保存到浏览器 `localStorage`，不会写回服务端。

| Provider | Base URL | Auth | Model 例子 |
|---|---|---|---|
| MiniMax（默认） | `https://api.minimaxi.com/v1/chat/completions`（留空也行） | `Authorization: Bearer` | `MiniMax-M3` |
| OpenAI 兼容 | `https://api.openai.com/v1/chat/completions` | `Authorization: Bearer` | `gpt-4o`、`gpt-4-vision-preview` |
| Anthropic 兼容 | `https://api.anthropic.com/v1/messages` | `x-api-key` | `claude-3-5-sonnet-latest` |

**用法示例 — OpenAI**：

```
Provider:    OpenAI 兼容
Base URL:    https://api.openai.com/v1/chat/completions
Model:       gpt-4o
API Key:     sk-...
```

**用法示例 — Anthropic**：

```
Provider:    Anthropic 兼容
Base URL:    https://api.anthropic.com/v1/messages
Model:       claude-3-5-sonnet-latest
API Key:     sk-ant-...
```

实现细节：

- `agents/llm_client.py` 是协议无关的 provider 抽象层（MiniMax / OpenAI 兼容 / Anthropic 兼容）。
- `agents/minimax_client.py::call_minimax` 在请求级路由：`LLM_PROVIDER=minimax`（默认）走原来的重试 + token 升级循环（针对 MiniMax M3 的 `<think>` 截断做了特殊处理）；其他 provider 委托给 `llm_client`。
- `api/views.py::_LLMOverride` 是一个上下文管理器，在 SSE 响应期间 swap `LLM_PROVIDER` 环境变量 + `MINIMAX_API_KEY`/`MINIMAX_BASE_URL`/`MINIMAX_MODEL` config 属性，结束后自动还原。
- `clean_minimax_output` 对 OpenAI/Anthropic 的返回也会跑（剥离 `<think>` 块和 markdown 围栏），所以响应可以直接进 `extract_json`。
- 单元测试 572 个继续通过（没有改任何 agent 调用代码）。

## 注意事项

1. 首次使用前请确认 MiniMax API Key 有效，且有 vision 模型调用权限。
2. 照片数量较多时，识别时间约 20-40 秒，请耐心等待。
3. 当前为 Demo 版本，主要用于验证 Rubric 和工作流，生产环境需要增加错误重试、限流、日志等。
4. 本项目为纯后端 API，前端需独立部署并通过 CORS 调用接口。

## 已知问题

1. **API key 由用户自填**：仓库不携带任何 `MINIMAX_API_KEY`；`.env` 已在 `.gitignore` 内，但用户必须**自己**在 `.env` 填入。如果曾在 fork 仓库里检查到任何真实 key，请**立即轮换**。前端可在 UI 顶部表单里临时切换不同的 provider key（见 `templates/index.html`）。
2. **无逐张进度（SSE）**：`assessment_orchestrator_stream` 当前只 yield 一个 `final` 事件，前端不会按照片逐张刷新。完整的 per-photo 渲染需要把 `master_assessment_agent` 重构成 async generator。修复前每次识别都是"黑盒 20-40 秒后出结果"。
3. **`master_agent.py:74` 的 `use_face_path` 默认 `True`**：直接调用 `master_assessment_agent` 的脚本/测试必须显式传 `use_face_path=False` 才能回到 legacy view path；稳定性测试场景下要小心。
4. **部分函数/文件超长**：`master_agent.py` (915 行)、`view_agent.py` (720 行)、`templates/index.html` (985 行) 都超过了 800 行的项目惯例阈值。后续重构时按子步骤拆分。

## 交付验收清单

收到仓库后，建议按下面顺序完成 bootstrap：

1. `cp .env.example .env` 并填入真实 `MINIMAX_API_KEY` 与 `DJANGO_SECRET_KEY`
2. `pip install -r requirements.txt`
3. `python manage.py migrate`
4. `python manage.py runserver` 启动后端
5. `cd stability_reports && python run_facepath_172852.py new` 跑 3 次 E2E 验证
6. 检查 `<BASE_DIR>/logs/*.log` 下 7 个 agent 日志都正常写入（master / orchestrator / planner / view / view_trace / faceprofiler / vision）
