"""API views for vehicle damage assessment.

Replaces the previous FastAPI backend.py while keeping the same endpoints and
SSE event-stream contract.  These views are synchronous so they work under
Django's default WSGI development server; async agent calls are bridged with
``async_to_sync``.
"""

import asyncio
import json
import os
import uuid
from typing import Any, Dict, Generator, List

from asgiref.sync import async_to_sync, sync_to_async
from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

from agents import (
    assessment_orchestrator_stream,
    build_vehicle_topology,
    damage_assessor_agent,
    extract_vehicle_info_from_auxiliary_photos,
    photo_locator_agent,
    validate_and_enrich,
    vehicle_prior_agent,
)
from agents.view_mapping import (
    EXTERIOR_VIEWS,
    NON_EXTERIOR_VIEWS,
    STANDARD_VIEWS,
    get_display_name,
    get_regions_for_view,
    is_exterior_view,
)
from api.models import UploadedPhoto, UploadedTask
from config import MAX_CONCURRENT_API_CALLS, PHOTO_LOCATOR_BATCH_SIZE


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


def health(request):
    """Liveness probe — 返回 ok 用于 load balancer / k8s readiness check。"""
    return JsonResponse({"status": "ok"})


def index(request):
    """Serve the built-in HTML debug console."""
    from django.shortcuts import render
    return render(request, "index.html")


def _parse_view_override(raw: str) -> Dict[str, str]:
    """Parse the ``views`` query string into a ``{photo_id: view_id}`` map.

    Format: ``photo1:front_left,photo2:front_right,...``
    Whitespace around keys/values is ignored.  Invalid view ids are dropped.
    """
    mapping: Dict[str, str] = {}
    if not raw:
        return mapping
    for pair in raw.split(","):
        if ":" not in pair:
            continue
        photo_id, view_id = pair.split(":", 1)
        photo_id = photo_id.strip()
        view_id = view_id.strip().lower()
        if photo_id and view_id in STANDARD_VIEWS:
            mapping[photo_id] = view_id
    return mapping


def _build_plan_from_view_override(
    files: List[Dict[str, Any]], view_override: Dict[str, str]
) -> Dict[str, Any]:
    """Build a planner-compatible plan from a manual photo->view mapping.

    Photos not present in the override keep their default ``unknown`` view.
    Coverage gaps are computed only for exterior views that remain empty.
    """
    file_by_id = {f["id"]: f for f in files}

    # Build photo_views: honour override, default everything else to unknown.
    photo_views: List[Dict[str, Any]] = []
    for f in files:
        photo_id = f["id"]
        view_id = view_override.get(photo_id, "unknown")
        is_overridden = photo_id in view_override
        photo_views.append(
            {
                "photo_id": photo_id,
                "view_id": view_id,
                "confidence": "high" if is_overridden else "low",
                "reason": "用户手动指定" if is_overridden else "",
            }
        )

    # Build view_groups with exterior photos only.
    groups: Dict[str, List[Dict[str, Any]]] = {view: [] for view in STANDARD_VIEWS}
    for entry in photo_views:
        view_id = entry["view_id"]
        photo_id = entry["photo_id"]
        photo = file_by_id.get(photo_id)
        if photo is None:
            continue
        enriched = dict(photo)
        enriched["_planner_view"] = view_id
        enriched["_planner_confidence"] = entry["confidence"]
        enriched["_planner_reason"] = entry["reason"]
        groups.setdefault(view_id, []).append(enriched)

    coverage_gaps: List[Dict[str, Any]] = []
    for view_id in EXTERIOR_VIEWS:
        if not groups.get(view_id):
            regions = get_regions_for_view(view_id)
            coverage_gaps.append(
                {
                    "missing_view": view_id,
                    "display_name": get_display_name(view_id),
                    "impacted_regions": regions,
                    "impacted_parts": [],
                    "suggested_action": f"补拍{get_display_name(view_id)}照片",
                }
            )

    priority_views = [v for v, g in groups.items() if g and is_exterior_view(v)]
    missing_critical_views = [g["missing_view"] for g in coverage_gaps]

    return {
        "photo_views": photo_views,
        "view_groups": groups,
        "coverage_gaps": coverage_gaps,
        "workflow_plan": {
            "summary": f"已覆盖外观视角：{', '.join(priority_views) or '无'}",
            "priority_views": priority_views,
            "missing_critical_views": missing_critical_views,
        },
    }


@csrf_exempt
def upload_files(request):
    """Receive uploaded vehicle photos and return a task_id.

    Matches the legacy FastAPI ``POST /api/upload`` endpoint.
    """
    if request.method != "POST":
        return JsonResponse({"message": "Method not allowed"}, status=405)

    brand = request.POST.get("brand", "")
    model = request.POST.get("model", "")
    year = request.POST.get("year", "")

    task = UploadedTask.objects.create(brand=brand, model=model, year=year)

    files = request.FILES.getlist("files")
    saved_count = 0
    for uploaded_file in files:
        filename = os.path.basename(uploaded_file.name)
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
        UploadedPhoto.objects.create(
            task=task,
            photo_id=filename,
            file=uploaded_file,
        )
        saved_count += 1

    return JsonResponse(
        {
            "task_id": str(task.task_id),
            "uploaded_count": saved_count,
            "vehicle_info": {"brand": brand, "model": model, "year": year},
        }
    )


def assess_stream(request, task_id: str):
    """SSE endpoint that runs the full assessment workflow.

    Matches the legacy FastAPI ``GET /api/assess/<task_id>`` endpoint.
    """
    brand = request.GET.get("brand", "")
    model = request.GET.get("model", "")
    year = request.GET.get("year", "")
    vehicle_info = {"brand": brand, "model": model, "year": year}

    try:
        uuid.UUID(task_id)
    except ValueError:
        return StreamingHttpResponse(
            _error_stream("任务ID格式无效"),
            content_type="text/event-stream",
        )

    task = None
    try:
        task = UploadedTask.objects.get(task_id=task_id)
    except UploadedTask.DoesNotExist:
        pass

    files = []
    if task is not None:
        for photo in task.photos.all():
            files.append(
                {
                    "id": photo.photo_id,
                    "path": photo.file_path,
                    "name": photo.photo_id,
                    "url": photo.file_url,
                }
            )

    # Backward compatibility: fall back to the filesystem for tasks created
    # by the previous FastAPI backend.
    if not files:
        task_dir = os.path.join(settings.MEDIA_ROOT, task_id)
        if os.path.isdir(task_dir):
            for filename in sorted(os.listdir(task_dir)):
                if not filename.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                file_path = os.path.join(task_dir, filename)
                file_url = f"{settings.BASE_URL}{settings.MEDIA_URL}{task_id}/{filename}"
                files.append(
                    {
                        "id": filename,
                        "path": file_path,
                        "name": filename,
                        "url": file_url,
                    }
                )

    if not files:
        return StreamingHttpResponse(
            _error_stream("任务不存在"),
            content_type="text/event-stream",
        )

    # Optional manual view override: ?views=photo1:front_left,photo2:front_right
    view_override = _parse_view_override(request.GET.get("views", ""))
    explicit_plan = None
    if view_override:
        explicit_plan = _build_plan_from_view_override(files, view_override)

    # Feature flag: use new orchestrator by default; allow fallback to legacy pipeline.
    use_orchestrator = request.GET.get("legacy", "").lower() not in ("true", "1", "yes")
    if use_orchestrator:
        return StreamingHttpResponse(
            _orchestrator_workflow_sync(files, vehicle_info, plan=explicit_plan),
            content_type="text/event-stream",
        )
    return StreamingHttpResponse(
        _assess_workflow_sync(files, vehicle_info),
        content_type="text/event-stream",
    )


def _sse_event(event_type: str, data: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error_stream(message: str) -> Generator[str, None, None]:
    yield _sse_event("error", {"message": message})


def _part_state_to_dict(part_state: Any) -> Dict[str, Any]:
    """Serialize a PartActualState (or already-dict value) to a frontend dict.

    Uses :meth:`models.part_state.PartActualState.to_dict` for list-shape
    `damage_type` / `evidence_photo` keys, which is what the frontend
    progressive renderer expects.
    """
    try:
        from models.part_state import PartActualState
        if isinstance(part_state, PartActualState):
            return part_state.to_dict()
    except Exception:
        pass
    if isinstance(part_state, dict):
        # Already a dict — coerce list-shape fields if string.
        d = dict(part_state)
        dt = d.get("damage_type")
        if isinstance(dt, str):
            d["damage_type"] = [s.strip() for s in dt.split(",") if s.strip()] or ["none"]
        ep = d.get("evidence_photo")
        if isinstance(ep, str):
            d["evidence_photo"] = [s.strip() for s in ep.split(",") if s.strip()]
        return d
    return {}


def _orchestrator_workflow_sync(
    files: List[Dict[str, Any]],
    vehicle_info: Dict[str, str],
    plan: Dict[str, Any] | None = None,
) -> Generator[str, None, None]:
    """Synchronous wrapper around the new orchestrator async workflow.

    Streams events back to the client in real time — we drive the async
    generator from a dedicated event loop on a worker thread, yielding each
    SSE event as soon as the async side produces it.  Without this, a naive
    ``async_to_sync(_collect_orchestrator_events)`` call would buffer every
    event until the async generator finished, defeating the purpose of SSE.

    修复(2026-07-04):之前的实现 `yield from async_to_sync(...)` 把所有事件
    收集到 list 才一次性 yield,导致浏览器看到进度条"卡住"。现在用
    bridge_thread + asyncio.run 让每个 SSE 事件立刻到达浏览器。
    """
    import asyncio
    import threading
    from queue import Queue, Empty

    q: "Queue[str | None]" = Queue()

    async def _bridge() -> None:
        try:
            async for event in _run_orchestrator_workflow(files, vehicle_info, plan=plan):
                q.put(event)
        except Exception as exc:  # propagate async errors as SSE error event
            q.put(_sse_event("error", {"message": str(exc)}))
        finally:
            q.put(None)  # sentinel: done

    def _run_bridge() -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_bridge())
        finally:
            loop.close()

    worker = threading.Thread(target=_run_bridge, daemon=True)
    worker.start()

    while True:
        try:
            item = q.get()
        except Empty:
            continue
        if item is None:
            break
        yield item


async def _collect_orchestrator_events(
    files: List[Dict[str, Any]],
    vehicle_info: Dict[str, str],
    plan: Dict[str, Any] | None = None,
) -> List[str]:
    """Consume the async orchestrator generator and return SSE event strings."""
    events = []
    async for event in _run_orchestrator_workflow(files, vehicle_info, plan=plan):
        events.append(event)
    return events


async def _run_orchestrator_workflow(
    files: List[Dict[str, Any]],
    vehicle_info: Dict[str, str],
    plan: Dict[str, Any] | None = None,
) -> Generator[str, None, None]:
    """Async generator that yields SSE events from the orchestrator workflow."""
    from agents.output_validator import validate_and_enrich
    from agents.planner_agent import get_coverage_summary, plan_to_location_map

    try:
        yield _sse_event(
            "step",
            {
                "step": 0,
                "name": "开始识别",
                "message": f"共 {len(files)} 张照片（多Agent编排）",
                "total": len(files),
            },
        )

        yield _sse_event(
            "step",
            {
                "step": 1,
                "name": "车型先验",
                "message": f"正在加载 {vehicle_info.get('brand', '')} {vehicle_info.get('model', '')} 的标准外观特征...",
            },
        )
        vehicle_prior = await vehicle_prior_agent(vehicle_info)
        topology = build_vehicle_topology(vehicle_info, vehicle_prior)
        vehicle_prior_with_topology = dict(vehicle_prior)
        vehicle_prior_with_topology["topology_model"] = topology.to_dict()
        yield _sse_event("vehicle_prior", vehicle_prior_with_topology)
        yield _sse_event("topology", {"topology_model": topology.to_dict()})

        yield _sse_event(
            "step",
            {
                "step": 2,
                "name": "视角规划",
                "message": "Planner Agent 正在为所有照片分配标准视角...",
            },
        )
        from agents.planner_agent import planner_agent
        if plan is None:
            plan = await planner_agent(files, vehicle_prior)
        coverage_summary = get_coverage_summary(plan)
        yield _sse_event(
            "locations",
            {
                "locations": list(plan_to_location_map(plan).values()),
                "coverage_summary": coverage_summary,
            },
        )

        yield _sse_event(
            "step",
            {
                "step": 3,
                "name": "视觉识别",
                "message": f"正在并发派发 {coverage_summary['covered_view_count']} 个 Vision Subagent...",
            },
        )

        # ---- Progressive streaming: consume the orchestrator stream so each
        # Vision Subagent completion is pushed to the client as it happens.
        # 渐进渲染 (2026-07-04): 每个 view subagent 完成后立刻推送 subagent_partial,
        # reviewer 完成推 review_partial,最终 result 事件保持兼容。
        orchestrator_result = None
        review: Dict[str, Any] = {}
        async for event in assessment_orchestrator_stream(files, vehicle_info, plan=plan):
            event_type = event.get("type")
            if event_type == "subagent_complete":
                sub_result = event.get("serializable_result", {})
                view_id = sub_result.get("view_id", "unknown")
                sub_parts = sub_result.get("parts", []) or []
                serialized_parts = [
                    _part_state_to_dict(p) for p in sub_parts
                    if p is not None and (isinstance(p, dict) or hasattr(p, "part_id"))
                ]
                yield _sse_event(
                    "subagent_partial",
                    {
                        "view_id": view_id,
                        "parts": serialized_parts,
                        "uncertain_items": sub_result.get("uncertain_items", []) or [],
                        "additional_findings": sub_result.get("additional_findings", []) or [],
                    },
                )
            elif event_type == "review":
                review = event.get("review", {})
                reviewed_states = review.get("reviewed_part_actual_states", []) or []
                serialized_reviewed = [_part_state_to_dict(p) for p in reviewed_states]
                yield _sse_event(
                    "review_partial",
                    {
                        "reviewed_parts": serialized_reviewed,
                        "summary": review.get("summary", ""),
                        "needs_rephotography": review.get("needs_rephotography", []) or [],
                    },
                )
            elif event_type == "final":
                orchestrator_result = event.get("result")

        yield _sse_event(
            "step",
            {
                "step": 4,
                "name": "复核审查",
                "message": "Reviewer Subagent 正在检查覆盖缺口和冲突...",
            },
        )
        yield _sse_event("review", {"review_summary": review.get("summary", "")})

        yield _sse_event(
            "step",
            {
                "step": 5,
                "name": "生成报告",
                "message": "正在校验输出并生成最终报告...",
            },
        )
        final_result = dict(orchestrator_result)
        final_result["vehicle_info"] = vehicle_info
        final_result["plan"] = plan
        final_result["review"] = review

        yield _sse_event("result", final_result)
        yield _sse_event("complete", {"message": "识别完成"})

    except Exception as e:
        yield _sse_event("error", {"message": str(e)})


def _assess_workflow_sync(
    files: List[Dict[str, Any]],
    vehicle_info: Dict[str, str],
) -> Generator[str, None, None]:
    """Synchronous wrapper around the async agent workflow."""
    yield from async_to_sync(_collect_assess_events)(files, vehicle_info)


async def _collect_assess_events(
    files: List[Dict[str, Any]],
    vehicle_info: Dict[str, str],
) -> List[str]:
    """Consume the async generator and return a list of SSE event strings."""
    events = []
    async for event in _run_assess_workflow(files, vehicle_info):
        events.append(event)
    return events


async def _run_assess_workflow(
    files: List[Dict[str, Any]],
    vehicle_info: Dict[str, str],
) -> Generator[str, None, None]:
    """Async generator that yields SSE events.

    This is the original workflow logic; it is consumed by the synchronous
    ``_assess_workflow_sync`` wrapper so the view can run under WSGI.
    """

    try:
        yield _sse_event(
            "step",
            {
                "step": 0,
                "name": "开始识别",
                "message": f"共 {len(files)} 张照片",
                "total": len(files),
            },
        )

        # Try to infer vehicle info from auxiliary photos (license, VIN, etc.)
        auxiliary_photos = [
            {"id": f["id"], "path": f["path"]}
            for f in files
            if f["name"].lower().startswith(("行驶证", "证件", "vin", "铭牌"))
            or any(kw in f["name"].lower() for kw in ["license", "vin", "plate", "证", "牌", "铭牌"])
        ]
        # Fallback: when no explicit auxiliary filename, include the first few
        # photos in case they contain license/VIN/logos. The extractor itself
        # will ignore non-text photos.
        if not auxiliary_photos and not all(vehicle_info.values()):
            auxiliary_photos = [{"id": f["id"], "path": f["path"]} for f in files[:3]]

        if auxiliary_photos and not all(vehicle_info.values()):
            inferred = await extract_vehicle_info_from_auxiliary_photos(auxiliary_photos)
            if inferred.get("brand") and not vehicle_info.get("brand"):
                vehicle_info["brand"] = inferred["brand"]
            if inferred.get("model") and not vehicle_info.get("model"):
                vehicle_info["model"] = inferred["model"]
            if inferred.get("year") and not vehicle_info.get("year"):
                vehicle_info["year"] = inferred["year"]

        yield _sse_event(
            "step",
            {
                "step": 1,
                "name": "车型先验",
                "message": f"正在加载 {vehicle_info.get('brand', '')} {vehicle_info.get('model', '')} 的标准外观特征...",
            },
        )
        vehicle_prior = await vehicle_prior_agent(vehicle_info)

        topology = build_vehicle_topology(vehicle_info, vehicle_prior)

        vehicle_prior_with_topology = dict(vehicle_prior)
        vehicle_prior_with_topology["topology_model"] = topology.to_dict()
        yield _sse_event("vehicle_prior", vehicle_prior_with_topology)
        yield _sse_event("topology", {"topology_model": topology.to_dict()})

        yield _sse_event(
            "step",
            {
                "step": 2,
                "name": "照片定位",
                "message": "正在判断每张照片的拍摄视角...",
            },
        )

        batches = [
            files[i : i + PHOTO_LOCATOR_BATCH_SIZE]
            for i in range(0, len(files), PHOTO_LOCATOR_BATCH_SIZE)
        ]

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

        async def locate_batch(batch):
            async with semaphore:
                return await photo_locator_agent(batch, vehicle_prior)

        batch_results = await asyncio.gather(*[locate_batch(b) for b in batches])

        all_locations = []
        for result in batch_results:
            if isinstance(result, list):
                all_locations.extend(result)

        location_map = {loc.get("photo_id"): loc for loc in all_locations}
        for f in files:
            if f["id"] not in location_map:
                location_map[f["id"]] = {
                    "photo_id": f["id"],
                    "location": "无法定位",
                    "location_detail": "未返回定位结果",
                    "primary_anchor": "无",
                    "confidence": "low",
                    "reason": "模型未返回该照片定位",
                    "visible_parts": [],
                }

        yield _sse_event("locations", {"locations": list(location_map.values())})

        yield _sse_event(
            "step",
            {
                "step": 3,
                "name": "损伤评估",
                "message": "正在结合多张照片评估每个部件的损伤情况...",
            },
        )
        damage_result = await damage_assessor_agent(
            files, list(location_map.values()), vehicle_prior, topology
        )

        yield _sse_event(
            "step",
            {
                "step": 4,
                "name": "生成报告",
                "message": "正在校验输出并生成最终报告...",
            },
        )
        final_result = validate_and_enrich(damage_result, topology)
        final_result["vehicle_info"] = vehicle_info

        yield _sse_event("result", final_result)
        yield _sse_event("complete", {"message": "识别完成"})

    except Exception as e:
        yield _sse_event("error", {"message": str(e)})
