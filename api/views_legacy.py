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
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from agents import (
    build_vehicle_topology,
    damage_assessor_agent,
    extract_vehicle_info_from_auxiliary_photos,
    photo_locator_agent,
    validate_and_enrich,
    vehicle_prior_agent,
)
from api.models import UploadedPhoto, UploadedTask
from config import MAX_CONCURRENT_API_CALLS, PHOTO_LOCATOR_BATCH_SIZE


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


def index(request):
    """Serve the built-in HTML debug console."""
    return render(request, "index.html")


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

    return StreamingHttpResponse(
        _assess_workflow_sync(files, vehicle_info),
        content_type="text/event-stream",
    )


def _sse_event(event_type: str, data: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error_stream(message: str) -> Generator[str, None, None]:
    yield _sse_event("error", {"message": message})


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
