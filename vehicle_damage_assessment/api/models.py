import os
import uuid
from pathlib import Path

from django.db import models


def task_upload_path(instance, filename):
    """Return MEDIA_ROOT/uploads/<task_id>/<filename>."""
    return os.path.join(str(instance.task.task_id), os.path.basename(filename))


class UploadedTask(models.Model):
    """A single vehicle assessment request."""

    task_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    brand = models.CharField(max_length=100, blank=True)
    model = models.CharField(max_length=100, blank=True)
    year = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "uploaded_task"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.brand} {self.model} {self.year} ({self.task_id})"


class UploadedPhoto(models.Model):
    """A photo uploaded as part of an assessment task."""

    task = models.ForeignKey(
        UploadedTask,
        related_name="photos",
        on_delete=models.CASCADE,
    )
    photo_id = models.CharField(max_length=255)
    file = models.ImageField(upload_to=task_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "uploaded_photo"
        ordering = ["photo_id"]
        unique_together = [["task", "photo_id"]]

    @property
    def file_url(self) -> str:
        from django.conf import settings

        if self.file and self.file.url:
            return f"{settings.BASE_URL}{self.file.url}"
        return ""

    @property
    def file_path(self) -> str:
        if self.file:
            return str(Path(self.file.path).resolve())
        return ""

    def __str__(self) -> str:
        return f"{self.task.task_id}/{self.photo_id}"


class VehicleSpec(models.Model):
    """Cached vehicle specification used for topology adaptation."""

    cache_key = models.CharField(max_length=255, unique=True)
    brand = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    year = models.CharField(max_length=20)
    body_style = models.CharField(max_length=50)
    doors = models.IntegerField()
    has_sunroof = models.BooleanField()
    has_roof_rack = models.BooleanField()
    headlight_layout = models.CharField(max_length=50)
    rear_door_type = models.CharField(max_length=50)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "vehicle_spec"
        ordering = ["brand", "model", "year"]

    def __str__(self) -> str:
        return f"{self.brand} {self.model} {self.year}"

    def to_dict(self) -> dict:
        return {
            "body_style": self.body_style,
            "doors": self.doors,
            "has_sunroof": self.has_sunroof,
            "has_roof_rack": self.has_roof_rack,
            "headlight_layout": self.headlight_layout,
            "rear_door_type": self.rear_door_type,
            "notes": self.notes,
        }
