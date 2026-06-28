"""URL configuration for the API app."""

from django.urls import path

from api.views import assess_stream, index, upload_files

urlpatterns = [
    path("", index, name="index"),
    path("api/upload", upload_files, name="upload_files"),
    path("api/assess/<str:task_id>", assess_stream, name="assess_stream"),
]
