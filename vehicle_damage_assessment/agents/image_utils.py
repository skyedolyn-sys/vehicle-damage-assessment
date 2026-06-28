import io
import base64
from pathlib import Path
from typing import Tuple

try:
    from PIL import Image
except ImportError:
    Image = None


def compress_image_to_base64(
    image_path: str,
    max_width: int = 1024,
    quality: int = 85,
    format: str | None = None,
) -> Tuple[str, str]:
    """
    压缩图片并返回 base64 data URI 和 mime type。

    Args:
        image_path: 图片路径
        max_width: 最大宽度，等比缩放
        quality: JPEG 质量
        format: 强制输出格式，None 则保持原格式

    Returns:
        (data_uri, mime_type)
    """
    if Image is None:
        raise RuntimeError("Pillow is required for image compression")

    img = Image.open(image_path)

    # 处理方向（EXIF Orientation）
    try:
        from PIL import ExifTags
        exif = img._getexif()
        if exif:
            orientation = exif.get(274)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass

    # 等比缩放
    width, height = img.size
    if width > max_width:
        ratio = max_width / width
        new_size = (max_width, int(height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # 确定输出格式
    ext = Path(image_path).suffix.lower()
    if format is None:
        if ext in [".jpg", ".jpeg"]:
            output_format = "JPEG"
            mime = "image/jpeg"
        elif ext == ".png":
            output_format = "PNG"
            mime = "image/png"
        elif ext == ".webp":
            output_format = "WEBP"
            mime = "image/webp"
        else:
            output_format = "JPEG"
            mime = "image/jpeg"
    else:
        output_format = format.upper()
        mime = f"image/{output_format.lower()}"

    # 统一转 RGB（JPEG 不支持透明）
    if output_format == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    buffer = io.BytesIO()
    if output_format == "JPEG":
        img.save(buffer, format=output_format, quality=quality, optimize=True)
    else:
        img.save(buffer, format=output_format, optimize=True)

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    return data_uri, mime
