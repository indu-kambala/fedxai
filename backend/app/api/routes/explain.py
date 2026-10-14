from io import BytesIO

from fastapi import APIRouter, File, UploadFile
from PIL import Image

from backend.app.services.xai_service import (
    generate_explanation,
)


router = APIRouter()


@router.post("/explain")
async def explain(
    file: UploadFile = File(...),
) -> dict:

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        return {
            "error": "Uploaded file must be an image."
        }

    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    return generate_explanation(image)