from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image

from backend.app.schemas.inference import AnalyzeResponse
from backend.app.services.vit_inference import predict_image


router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
)
async def analyze(
    hospital_id: str = Form(...),
    case_id: str | None = Form(None),
    model_name: str | None = Form(None),
    file: UploadFile = File(...),
) -> AnalyzeResponse:

    if not file.content_type or not file.content_type.startswith(
        "image/"
    ):
        raise ValueError("Uploaded file must be an image.")

    image_bytes = await file.read()

    from io import BytesIO

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    result = predict_image(image)

    return AnalyzeResponse(
        case_id=case_id or "generated-case",
        predicted_class=result["predicted_class"],
        model_probability=result["model_probability"],
        calibration_status="not_calibrated",
        disclaimer=(
            "AI-assisted research prototype; "
            "this output is not a diagnosis. "
            "A qualified clinician must review "
            "the image and clinical context."
        ),
        is_diagnosis=False,
    )