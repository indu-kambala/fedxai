from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    hospital_id: str = Field(..., description="Client that retains the raw image")
    case_id: str | None = None
    model_name: str | None = None
    # The image is uploaded via multipart in later phases; this JSON contract
    # documents that the central FL aggregator is not the inference upload path.


class AnalyzeResponse(BaseModel):
    case_id: str
    predicted_class: str
    model_probability: float
    calibration_status: str
    disclaimer: str
    is_diagnosis: bool = False
