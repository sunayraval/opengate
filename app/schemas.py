from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ActionInferenceRequest(BaseModel):
    """
    JSON request schema for action inference when sending base64 images
    instead of multipart/form-data file uploads.
    """
    candidate_actions: List[str] = Field(
        ...,
        description="List of candidate natural language actions for the robot (e.g., ['move forward', 'turn left', 'stop']).",
        example=["move forward", "turn left", "turn right", "stop"]
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional model identifier to use. Defaults to DEFAULT_ACTION_MODEL if not specified."
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded image string (JPEG or PNG). Optional if image is provided via multipart upload."
    )
    image_url: Optional[str] = Field(
        default=None,
        description="HTTP/HTTPS URL link of an image to download and analyze."
    )


class ActionInferenceResponse(BaseModel):
    """
    Response schema for action prediction inference.
    """
    action: str = Field(..., description="The top selected action predicted by the vision-language model.")
    confidence: float = Field(..., description="Confidence score (probability) of the top predicted action (0.0 to 1.0).")
    all_scores: Dict[str, float] = Field(..., description="Mapping of all candidate actions to their computed confidence scores.")
    model_used: str = Field(..., description="The name of the model that executed the inference.")
    inference_time_ms: float = Field(..., description="Inference execution time in milliseconds.")


class BoundingBox(BaseModel):
    """
    Schema for a single detected object bounding box.
    Coordinates are typically normalized (0.0 to 1.0) or in pixel values depending on model output.
    """
    xmin: float = Field(..., description="Left coordinate of the bounding box.")
    ymin: float = Field(..., description="Top coordinate of the bounding box.")
    xmax: float = Field(..., description="Right coordinate of the bounding box.")
    ymax: float = Field(..., description="Bottom coordinate of the bounding box.")
    confidence: float = Field(..., description="Detection confidence score (0.0 to 1.0).")
    class_name: str = Field(..., description="Detected object class label (e.g., 'person', 'obstacle', 'ball').")
    class_id: Optional[int] = Field(default=None, description="Integer class ID from the model vocabulary.")


class DetectionInferenceResponse(BaseModel):
    """
    Response schema for object detection inference (e.g., YOLOv8).
    """
    bounding_boxes: List[BoundingBox] = Field(
        default_factory=list,
        description="List of detected object bounding boxes."
    )
    model_used: str = Field(..., description="The name of the detection model used.")
    inference_time_ms: float = Field(..., description="Inference execution time in milliseconds.")


class HealthResponse(BaseModel):
    """
    Response schema for system health, GPU diagnostics, and model registry status.
    """
    status: str = Field(default="ok", description="Overall system status (e.g., 'ok' or 'degraded').")
    gpu_available: bool = Field(..., description="True if a CUDA-enabled GPU is detected and available.")
    gpu_name: Optional[str] = Field(default=None, description="Name of the detected GPU device (e.g., 'NVIDIA RTX 4090').")
    loaded_models: List[str] = Field(..., description="List of model names currently loaded into memory/VRAM.")
    vram_used_mb: float = Field(..., description="Current GPU VRAM memory allocated by PyTorch in megabytes.")


class ModelInfo(BaseModel):
    """
    Schema representing information about an available or loaded AI model.
    """
    name: str = Field(..., description="Unique model identifier.")
    task_type: str = Field(..., description="Task category (e.g., 'action_prediction', 'object_detection').")
    is_loaded: bool = Field(..., description="Whether the model weights are currently loaded into memory.")
    device: str = Field(default="cpu", description="Device where model is loaded ('cuda' or 'cpu').")


class CompletionMessage(BaseModel):
    """
    Message schema for OpenAI-compatible chat completion requests.
    """
    role: str = Field(default="user", description="Message role ('user', 'assistant', or 'system').")
    content: Any = Field(..., description="Message text string OR a list of multimodal content blocks (e.g., text and image_url).")


class CompletionRequest(BaseModel):
    """
    JSON request schema for Vision-Language AI completion and chat endpoints.
    Supports OpenAI-compatible message formats and base64 image sending.
    """
    model: Optional[str] = Field(default=None, alias="model_name", description="Model identifier to use (e.g. 'openbmb/MiniCPM-V').")
    prompt: Optional[str] = Field(default=None, description="Simple text prompt string for completion.")
    messages: Optional[List[CompletionMessage]] = Field(default=None, description="OpenAI-style chat messages list.")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image string (JPEG or PNG). Optional if image is passed via multipart upload or inside messages.")
    image_url: Optional[str] = Field(default=None, description="HTTP/HTTPS URL link of an image to download and analyze.")
    temperature: float = Field(default=0.7, description="Sampling temperature (0.0 for greedy deterministic decoding).")
    max_tokens: int = Field(default=512, alias="max_new_tokens", description="Maximum number of tokens to generate.")
    stream: bool = Field(default=False, description="Whether to stream response tokens (currently returns full completion).")

    class Config:
        populate_by_name = True


class CompletionChoice(BaseModel):
    """
    Single choice item in completion response.
    """
    index: int = Field(default=0, description="Choice index.")
    text: str = Field(..., description="Generated completion text.")
    message: Dict[str, str] = Field(..., description="OpenAI-style message dictionary with role and content.")
    finish_reason: str = Field(default="stop", description="Reason generation finished.")


class CompletionUsage(BaseModel):
    """
    Token usage statistics.
    """
    prompt_tokens: int = Field(default=0, description="Number of tokens in prompt.")
    completion_tokens: int = Field(default=0, description="Number of tokens generated.")
    total_tokens: int = Field(default=0, description="Total tokens used.")


class CompletionResponse(BaseModel):
    """
    Response schema for Vision-Language AI completion and chat endpoints.
    """
    id: str = Field(..., description="Unique completion ID.")
    object: str = Field(default="chat.completion", description="Object type identifier.")
    created: int = Field(..., description="Unix timestamp of completion creation.")
    model: str = Field(..., description="Model identifier used for generation.")
    choices: List[CompletionChoice] = Field(..., description="List of completion choices.")
    usage: CompletionUsage = Field(..., description="Token usage statistics.")
    inference_time_ms: float = Field(..., description="Inference execution latency in milliseconds.")
    raw_response: str = Field(..., description="Raw text output generated by the vision-language model.")
