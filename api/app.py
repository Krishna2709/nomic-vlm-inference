"""
ColPali Multimodal Embedding API

A production-ready FastAPI service for generating embeddings from text and images
using the ColPali ColQwen2.5 model with LoRA adapter support.

This module provides:
- Text embeddings (col and dense variants)
- Image embeddings (col and dense variants)
- Offline model loading with LoRA adapter support
- Comprehensive error handling and validation
- Authentication support
- Health monitoring

Author: AI Engineering Team
License: MIT
"""

import base64
import io
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException, Header
from PIL import Image
from pydantic import BaseModel, Field
from transformers.utils.import_utils import is_flash_attn_2_available

# Set offline environment variables before importing ColPali
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from peft import PeftModel

MODEL_ID = os.getenv("MODEL_ID", "nomic-ai/colnomic-embed-multimodal-3b")
MODEL_REV = os.getenv("MODEL_REV")
MODEL_DIR = os.getenv("MODEL_DIR")  # Local model directory path
INTERNAL_KEY = os.getenv("INTERNAL_KEY")  # Cloudflare->origin shared secret
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
DTYPE = (
    torch.bfloat16
    if (DEVICE == "cuda" and torch.cuda.is_bf16_supported())
    else torch.float16
)

MAX_BATCH = int(os.getenv("MAX_BATCH_ITEMS", "64"))
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "2048"))

# Set device_map based on available hardware
device_map = (
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Use local model directory if available, otherwise fall back to MODEL_ID
model_path = MODEL_DIR if MODEL_DIR and os.path.exists(MODEL_DIR) else MODEL_ID
model_info = model_path + (f"@{MODEL_REV}" if MODEL_REV and not MODEL_DIR else "")

# Check if this is a LoRA adapter model and handle accordingly
is_lora_model = False
base_model_path = None

if MODEL_DIR and os.path.exists(MODEL_DIR):
    adapter_config_path = os.path.join(MODEL_DIR, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        import json

        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path")
        if base_model and os.path.exists("/models/base"):
            is_lora_model = True
            base_model_path = "/models/base"
            print(f"Detected LoRA adapter model. Base model: {base_model}")
        else:
            print(f"Warning: Base model {base_model} not found locally at /models/base")

# Load the model - handle LoRA adapter if needed
if is_lora_model:
    print("Loading base model and LoRA adapter...")
    # Load base model first
    base_model = ColQwen2_5.from_pretrained(
        base_model_path,
        local_files_only=True,
        torch_dtype=DTYPE,
        device_map=device_map,
        attn_implementation=(
            "flash_attention_2" if is_flash_attn_2_available() else None
        ),
    )
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, MODEL_DIR, local_files_only=True)
    model = model.eval()
else:
    # Load regular model
    model = ColQwen2_5.from_pretrained(
        model_path,
        local_files_only=True,
        revision=(
            MODEL_REV if not MODEL_DIR else None
        ),  # Don't use revision for local models
        torch_dtype=DTYPE,
        device_map=device_map,
        attn_implementation=(
            "flash_attention_2" if is_flash_attn_2_available() else None
        ),
    ).eval()

# Load processor
processor = ColQwen2_5_Processor.from_pretrained(model_path, local_files_only=True)


class TextInput(BaseModel):
    texts: List[str] = Field(..., description="List of text inputs")


class ImageInput(BaseModel):
    image_b64: Optional[List[str]] = None  # base64 PNG/JPEG list
    image_urls: Optional[List[str]] = None  # (disabled on serverless without egress)


class EmbedOptions(BaseModel):
    variant: str = Field("col", description="'col' (multi-vector) or 'dense'")
    normalize: bool = True


class EmbedRequest(BaseModel):
    input: Union[TextInput, ImageInput]
    options: EmbedOptions = EmbedOptions()


class EmbedResponse(BaseModel):
    model: str
    variant: str
    data: List[Any]  # More flexible to handle both dense and col variants
    usage: dict


app = FastAPI(title="ColNomic Multimodal Embeddings", version="1.0")


def _auth(x_internal_key: Optional[str]) -> None:
    """
    Authenticate requests using internal key.

    Args:
        x_internal_key: The internal key provided in the request header

    Raises:
        HTTPException: If authentication fails (401 Unauthorized)
    """
    if INTERNAL_KEY and x_internal_key != INTERNAL_KEY:
        raise HTTPException(401, "unauthorized")


def _normalize(t: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor using L2 normalization.

    Args:
        t: Input tensor to normalize

    Returns:
        L2 normalized tensor
    """
    return torch.nn.functional.normalize(t, p=2, dim=-1)


def _pool_to_dense(mv: torch.Tensor) -> torch.Tensor:
    """
    Convert multi-vector embeddings to dense embeddings by averaging.

    Args:
        mv: Multi-vector tensor (either 3D or 2D)

    Returns:
        Dense tensor (averaged across vector dimension)
    """
    return mv.mean(dim=1) if mv.dim() == 3 else mv.mean(dim=0, keepdim=True)


def _load_b64(imgs: List[str]) -> List[Image.Image]:
    """
    Load images from base64 encoded strings.

    Args:
        imgs: List of base64 encoded image strings

    Returns:
        List of PIL Image objects

    Raises:
        HTTPException: If image data is invalid (400 Bad Request)
    """
    out: List[Image.Image] = []
    for b in imgs:
        try:
            # Decode base64 and open image
            img_data = base64.b64decode(b)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")

            # Ensure minimum size to avoid processing issues
            min_size = 32  # Minimum 32x32 pixels
            if img.size[0] < min_size or img.size[1] < min_size:
                # Resize small images to minimum size
                img = img.resize((min_size, min_size), Image.Resampling.LANCZOS)

            out.append(img)
        except Exception as e:
            raise HTTPException(400, f"Invalid image data: {str(e)}")
    return out


@app.get("/healthz")
def healthz() -> Dict[str, Union[bool, str]]:
    """
    Health check endpoint.

    Returns:
        Dictionary containing service status, model info, device, and offline status
    """
    return {
        "ok": True,
        "model": model_info,
        "device": DEVICE,
        "offline": bool(MODEL_DIR),
    }


@app.post("/v1/embed", response_model=EmbedResponse)
def embed(
    req: EmbedRequest, x_internal_key: Optional[str] = Header(None)
) -> EmbedResponse:
    """
    Generate embeddings for text or images.

    This endpoint supports both text and image inputs, with two embedding variants:
    - 'col': Multi-vector embeddings (default)
    - 'dense': Single dense vector embeddings

    Args:
        req: Embedding request containing input data and options
        x_internal_key: Optional authentication key

    Returns:
        EmbedResponse containing embeddings, metadata, and usage information

    Raises:
        HTTPException: For various error conditions (400, 401, 413, 500)
    """
    _auth(x_internal_key)
    t0 = time.time()
    mv_outputs = []

    if isinstance(req.input, TextInput):
        texts = req.input.texts
        if not texts or len(texts) > MAX_BATCH:
            raise HTTPException(413, f"Batch 1..{MAX_BATCH}")
        if any(len(t) > MAX_TEXT_LEN for t in texts):
            raise HTTPException(413, f"text too long")
        batch = processor.process_queries(texts).to(model.device)
        with torch.no_grad():
            mv = model(**batch)
        mv_outputs = list(mv) if isinstance(mv, (list, tuple)) else [mv]

    else:
        if not (req.input.image_b64 or req.input.image_urls):
            raise HTTPException(400, "provide image_b64")
        if req.input.image_urls:
            raise HTTPException(400, "image_urls disabled; send base64")
        try:
            imgs = _load_b64(req.input.image_b64 or [])
            if not imgs or len(imgs) > MAX_BATCH:
                raise HTTPException(413, f"Batch 1..{MAX_BATCH}")

            # Validate image sizes and formats
            for i, img in enumerate(imgs):
                if img.size[0] < 1 or img.size[1] < 1:
                    raise HTTPException(
                        400, f"Image {i+1} has invalid dimensions: {img.size}"
                    )
                if img.mode != "RGB":
                    raise HTTPException(
                        400, f"Image {i+1} must be RGB format, got: {img.mode}"
                    )

            # Process images with the ColPali processor
            batch = processor.process_images(imgs).to(model.device)
            with torch.no_grad():
                mv = model(**batch)
            mv_outputs = list(mv) if isinstance(mv, (list, tuple)) else [mv]
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            raise HTTPException(500, f"Image processing error: {str(e)}")

    res = []
    for item in mv_outputs:
        v = _normalize(item) if req.options.normalize else item
        if req.options.variant == "dense":
            v = _pool_to_dense(v).squeeze(0)
            res.append(v.float().cpu().tolist())
        else:
            res.append(v.float().cpu().tolist())

    usage = {
        "latency_ms": int((time.time() - t0) * 1000),
        "batch_size": len(mv_outputs),
        "variant": req.options.variant,
        "request_id": str(uuid.uuid4()),
    }
    return EmbedResponse(
        model=model_info,
        variant=req.options.variant,
        data=res,
        usage=usage,
    )
