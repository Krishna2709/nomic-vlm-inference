import os, io, time, uuid, base64, json
from typing import List, Optional, Union
from pydantic import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn2_available

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

MODEL_ID  = os.getenv("MODEL_ID", "nomic-ai/colnomic-embed-multimodal-3b")
MODEL_REV = os.getenv("MODEL_REV")
INTERNAL_KEY = os.getenv("INTERNAL_KEY")  # Cloudflare->origin shared secret
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

MAX_BATCH = int(os.getenv("MAX_BATCH_ITEMS", "64"))
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "2048"))

attn_impl = "flash_attn_2" if is_flash_attn2_available() else None
model = ColQwen2_5.from_pretrained(
    MODEL_ID,
    revision=MODEL_REV,
    torch_dtype=DTYPE,
    device=DEVICE,
    attn_impl=attn_impl
)
processor = ColQwen2_5_Processor.from_pretrained(MODEL_ID)

class TextInput(BaseModel):
    texts: List[str] = Field(..., description="List of text inputs")

class ImageInput(BaseModel):
    image_b64: Optional[List[str]] = None  # base64 PNG/JPEG list
    image_urls: Optional[List[str]] = None # (disabled on serverless without egress)
    
class EmbedOptions(BaseModel):
    variant: str = Field("col", description="'col' (multi-vector) or 'dense'")
    normalize: bool = True

class EmbedRequest(BaseModel):
    input: Union[TextInput, ImageInput]
    options: EmbedOptions = EmbedOptions()

class EmbedResponse(BaseModel):
    model: str
    variant: str
    data: List[Union[List[List[float]], List[float]]]
    usage: dict

app = FastAPI(title="ColNomic Multimodal Embeddings", version="1.0")

def _auth(x_internal_key: Optional[str]):
    if INTERNAL_KEY and x_internal_key != INTERNAL_KEY:
        raise HTTPException(401, "unauthorized")

def _normalize(t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(t, p=2, dim=-1)

def _pool_to_dense(mv: torch.Tensor) -> torch.Tensor:
    return mv.mean(dim=1) if mv.dim() == 3 else mv.mean(dim=0, keepdim=True)

def _load_b64(imgs: List[str]):
    out = []
    for b in imgs:
        out.append(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"))
    return out


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL_ID + (f"@{MODEL_REV}" if MODEL_REV else ""), "device": DEVICE}


@app.post("/v1/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, x_internal_key: Optional[str] = Header(None)):
    _auth(x_internal_key)
    t0 = time.time()
    mv_outputs = []

    if isinstance(req.input, TextInput):
        texts = req.input.texts
        if not texts or len(texts) > MAX_BATCH: raise HTTPException(413, f"Batch 1..{MAX_BATCH}")
        if any(len(t) > MAX_TEXT_LEN for t in texts): raise HTTPException(413, f"text too long")
        batch = processor.process_queries(texts).to(model.device)
        with torch.no_grad(): mv = model(**batch)
        mv_outputs = list(mv) if isinstance(mv, (list, tuple)) else [mv]

    else:
        if not (req.input.image_b64 or req.input.image_urls):
            raise HTTPException(400, "provide image_b64")
        if req.input.image_urls:
            raise HTTPException(400, "image_urls disabled; send base64")
        imgs = _load_b64(req.input.image_b64 or [])
        if not imgs or len(imgs) > MAX_BATCH: raise HTTPException(413, f"Batch 1..{MAX_BATCH}")
        batch = processor.process_images(imgs).to(model.device)
        with torch.no_grad(): mv = model(**batch)
        mv_outputs = list(mv) if isinstance(mv, (list, tuple)) else [mv]

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
        model=MODEL_ID + (f"@{MODEL_REV}" if MODEL_REV else ""),
        variant=req.options.variant,
        data=res,
        usage=usage,
    )