FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

# --- Bake the HF snapshot to avoid cold-start downloads on RunPod ---
ARG MODEL_ID="nomic-ai/colnomic-embed-multimodal-3b"
ARG MODEL_REV=""
# If the model is gated, pass HF_TOKEN at build time (see command below)
RUN --mount=type=secret,id=hf \
    python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
mid=os.environ.get("MODEL_ID")
rev=os.environ.get("MODEL_REV") or None
tok=os.environ.get("HF_TOKEN")
snapshot_download(repo_id=mid, revision=rev, local_dir="/models", use_auth_token=tok)
print("Baked model to /models")
PY

ENV MODEL_ID=${MODEL_ID} MODEL_REV=${MODEL_REV} MODEL_DIR=/models

EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","1"]