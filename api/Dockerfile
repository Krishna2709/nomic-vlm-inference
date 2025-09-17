FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
COPY app.py /app/app.py

# Bake model snapshot at build time (faster cold starts)
ARG MODEL_ID="nomic-ai/colnomic-embed-multimodal-3b"
ARG MODEL_REV=""
RUN --mount=type=secret,id=hf \
    python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
mid=os.environ.get("MODEL_ID"); rev=os.environ.get("MODEL_REV") or None
tok=os.environ.get("HF_TOKEN")  # set via BuildKit secret if needed
snapshot_download(repo_id=mid, revision=rev, local_dir="/models", use_auth_token=tok)
print("Baked model to /models")
PY

ENV MODEL_ID=${MODEL_ID} MODEL_REV=${MODEL_REV}
EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","1"]