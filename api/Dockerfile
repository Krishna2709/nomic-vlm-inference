FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# OS deps (git needed for colpali install from GitHub)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps (torch/CUDA already in base) ----
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# ---- Bake the HF snapshot ONCE into the image ----
# If the model is public, you can remove the --mount line and "tok" usage.
ARG MODEL_ID="nomic-ai/colnomic-embed-multimodal-3b"
ARG MODEL_REV=""
RUN --mount=type=secret,id=hf \
    python3 - <<'PY'
import os
import json
from huggingface_hub import snapshot_download
mid = os.environ.get("MODEL_ID")
rev = os.environ.get("MODEL_REV") or None
tok = os.environ.get("HF_TOKEN")  # provided via BuildKit secret if needed
print(f"Downloading model {mid} (rev: {rev}) to /models/3b")

# Download the adapter model first
snapshot_download(
    repo_id=mid, 
    revision=rev, 
    local_dir="/models/3b", 
    use_auth_token=tok,
    local_files_only=False  # Allow download during build
)

# Read the adapter config to get the base model
with open("/models/3b/adapter_config.json", "r") as f:
    adapter_config = json.load(f)
base_model = adapter_config.get("base_model_name_or_path")
print(f"Base model: {base_model}")

# Download the base model
if base_model:
    print(f"Downloading base model {base_model} to /models/base")
    snapshot_download(
        repo_id=base_model,
        local_dir="/models/base",
        use_auth_token=tok,
        local_files_only=False
    )

print("Successfully baked models")
print("Files in /models/3b:")
for f in sorted(os.listdir("/models/3b")):
    print(f"  {f}")
print("Files in /models/base:")
for f in sorted(os.listdir("/models/base")):
    print(f"  {f}")
PY

# Let the app know to load locally and avoid network at runtime
ENV MODEL_ID=${MODEL_ID} \
    MODEL_REV=${MODEL_REV} \
    MODEL_DIR=/models/3b \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/models/3b \
    HF_DATASETS_OFFLINE=1

# ---- App code ----
COPY app.py /app/app.py

EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","1"]