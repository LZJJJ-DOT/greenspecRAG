# CUDA 12.1 is compatible with the host's NVIDIA 536.67 driver. This base is
# intentionally used instead of the much larger PyTorch Docker Hub image.
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/.cache/huggingface \
    TRANSFORMERS_CACHE=/models/.cache/huggingface/transformers

WORKDIR /app

RUN apt-get update && \
    apt-get install --yes --no-install-recommends python3 python3-pip python-is-python3 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.runtime.txt /tmp/requirements.runtime.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
RUN \
    python -m pip install -r /tmp/requirements.runtime.txt

COPY pyproject.toml README.md ./
COPY greenspec_rag ./greenspec_rag
COPY deploy ./deploy
RUN python -m pip install --no-deps . && \
    python -c "import sqlite3; assert 'ENABLE_FTS5' in {row[0] for row in sqlite3.connect(':memory:').execute('pragma compile_options')}; import jieba; assert jieba.__version__ == '0.42.1'"

CMD ["uvicorn", "greenspec_rag.api:app", "--host", "0.0.0.0", "--port", "8787"]
