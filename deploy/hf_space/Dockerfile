# The one site (`frontend/`) as a Hugging Face Docker Space.
#
# The weights are NOT in this image: `start.py` downloads them at boot from the private
# model repo named by the Space variable VULNLOC_WEIGHTS_REPO, using the Space secret
# HF_TOKEN. Assembled by `deploy/hf_space/stage.py`; see the docstring there.
#
# Python 3.12 and requirements.txt come from the development lock file, torch swapped for
# its CPU wheel — unlike the main Dockerfile, whose versions float (landmine #69).
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Spaces run the container as uid 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY configs ./configs
COPY frontend ./frontend
COPY start.py ./

# The app directory must belong to the user: start.py writes artifacts/ into it
# (landmine #69 — a root-owned work directory stopped the first image).
RUN mkdir -p artifacts && chown -R user:user /home/user/app
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

EXPOSE 7860
CMD ["python", "start.py"]
