FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Base deps are lean. Speaker diarization (torch/resemblyzer/librosa) is opt-in via a build arg so
# the default image stays small: `docker compose` sets DIARIZATION=1 (see docker-compose.yml).
ARG DIARIZATION=0
COPY requirements.txt requirements-diarization.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$DIARIZATION" = "1" ]; then \
         pip install --no-cache-dir -r requirements-diarization.txt \
         && pip install --no-cache-dir --no-deps resemblyzer==0.1.4; \
       fi

COPY . ./

USER 10001:10001

EXPOSE 5000 8765

CMD ["python", "main.py"]