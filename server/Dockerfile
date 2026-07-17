FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo
COPY nfl-model/requirements.txt server/requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-server.txt

COPY . /repo
WORKDIR /repo/server
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
