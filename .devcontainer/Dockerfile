FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y git && \
    pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
