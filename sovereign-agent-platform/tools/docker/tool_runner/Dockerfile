FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir jsonschema requests
COPY . /app/tools
ENV PYTHONPATH=/app
ENTRYPOINT ["python", "/app/tools/docker/tool_runner/runner.py"]
