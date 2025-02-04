# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app directory
COPY app/ /app/

# Create model directory
RUN mkdir -p /app/model

# Copy model files
COPY ./model /app/model

# Set the Python path and model path
ENV PYTHONPATH=/app

# Run flask api
ENTRYPOINT ["python3"]
CMD ["app.py"]