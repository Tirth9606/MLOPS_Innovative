# Phase 3: containerize the FastAPI model-serving app.
FROM python:3.10-slim

WORKDIR /app

# Install serving dependencies first (better layer caching)
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the application code and the trained model
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# FastAPI will listen on port 8000
EXPOSE 8000

# Start the API with uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
