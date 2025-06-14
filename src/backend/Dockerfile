FROM python:3.11-slim

WORKDIR /app

# Accept build argument for OpenAI API key
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Install system dependencies required by some Python packages (e.g., psycopg2)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY src/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy config folder and backend source code
COPY config /app/config
COPY src/backend /app/src/backend

# Add the app directory to PYTHONPATH
ENV PYTHONPATH=/app

# Expose backend port
EXPOSE 5001

# Run your backend app
CMD ["python", "/app/src/backend/api.py"]
