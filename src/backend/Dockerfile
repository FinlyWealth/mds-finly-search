FROM python:3.11-slim

WORKDIR /app

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

# Copy config folder from root level
COPY config /app/config
# Copy backend source code
COPY src/backend /app/src/backend
WORKDIR /app/src/backend

# Expose backend port (assuming Flask default 5000)
EXPOSE 5001

# Run your backend app
CMD ["python", "api.py"]
