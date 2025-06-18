# Product Search API Documentation

A Flask-based REST API for multimodal product search using vector embeddings and hybrid retrieval. The API supports text search, image search, and multimodal search (text + image) with intelligent result reordering using LLM.

## Table of Contents

- [Product Search API Documentation](#product-search-api-documentation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Base URL](#base-url)
  - [Authentication](#authentication)
  - [API Endpoints](#api-endpoints)
    - [Health Check](#health-check)
    - [API Status](#api-status)
    - [Product Search](#product-search)
    - [User Feedback](#user-feedback)
  - [Search Types](#search-types)
    - [Text Search](#text-search)
    - [Image Search](#image-search)
    - [Multimodal Search](#multimodal-search)
  - [Response Formats](#response-formats)
    - [Product Object](#product-object)
    - [Search Response](#search-response)
  - [Error Handling](#error-handling)
  - [Examples](#examples)
    - [Complete Search Workflow](#complete-search-workflow)
    - [Python Example](#python-example)
  - [Environment Variables](#environment-variables)
  - [Technical Details](#technical-details)
    - [Retrieval Components](#retrieval-components)
    - [Initialization Process](#initialization-process)

## Overview

This API provides advanced product search capabilities using:
- **MiniLM Model**: For text embeddings
- **CLIP Model**: For image embeddings and multimodal fusion
- **FAISS Indices**: For fast vector similarity search
- **PostgreSQL**: For product data storage and text search
- **LLM Integration**: For intelligent result reordering (Google/OpenAI)

## Base URL

```
http://localhost:5001
```

For production deployments, replace with your actual domain.

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

## API Endpoints

### Health Check

**GET** `/`

Simple health check endpoint to verify the API is running.

**Response:**
```json
"Backend API is running!"
```

**Example:**
```bash
curl http://localhost:5001/
```

### API Status

**GET** `/api/ready`

Check if the API is ready to accept queries. This endpoint monitors the initialization status of all required components.

**Response:**
```json
{
  "state": "ready",
  "ready": true,
  "components": {
    "minilm_model": true,
    "clip_model": true,
    "faiss_indices": true,
    "database": true
  },
  "elapsed_seconds": 45.2
}
```

**States:**
- `"starting"`: API is still initializing
- `"ready"`: API is ready to accept requests
- `"failed"`: Initialization failed

**Example:**
```bash
curl http://localhost:5001/api/ready
```

### Product Search

**POST** `/api/search`

Perform product search using text, image, or multimodal queries.

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | No* | Text search query |
| `file` | file | No* | Image file for image search |
| `image_path` | string | No* | URL or path to image for image search |
| `search_type` | string | No | Explicit search type: `"text"`, `"image"`, or `"multimodal"` |

*At least one of `query`, `file`, or `image_path` is required.

**Search Type Auto-Detection:**
- If both `query` and image provided → `multimodal`
- If only image provided → `image`
- If only `query` provided → `text`

**Response:**
```json
{
  "results": [
    {
      "Pid": "12345",
      "Name": "Product Name",
      "Description": "Product description",
      "Brand": "Brand Name",
      "Category": "Category",
      "Color": "Red",
      "Gender": "Unisex",
      "Size": "M",
      "Price": "29.99",
      "similarity": 0.85
    }
  ],
  "elapsed_time_sec": 1.234,
  "category_distribution": {
    "Shoes": 45,
    "Clothing": 35,
    "Accessories": 20
  },
  "brand_distribution": {
    "Nike": 30,
    "Adidas": 25,
    "Puma": 20,
    "Other": 25
  },
  "price_range": [15.99, 199.99],
  "average_price": 67.50,
  "session_id": "uuid-string",
  "reasoning": "LLM reasoning for result reordering"
}
```

**Example Requests:**

**Text Search:**
```bash
curl -X POST http://localhost:5001/api/search \
  -F "query=red running shoes"
```

**Image Search:**
```bash
curl -X POST http://localhost:5001/api/search \
  -F "file=@/path/to/image.jpg"
```

**Image Search with URL:**
```bash
curl -X POST http://localhost:5001/api/search \
  -F "image_path=https://example.com/image.jpg"
```

**Multimodal Search:**
```bash
curl -X POST http://localhost:5001/api/search \
  -F "query=red running shoes" \
  -F "file=@/path/to/image.jpg"
```

### User Feedback

**POST** `/api/feedback`

Submit user feedback for search results (thumbs up/down).

**Content-Type:** `application/json`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query_text` | string | No | Original text query |
| `image_path` | string | No | Original image path/URL |
| `pid` | string | Yes | Product ID to provide feedback for |
| `feedback` | boolean | Yes | `true` for thumbs up, `false` for thumbs down |
| `session_id` | string | Yes | Session ID from search response |

**Response:**
```json
{
  "success": true
}
```

**Example:**
```bash
curl -X POST http://localhost:5001/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "red running shoes",
    "pid": "12345",
    "feedback": true,
    "session_id": "uuid-from-search-response"
  }'
```

## Search Types

### Text Search
- Uses MiniLM model for text embeddings
- Combines vector search with PostgreSQL text search
- Weights: 50% fusion embedding, 50% text search
- Includes LLM-based result reordering

### Image Search
- Uses CLIP model for image embeddings
- Pure vector similarity search
- Weights: 100% image CLIP embedding
- No LLM reordering (image-only)

### Multimodal Search
- Combines CLIP image embeddings with MiniLM text embeddings
- Fusion of visual and textual features
- Weights: 50% fusion embedding, 50% text search
- Includes LLM-based result reordering

## Response Formats

### Product Object
```json
{
  "Pid": "string",
  "Name": "string",
  "Description": "string",
  "Brand": "string",
  "Category": "string",
  "Color": "string",
  "Gender": "string",
  "Size": "string",
  "Price": "string",
  "similarity": "float"
}
```

### Search Response
```json
{
  "results": "array of product objects",
  "elapsed_time_sec": "float",
  "category_distribution": "object",
  "brand_distribution": "object",
  "price_range": "[min_price, max_price]",
  "average_price": "float",
  "session_id": "string",
  "reasoning": "string"
}
```

## Error Handling

The API returns appropriate HTTP status codes:

- **200**: Success
- **400**: Bad Request (missing required parameters)
- **500**: Internal Server Error
- **503**: Service Unavailable (API not ready)

**Error Response Format:**
```json
{
  "error": "Error message description"
}
```

## Examples

### Complete Search Workflow

1. **Check API Status:**
```bash
curl http://localhost:5001/api/ready
```

2. **Perform Search:**
```bash
curl -X POST http://localhost:5001/api/search \
  -F "query=athletic shoes for running" \
  -F "file=@shoe_image.jpg"
```

3. **Submit Feedback:**
```bash
curl -X POST http://localhost:5001/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "athletic shoes for running",
    "pid": "12345",
    "feedback": true,
    "session_id": "response-session-id"
  }'
```

### Python Example

```python
import requests

# Check if API is ready
response = requests.get("http://localhost:5001/api/ready")
if response.json()["ready"]:
    # Perform search
    with open("image.jpg", "rb") as f:
        files = {"file": f}
        data = {"query": "red running shoes"}
        response = requests.post("http://localhost:5001/api/search", 
                               files=files, data=data)
    
    results = response.json()
    print(f"Found {len(results['results'])} products")
    
    # Submit feedback
    feedback_data = {
        "query_text": "red running shoes",
        "pid": results["results"][0]["Pid"],
        "feedback": True,
        "session_id": results["session_id"]
    }
    requests.post("http://localhost:5001/api/feedback", 
                 json=feedback_data)
```

## Environment Variables

The following environment variables can be configured:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `5001` |
| `GOOGLE_API_KEY` | Google API key for LLM reordering | None |
| `OPENAI_API_KEY` | OpenAI API key for LLM reordering | None |

**Note:** LLM reordering is only available when either `GOOGLE_API_KEY` or `OPENAI_API_KEY` is set. Without these keys, search results will be returned without LLM-based reordering.

## Technical Details

### Retrieval Components
The API uses three retrieval components:
1. **FaissVectorRetrieval** (fusion_embedding): For multimodal embeddings
2. **FaissVectorRetrieval** (image_clip_embedding): For image-only search
3. **TextSearchRetrieval**: For PostgreSQL text search

### Initialization Process
The API initializes the following components on startup:
- MiniLM model for text embeddings
- CLIP model for image embeddings
- FAISS indices for vector search
- Database connection for product data
- spaCy for text processing