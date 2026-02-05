# AI-Generated Voice Detection API (Multi-Language)

API that determines whether a voice sample is **AI-generated** or **human-generated**. Supports Tamil, English, Hindi, Malayalam, and Telugu. Accepts Base64-encoded MP3 and returns classification, confidence score, and explanation.

## Prerequisites

- Python 3.10+

## Setup & run

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```
2. Set your Gemini API key (required):
   - `GEMINI_API_KEY` or `GOOGLE_API_KEY`
3. Optional: API keys for the endpoint (comma-separated; default includes `sk_test_123456789`):
   - `VOICE_DETECTION_API_KEYS=sk_test_123456789,your_key`
4. Run the API:
   ```bash
   python main.py
   ```
   Or: `uvicorn main:app --host 0.0.0.0 --port 8000`

## API

- **Endpoint:** `POST /api/voice-detection`
- **Headers:** `x-api-key: <your-api-key>`
- **Body (JSON):** `language`, `audioFormat` (`"mp3"`), `audioBase64`
- **Response:** `status`, `language`, `classification` (AI_GENERATED | HUMAN), `confidenceScore` (0–1), `explanation`

**Health:** `GET /health`
