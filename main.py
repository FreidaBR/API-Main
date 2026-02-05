"""
Hybrid AI Voice Detection API
Signal Forensics + Gemini Reasoning
Supports: Tamil, English, Hindi, Malayalam, Telugu
"""

import base64
import os
import io
import json
from enum import Enum
from typing import Literal

import numpy as np
import librosa
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field

# Gemini
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
except:
    GEMINI_AVAILABLE = False


# ==============================
# Supported Languages
# ==============================
class SupportedLanguage(str, Enum):
    Tamil = "Tamil"
    English = "English"
    Hindi = "Hindi"
    Malayalam = "Malayalam"
    Telugu = "Telugu"


# ==============================
# Request / Response Models
# ==============================
class VoiceDetectionRequest(BaseModel):
    language: SupportedLanguage
    audioFormat: str = Field(..., description="Must be mp3")
    audioBase64: str


class VoiceDetectionSuccessResponse(BaseModel):
    status: Literal["success"] = "success"
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float
    explanation: str


class VoiceDetectionErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str


# ==============================
# Signal Forensics Engine
# ==============================
def extract_signal_features(audio_bytes):

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    features = {}

    try:
        pitch = librosa.yin(y, 50, 400)
        features["pitch_var"] = float(np.var(pitch))
    except:
        features["pitch_var"] = 0.0

    rms = librosa.feature.rms(y=y)
    features["energy_var"] = float(np.var(rms))

    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))

    flatness = librosa.feature.spectral_flatness(y=y)
    features["flatness"] = float(np.mean(flatness))

    silence_ratio = np.mean(np.abs(y) < 0.002)
    features["silence_ratio"] = float(silence_ratio)

    return features


def signal_ai_probability(features):

    score = 0
    total = 5

    if features["pitch_var"] < 0.002:
        score += 1
    if features["energy_var"] < 0.001:
        score += 1
    if features["zcr_mean"] < 0.03:
        score += 1
    if features["flatness"] > 0.4:
        score += 1
    if features["silence_ratio"] > 0.2:
        score += 1

    return score / total


# ==============================
# Gemini Forensic Analysis
# ==============================
SYSTEM_INSTRUCTION = """
You are an expert forensic audio analyst.
Detect whether voice is AI generated or human.
If unsure, reduce confidence instead of guessing.
"""

def analyze_audio_with_gemini(audio_base64_seen, language):

    if not GEMINI_AVAILABLE:
        return {
            "classification": "HUMAN",
            "confidenceScore": 0.5,
            "explanation": "Gemini not configured. Signal analysis only."
        }

    clean_b64 = audio_base64_seen.split(",", 1)[-1].strip()
    audio_bytes = base64.b64decode(clean_b64)

    client = genai.Client()

    prompt = f"""
    Perform forensic detection of AI vs Human voice.
    Language: {language}
    Check spectral artifacts, breathing, prosody, noise floor, consonant realism.
    """

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        temperature=0.2
    )

    contents = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
    ]

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=contents,
        config=config
    )

    try:
        return json.loads(response.text.strip())
    except:
        return {
            "classification": "HUMAN",
            "confidenceScore": 0.5,
            "explanation": "Gemini response parsing failed."
        }


# ==============================
# API KEY
# ==============================
VALID_API_KEYS = {"sk_test_123456789"}


# ==============================
# FastAPI App
# ==============================
app = FastAPI(title="Hybrid AI Voice Detection API")


@app.get("/")
def root():
    return {"message": "Voice Detection API Running"}


@app.get("/health")
def health():
    return {"status": "ok", "gemini_enabled": GEMINI_AVAILABLE}


# ==============================
# MAIN DETECTION ENDPOINT
# ==============================
@app.post("/api/voice-detection",
          response_model=VoiceDetectionSuccessResponse | VoiceDetectionErrorResponse)
def detect_voice(body: VoiceDetectionRequest,
                 x_api_key: str | None = Header(None, alias="x-api-key")):

    if not x_api_key or x_api_key not in VALID_API_KEYS:
        return VoiceDetectionErrorResponse(
            message="Invalid API key"
        )

    if body.audioFormat.lower() != "mp3":
        return VoiceDetectionErrorResponse(
            message="Only MP3 supported"
        )

    try:
        clean_b64 = body.audioBase64.split(",", 1)[-1].strip()
        audio_bytes = base64.b64decode(clean_b64)

        # Signal Analysis
        features = extract_signal_features(audio_bytes)
        signal_prob = signal_ai_probability(features)

        # Gemini Analysis
        gemini_result = analyze_audio_with_gemini(body.audioBase64, body.language.value)

        gemini_prob = 0.8 if gemini_result.get("classification") == "AI_GENERATED" else 0.2

        # Final Ensemble Decision
        final_prob = (0.6 * signal_prob) + (0.4 * gemini_prob)

        classification = "AI_GENERATED" if final_prob > 0.5 else "HUMAN"

        confidence = abs(final_prob - 0.5) * 2
        confidence = float(min(max(confidence, 0.1), 0.99))

        explanation = f"""
Signal Analysis:
Pitch Var: {features['pitch_var']:.6f}
Energy Var: {features['energy_var']:.6f}
Flatness: {features['flatness']:.4f}
Silence Ratio: {features['silence_ratio']:.4f}

LLM Analysis:
{gemini_result.get('explanation', 'No explanation')}
"""

        return VoiceDetectionSuccessResponse(
            language=body.language.value,
            classification=classification,
            confidenceScore=round(confidence, 4),
            explanation=explanation.strip()
        )

    except Exception as e:
        return VoiceDetectionErrorResponse(
            message="Internal processing error"
        )


# ==============================
# Local Run
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
