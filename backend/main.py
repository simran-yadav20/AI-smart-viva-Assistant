from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from ai_engine import generate_questions, evaluate_answer, improve_answer, get_correct_answer
import uuid
import os
import io
import re
import time
import glob

# Try to import pyttsx3, fallback to gTTS
try:
    import pyttsx3
    USE_PYTTSX3 = True
except ImportError:
    from gtts import gTTS
    USE_PYTTSX3 = False
    print("⚠️ pyttsx3 not available, using gTTS instead")

app = FastAPI(title="AI Viva Assistant API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static folder
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Audio files older than this many seconds will be deleted
AUDIO_MAX_AGE_SECONDS = 300  # 5 minutes


def cleanup_old_audio():
    """Delete WAV files in static/ that are older than AUDIO_MAX_AGE_SECONDS."""
    now = time.time()
    for filepath in glob.glob(os.path.join(STATIC_DIR, "*.wav")):
        try:
            if now - os.path.getmtime(filepath) > AUDIO_MAX_AGE_SECONDS:
                os.remove(filepath)
        except Exception:
            pass


def generate_tts(text: str, filepath: str) -> bool:
    """Generate TTS audio to a file. Returns True on success."""
    try:
        if USE_PYTTSX3:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.save_to_file(text, filepath)
            engine.runAndWait()
            engine.stop()
        else:
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(filepath)
        return True
    except Exception as e:
        print(f"Audio generation error: {e}")
        return False


# ==========================
# Models
# ==========================

class AnswerRequest(BaseModel):
    question: str
    answer: str


class SkipRequest(BaseModel):
    question: str


class SpeechRequest(BaseModel):
    question: str


# ==========================
# Routes
# ==========================

@app.get("/")
async def root():
    return {"message": "AI Viva Assistant API Running 🚀"}


@app.get("/generate/{topic}")
def generate(topic: str):
    try:
        questions = generate_questions(topic)
        return {"questions": questions}
    except RuntimeError as e:
        return {"error": str(e)}


@app.post("/evaluate/")
def evaluate(data: AnswerRequest):
    cleanup_old_audio()

    evaluation = evaluate_answer(data.question, data.answer)

    # Extract score from "Score: X/10" or "X/10" patterns
    score = 5  # default fallback
    match = re.search(r"(\d+)\s*/\s*10", evaluation)
    if match:
        score = min(int(match.group(1)), 10)

    improved = improve_answer(data.question, data.answer)

    filename = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(STATIC_DIR, filename)
    audio_url = None

    if generate_tts(improved, filepath):
        audio_url = f"http://127.0.0.1:8000/static/{filename}"

    return {
        "evaluation": evaluation,
        "score": score,
        "improved_answer": improved,
        "improved_answer_audio_url": audio_url,
    }


@app.post("/speak/")
def speak(data: SpeechRequest):
    try:
        if USE_PYTTSX3:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            temp_file = os.path.join(STATIC_DIR, f"temp_{uuid.uuid4().hex}.wav")
            engine.save_to_file(data.question, temp_file)
            engine.runAndWait()
            engine.stop()
            with open(temp_file, "rb") as f:
                audio_data = f.read()
            os.remove(temp_file)
        else:
            tts = gTTS(text=data.question, lang="en", slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_data = audio_buffer.getvalue()

        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/mpeg")

    except Exception as e:
        return {"error": str(e)}


@app.post("/skip/")
def skip(data: SkipRequest):
    cleanup_old_audio()

    correct = get_correct_answer(data.question)

    filename = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(STATIC_DIR, filename)
    audio_url = None

    if generate_tts(correct, filepath):
        audio_url = f"http://127.0.0.1:8000/static/{filename}"

    return {
        "correct_answer": correct,
        "audio_url": audio_url,
    }