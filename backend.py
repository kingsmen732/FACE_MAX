# --- Protobuf compatibility patch for MediaPipe ---
from google.protobuf import message_factory as mf
from google.protobuf import symbol_database as sdb

if not hasattr(mf.MessageFactory, "GetPrototype"):
    def _get_prototype(self, descriptor):
        return self.GetMessageClass(descriptor)
    mf.MessageFactory.GetPrototype = _get_prototype

if not hasattr(sdb.SymbolDatabase, "GetPrototype"):
    def _get_prototype_db(self, descriptor):
        return self.GetMessageClass(descriptor)
    sdb.SymbolDatabase.GetPrototype = _get_prototype_db
# ---------------------------------------------------

import os
import io
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel
from supabase import create_client, Client
import requests
from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Supabase credentials missing")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Face Maxing API")

# Allow CORS (for frontend clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HabitAnswers(BaseModel):
    sleep_hours: float
    skincare: bool
    workout_freq: str
    hydration_liters: float
    eats_processed: bool
    gender: str
    ethnicity: str


# ------------------- Analysis Functions -------------------

def measure_symmetry(landmarks):
    left = [landmarks[i] for i in range(0, 234)]
    right = [landmarks[i] for i in range(234, 468)]
    diff_sum = 0
    for l, r in zip(left, reversed(right)):
        diff_sum += abs(l.x - (1 - r.x))
    avg_diff = diff_sum / len(left)
    return max(0.0, 1 - avg_diff * 5)


def measure_jawline(landmarks):
    jaw_indices = [152, 234, 454]
    points = [(landmarks[i].x, landmarks[i].y) for i in jaw_indices]
    width = abs(points[1][0] - points[2][0])
    height = abs(points[0][1] - (points[1][1] + points[2][1]) / 2)
    ratio = height / (width + 1e-6)
    return min(1.0, ratio / 1.5)


def analyze_skin(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = max(0.0, min(1.0, 1 / (laplacian + 1e-6)))
    return 1 - blur_score


def measure_eye_symmetry(landmarks):
    left_eye = [landmarks[i] for i in [33, 133]]
    right_eye = [landmarks[i] for i in [362, 263]]
    dist_left = abs(left_eye[0].x - left_eye[1].x)
    dist_right = abs(right_eye[0].x - right_eye[1].x)
    return 1 - abs(dist_left - dist_right)


def measure_face_proportion(landmarks):
    top = landmarks[10].y
    bottom = landmarks[152].y
    left = landmarks[234].x
    right = landmarks[454].x
    height = abs(bottom - top)
    width = abs(right - left)
    ratio = height / (width + 1e-6)
    return min(1.0, 1 - abs(ratio - 1.6))


def measure_lip_symmetry(landmarks):
    left_lip = landmarks[61].x
    right_lip = landmarks[291].x
    center = (left_lip + right_lip) / 2
    diff = abs(center - 0.5)
    return max(0.0, 1 - diff * 5)


def analyze_image(image_bytes: bytes) -> float:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return 0.0
        landmarks = results.multi_face_landmarks[0].landmark
        symmetry_score = measure_symmetry(landmarks)
        jawline_score = measure_jawline(landmarks)
        skin_score = analyze_skin(img_rgb)
        eye_score = measure_eye_symmetry(landmarks)
        proportion_score = measure_face_proportion(landmarks)
        lip_score = measure_lip_symmetry(landmarks)

        current_score = (
            0.2 * symmetry_score +
            0.15 * jawline_score +
            0.15 * skin_score +
            0.15 * eye_score +
            0.15 * proportion_score +
            0.2 * lip_score
        )
        return round(current_score * 100, 2)


def calculate_habit_score(answers: HabitAnswers) -> float:
    score = 0.0
    score += min(answers.sleep_hours / 8, 1.0) * 2
    score += 2 if answers.skincare else 0
    workout_map = {'none': 0, '1-2': 1, '3-5': 2, '6+': 2.5}
    score += workout_map.get(answers.workout_freq, 0)
    score += min(answers.hydration_liters / 2, 1.0) * 2
    score += 0 if answers.eats_processed else 1
    return min(score, 5.0)


def get_groq_improvements(habits: HabitAnswers, current_score: float) -> str:
    prompt = (
        f"The user has a current facial attractiveness score of {current_score}/100.\n"
        f"Gender: {habits.gender}\n"
        f"Ethnicity: {habits.ethnicity}\n"
        f"Lifestyle habits:\n"
        f"- Sleep: {habits.sleep_hours} hours\n"
        f"- Skincare: {'Yes' if habits.skincare else 'No'}\n"
        f"- Workout Frequency: {habits.workout_freq}\n"
        f"- Water Intake: {habits.hydration_liters} liters/day\n"
        f"- Eats Processed Foods: {'Yes' if habits.eats_processed else 'No'}\n"
        f"Give actionable advice to improve their facial attractiveness based on these habits."
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a beauty and skincare expert giving advice."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300,
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error fetching improvements: {e}"


# ------------------- API Routes -------------------

@app.post("/analyze")
async def analyze_face(
    file: UploadFile,
    email: str = Form(...),
    sleep_hours: float = Form(...),
    skincare: bool = Form(...),
    workout_freq: str = Form(...),
    hydration: float = Form(...),
    eats_processed: bool = Form(...),
    gender: str = Form(...),
    ethnicity: str = Form(...),
):
    try:
        contents = await file.read()
        current_score = analyze_image(contents)

        answers = HabitAnswers(
            sleep_hours=sleep_hours,
            skincare=skincare,
            workout_freq=workout_freq,
            hydration_liters=hydration,
            eats_processed=eats_processed,
            gender=gender,
            ethnicity=ethnicity
        )

        habit_score = calculate_habit_score(answers)
        potential_score = min(current_score + habit_score, 100)

        # Save to Supabase storage
        file_path = f"{email}/{file.filename}"
        res = supabase.storage.from_("images").upload(file_path, contents, {"upsert": True})
        if getattr(res, 'error', None):
            raise HTTPException(status_code=500, detail=f"Image upload failed: {res.error}")

        image_url = f"{SUPABASE_URL}/storage/v1/object/public/images/{file_path}"

        # Save to Supabase DB
        supabase.table("face_scores").insert({
            "user_id": email,
            "image_url": image_url,
            "current_score": current_score,
            "potential_score": potential_score
        }).execute()

        supabase.table("habit_answers").insert({"user_id": email, **answers.dict()}).execute()

        improvements = get_groq_improvements(answers, current_score)

        return {
            "status": "success",
            "current_score": current_score,
            "potential_score": potential_score,
            "improvements": improvements,
            "image_url": image_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
