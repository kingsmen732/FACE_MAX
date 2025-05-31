import os
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import numpy as np
import cv2
import mediapipe as mp
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()  # Load .env file

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Supabase credentials missing in environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://v0-face-maxing-app.vercel.app"],  # Change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HabitAnswers(BaseModel):
    sleep_hours: float
    skincare: bool
    workout_freq: str  # 'none', '1-2', '3-5', '6+'
    hydration_liters: float
    eats_processed: bool

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

        current_score = 0.4 * symmetry_score + 0.3 * jawline_score + 0.3 * skin_score
        return round(current_score * 10, 2)

def measure_symmetry(landmarks):
    left = [landmarks[i] for i in range(0, 234)]
    right = [landmarks[i] for i in range(234, 468)]
    diff_sum = 0
    for l, r in zip(left, reversed(right)):
        diff_sum += abs(l.x - (1 - r.x))
    avg_diff = diff_sum / len(left)
    return max(0.0, 1 - avg_diff * 5)  # Normalize to [0,1]

def measure_jawline(landmarks):
    jaw_indices = [152, 234, 454]
    points = [(landmarks[i].x, landmarks[i].y) for i in jaw_indices]
    width = abs(points[1][0] - points[2][0])
    height = abs(points[0][1] - (points[1][1] + points[2][1]) / 2)
    ratio = height / (width + 1e-6)
    return min(1.0, ratio / 1.5)  # Normalize with ideal ratio ~1.5

def analyze_skin(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = max(0.0, min(1.0, 1 / (laplacian + 1e-6)))
    return 1 - blur_score

def calculate_habit_score(answers: HabitAnswers) -> float:
    score = 0.0
    score += min(answers.sleep_hours / 8, 1.0) * 2
    score += 2 if answers.skincare else 0
    workout_map = {'none': 0, '1-2': 1, '3-5': 2, '6+': 2.5}
    score += workout_map.get(answers.workout_freq, 0)
    score += min(answers.hydration_liters / 2, 1.0) * 2
    score += 0 if answers.eats_processed else 1
    return min(score, 5.0)

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    contents = await file.read()
    current_score = analyze_image(contents)

    # Upload image to Supabase Storage
    file_path = f"{user_id}/{file.filename}"
    res = supabase.storage.from_("images").upload(file_path, contents)
    if res.status_code != 200 and res.status_code != 201:
        raise HTTPException(status_code=500, detail="Failed to upload image")

    image_url = f"{SUPABASE_URL}/storage/v1/object/public/images/{file_path}"

    # Insert face_scores record
    supabase.table("face_scores").insert({
        "user_id": user_id,
        "image_url": image_url,
        "current_score": current_score,
        "potential_score": None,
    }).execute()

    return {"current_score": current_score, "image_url": image_url}

@app.post("/submit-answers")
def submit_answers(
    answers: HabitAnswers,
    user_id: str = Form(...)
):
    habit_score = calculate_habit_score(answers)

    # Fetch latest current_score
    data = supabase.table("face_scores")\
        .select("id,current_score")\
        .eq("user_id", user_id)\
        .order("submitted_at", desc=True)\
        .limit(1).execute()
    if not data.data:
        raise HTTPException(status_code=404, detail="No face score found for user")
    face_score_id = data.data[0]["id"]
    current_score = data.data[0]["current_score"]

    potential_score = min(current_score + habit_score, 10)

    # Insert habit answers
    supabase.table("habit_answers").insert({
        "user_id": user_id,
        **answers.dict()
    }).execute()

    # Update potential_score in face_scores
    supabase.table("face_scores").update({
        "potential_score": potential_score
    }).eq("id", face_score_id).execute()

    return {"potential_score": potential_score}

@app.get("/get-results")
def get_results(user_id: str = Query(...)):
    data = supabase.table("face_scores")\
        .select("current_score", "potential_score")\
        .eq("user_id", user_id)\
        .order("submitted_at", desc=True)\
        .limit(1).execute()
    if not data.data:
        return {"current_score": 0, "potential_score": 0}
    return data.data[0]
