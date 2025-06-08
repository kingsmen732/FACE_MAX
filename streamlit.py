import os
import io
import uuid
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from supabase import create_client, Client
import requests

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")  # Set your Groq API URL here
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL= os.getenv("GROQ_MODEL")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Supabase credentials missing in environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY is not set."

    prompt = (
        f"The user's current facial attractiveness score is {current_score}/10.\n"
        f"Here are their lifestyle habits:\n"
        f"- Sleep Hours: {habits.sleep_hours}\n"
        f"- Skincare Routine: {'Yes' if habits.skincare else 'No'}\n"
        f"- Workout Frequency: {habits.workout_freq}\n"
        f"- Daily Water Intake: {habits.hydration_liters} liters\n"
        f"- Eats Processed Foods: {'Yes' if habits.eats_processed else 'No'}\n\n"
        f"Give a direct and simple list of personalized suggestions the user can follow to improve their facial attractiveness score."
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a beauty and wellness expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300,
    }

    try:
        response = requests.post(GROQ_API_URL, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error fetching improvements: {e}"

# Streamlit UI
st.title("Face Maxing App")
user_id = st.text_input("User ID")
uploaded_file = st.file_uploader("Upload your face image", type=["jpg", "png", "jpeg"])

sleep_hours = st.slider("Hours of sleep", 0.0, 12.0, 7.0)
skincare = st.checkbox("Do you follow a skincare routine?")
workout_freq = st.selectbox("Workout frequency", ['none', '1-2', '3-5', '6+'])
hydration = st.slider("Liters of water per day", 0.0, 5.0, 2.0)
eats_processed = st.checkbox("Do you frequently eat processed foods?")

if st.button("Analyze Face and Habits"):
    if not uploaded_file or not user_id:
        st.error("Please upload your image and enter a User ID")
    else:
        contents = uploaded_file.read()
        current_score = analyze_image(contents)
        answers = HabitAnswers(
            sleep_hours=sleep_hours,
            skincare=skincare,
            workout_freq=workout_freq,
            hydration_liters=hydration,
            eats_processed=eats_processed
        )
        habit_score = calculate_habit_score(answers)
        potential_score = min(current_score + habit_score, 10)

        file_path = f"{user_id}/{uploaded_file.name}"
        supabase.storage.from_("images").upload(file_path, contents)
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/images/{file_path}"

        supabase.table("face_scores").insert({
            "user_id": user_id,
            "image_url": image_url,
            "current_score": current_score,
            "potential_score": potential_score,
        }).execute()

        supabase.table("habit_answers").insert({"user_id": user_id, **answers.dict()}).execute()

        improvements = get_groq_improvements(answers, current_score)

        st.metric("Current Score", current_score)
        st.metric("Potential Score", potential_score)
        st.markdown("### Personalized Recommendations")
        st.markdown(improvements)
