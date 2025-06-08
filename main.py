import os
import io
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from supabase import create_client, Client
import requests
import plotly.graph_objects as go

# Load env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Supabase credentials missing")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class HabitAnswers(BaseModel):
    sleep_hours: float
    skincare: bool
    workout_freq: str
    hydration_liters: float
    eats_processed: bool

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

def analyze_image(image_bytes: bytes) -> tuple:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return 0.0, 0.0, 0.0, 0.0
        landmarks = results.multi_face_landmarks[0].landmark
        symmetry_score = measure_symmetry(landmarks)
        jawline_score = measure_jawline(landmarks)
        skin_score = analyze_skin(img_rgb)
        current_score = 0.4 * symmetry_score + 0.3 * jawline_score + 0.3 * skin_score
        return round(current_score * 100, 2), symmetry_score, jawline_score, skin_score

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
            {"role": "system", "content": "You are a beauty and skincare expert giving advice to users based on lifestyle habits. give only actionable advice as a doctor only in text format"},
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

def show_face_radar_chart(symmetry, jawline, skin, current_score):
    labels = [
        "Facial Symmetry",
        "Jawline Definition",
        "Skin Quality",
        "Overall Score"
    ]
    values = [
        round(symmetry * 100, 2),
        round(jawline * 100, 2),
        round(skin * 100, 2),
        round(current_score, 2)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Face Attributes',
        line_color='deepskyblue'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont_size=10
            ),
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.markdown("### Face Attribute Breakdown")
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.title("Face Maxing App: Upload Face + Habits")

user_id = st.text_input("User ID")
uploaded_file = st.file_uploader("Upload your face image", type=["jpg", "png", "jpeg"])

st.subheader("Lifestyle Questions")
sleep_hours = st.slider("Hours of sleep", 0.0, 12.0, 7.0)
skincare = st.checkbox("Do you follow a skincare routine?")
workout_freq = st.selectbox("Workout frequency", ['none', '1-2', '3-5', '6+'])
hydration = st.slider("Liters of water per day", 0.0, 5.0, 2.0)
eats_processed = st.checkbox("Do you frequently eat processed foods?")

if st.button("Submit All") and user_id and uploaded_file:
    contents = uploaded_file.read()
    current_score, symmetry, jawline, skin = analyze_image(contents)
    answers = HabitAnswers(
        sleep_hours=sleep_hours,
        skincare=skincare,
        workout_freq=workout_freq,
        hydration_liters=hydration,
        eats_processed=eats_processed
    )
    habit_score = calculate_habit_score(answers)
    potential_score = min(current_score + habit_score, 100)

    file_path = f"{user_id}/{uploaded_file.name}"
    try:
        res = supabase.storage.from_("images").upload(file_path, contents)
        res_error = getattr(res, 'error', None)
        if res_error:
            st.error(f"Failed to upload image: {res_error}")
            st.stop()
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/images/{file_path}"

        supabase.table("face_scores").insert({
            "user_id": user_id,
            "image_url": image_url,
            "current_score": current_score,
            "potential_score": potential_score
        }).execute()

        supabase.table("habit_answers").insert({"user_id": user_id, **answers.dict()}).execute()

        improvements = get_groq_improvements(answers, current_score)

        st.success("Submission successful!")
        show_face_radar_chart(symmetry, jawline, skin, current_score)
        st.metric("Current Score", current_score)
        st.metric("Potential Score", potential_score)
        st.markdown("### Suggested Improvements from AI:")
        st.write(improvements)

    except Exception as e:
        st.error(f"Error during submission: {e}")

elif st.button("Get Last Results") and user_id:
    data = supabase.table("face_scores").select("current_score", "potential_score").eq("user_id", user_id).order("submitted_at", desc=True).limit(1).execute()
    if not data.data:
        st.write("No data found")
    else:
        st.metric("Current Score", data.data[0]["current_score"])
        st.metric("Potential Score", data.data[0]["potential_score"])
