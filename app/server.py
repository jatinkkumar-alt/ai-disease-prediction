from flask import Flask, request, jsonify, send_from_directory, send_file, session, redirect


import joblib
import numpy as np
import os
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import requests
import csv
import json
import re


app = Flask(__name__)
app.secret_key = "super-simple-secret-key-change-later"


# --- Load trained model and symptom columns ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "..", "web")

model_path = os.path.join(BASE_DIR, "..", "models", "disease_model.pkl")
columns_path = os.path.join(BASE_DIR, "..", "models", "symptom_columns.pkl")


model = joblib.load(model_path)
symptom_columns = joblib.load(columns_path)

print(f"✅ Model loaded. Total symptoms: {len(symptom_columns)}")

# --- Logging config: where to store predictions ---

LOG_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(LOG_DIR, exist_ok=True)
PREDICTIONS_CSV = os.path.join(LOG_DIR, "predictions.csv")


def log_prediction(symptoms, result, user_email="guest"):
    """
    Append a single prediction record to data/predictions.csv.

    Columns:
    timestamp, user_email, disease, confidence, symptoms (comma-separated)
    """
    try:
        file_exists = os.path.exists(PREDICTIONS_CSV)

        with open(PREDICTIONS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header only once
            if not file_exists:
                writer.writerow(["timestamp", "user_email", "disease", "confidence", "symptoms"])

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            disease = result.get("disease", "")
            confidence = result.get("confidence", "")
            # symptoms is a list of internal names (fever, chills, etc.)
            symptoms_str = ",".join(symptoms)

            writer.writerow([timestamp, user_email, disease, confidence, symptoms_str])
    except Exception as e:
        # Don't crash the app if logging fails
        print("Error logging prediction:", e)

def safe_text(text: str) -> str:
    """
    Convert text to something safe for FPDF core fonts (latin-1 only).
    Strips or ignores unsupported Unicode characters (like fancy dashes, quotes).
    """
    if text is None:
        return ""
    try:
        return str(text).encode("latin-1", "ignore").decode("latin-1")
    except Exception:
        return str(text)


# --- User storage (for signup/login) ---

USERS_JSON = os.path.join(LOG_DIR, "users.json")


def save_users(users):
    with open(USERS_JSON, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def load_users():
    """
    Load users from users.json.
    If file does not exist, create a default admin user.
    """
    if not os.path.exists(USERS_JSON):
        # Default admin account (for future admin panel)
        default_admin = {
            "id": 1,
            "name": "Admin",
            "email": "admin@medai.com",
            "password": "admin123",  # plain text for project simplicity
            "role": "admin"
        }
        save_users([default_admin])
        return [default_admin]

    try:
        with open(USERS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # If file is corrupted, reset with just admin
        default_admin = {
            "id": 1,
            "name": "Admin",
            "email": "admin@medai.com",
            "password": "admin123",
            "role": "admin"
        }
        save_users([default_admin])
        return [default_admin]


def find_user_by_email(email):
    users = load_users()
    for u in users:
        if u.get("email", "").lower() == email.lower():
            return u
    return None



# --- Gemini / LLM config (REPLACE with your actual Render URL) ---
GEMINI_API_URL = "https://intelli-ai-server-29nh.onrender.com/intelli-ai"  # <-- put your real URL here

# If your Render app expects an API key in header, you can optionally use:
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Helper: make prediction + info from selected symptoms ---

def make_prediction(selected_symptoms):
    selected_set = set(selected_symptoms)

    # Build input vector
    input_vector = []
    for col in symptom_columns:
        input_vector.append(1 if col in selected_set else 0)
    input_vector = np.array(input_vector).reshape(1, -1)

    # Main prediction
    predicted_disease = model.predict(input_vector)[0]

    # Top-3 predictions
    top_predictions = []
    confidence = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_vector)[0]
        classes = model.classes_
        sorted_indices = np.argsort(proba)[::-1]

        for idx in sorted_indices[:3]:
            top_predictions.append({
                "disease": str(classes[idx]),
                "confidence": float(proba[idx])
            })

        confidence = float(proba[sorted_indices[0]])
    else:
        top_predictions.append({
            "disease": predicted_disease,
            "confidence": None
        })

    # info = disease_info.get(predicted_disease, default_info)

    return {
        "disease": predicted_disease,
        "confidence": confidence,
        "top_predictions": top_predictions,
        # "info": info
    }


# --- Serve frontend ---

@app.route("/")
def home():
    # If user is not logged in, send them to login page
    if "user_email" not in session:
        return redirect("/login")
    # Otherwise show main AI app
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/login")
def login_page():
    return send_from_directory(WEB_DIR, "login.html")


@app.route("/signup")
def signup_page():
    return send_from_directory(WEB_DIR, "signup.html")

@app.route("/history")
def history_page():
    if "user_email" not in session:
        return redirect("/login")
    return send_from_directory(WEB_DIR, "history.html")

@app.route("/admin")
def admin_page():
    if "user_email" not in session:
        return redirect("/login")

    if session.get("user_role") != "admin":
        return redirect("/")

    return send_from_directory(WEB_DIR, "admin.html")

# --- API: list symptoms ---

@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": symptom_columns})


# --- API: prediction (used by UI) ---

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    selected_symptoms = data.get("symptoms", [])

    result = make_prediction(selected_symptoms)

    # If user is logged in, log with their email; otherwise "guest"
    user_email = session.get("user_email", "guest")

    try:
        log_prediction(selected_symptoms, result, user_email=user_email)
    except Exception as e:
        print("Error logging prediction:", e)

    return jsonify(result)



# --- API: generate PDF report ---

@app.route("/generate-report", methods=["POST"])
def generate_report():
    """
    Request JSON:
    {
        "symptoms": [...],
        "age": "22",
        "gender": "Male",
        "duration": "5"
    }
    Returns: PDF file as download.
    """
    data = request.get_json() or {}
    selected_symptoms = data.get("symptoms", [])
    age = data.get("age") or ""
    gender = data.get("gender") or ""
    duration = data.get("duration") or ""

    # Reuse main prediction logic
    result = make_prediction(selected_symptoms)

    disease = result.get("disease", "Unknown")
    confidence = result.get("confidence", None)
    top_predictions = result.get("top_predictions", [])

    # --- Ask Gemini for ABOUT + PRECAUTIONS (same style as UI) ---
    about_text = ""
    precautions = []

    # Build prompt similar to /ask-ai, but fixed for report
    try:
        lines = []
        lines.append("You are an AI medical information assistant.")
        lines.append("You must NOT give a confirmed diagnosis or prescribe medicine.")
        lines.append("Always encourage the user to consult a doctor for actual medical decisions.")
        lines.append("")
        lines.append("Here is an AI model based disease risk assessment:")
        lines.append(f"- Predicted disease: {disease}")
        if confidence is not None:
            lines.append(f"- Model confidence: {confidence*100:.2f}%")
        if top_predictions:
            lines.append("- Top possible conditions:")
            for p in top_predictions[:3]:
                c = p.get("confidence")
                if c is not None:
                    conf_str = f"{c*100:.2f}%"
                else:
                    conf_str = "N/A"
                lines.append(f"  - {p.get('disease','Unknown')} ({conf_str})")
        if selected_symptoms:
            lines.append(f"- Reported symptoms: {', '.join(selected_symptoms)}")
        lines.append("")
        lines.append(
            "Explain this result in 1 short paragraph and list 4-6 general precautions as bullet points. "
            "Respond in exactly this format:\n"
            "ABOUT:\n"
            "<one short paragraph>\n"
            "PRECAUTIONS:\n"
            "- point 1\n"
            "- point 2\n"
            "- point 3\n"
            "- point 4\n"
        )

        prompt = "\n".join(lines)

        # Call same Gemini Render API as /ask-ai
        payload = {"message": prompt}
        resp = requests.post(GEMINI_API_URL, json=payload, timeout=20)
        resp.raise_for_status()
        resp_json = resp.json()
        raw = resp_json.get("reply", str(resp_json))

        # Parse ABOUT / PRECAUTIONS
        parts = re.split(r"PRECAUTIONS:", raw, flags=re.IGNORECASE)
        if len(parts) > 1:
            about_raw = re.sub(r"ABOUT:", "", parts[0], flags=re.IGNORECASE).strip()
            precautions_block = parts[1].strip()

            about_text = about_raw

            precautions = []
            for line in precautions_block.splitlines():
                # remove leading bullets / numbers
                cleaned = re.sub(r"^[-•\d\.\s]+", "", line).strip()
                if cleaned:
                    precautions.append(cleaned)
        else:
            about_text = raw.strip()

    except Exception as e:
        print("Error getting Gemini info for PDF:", e)
        about_text = (
            "Detailed AI-generated information could not be fetched at this moment. "
            "Please consult a qualified doctor for more guidance."
        )
        precautions = []

    # --- Build PDF ---
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, safe_text("AI Disease Prediction Report"), ln=True, align="C")
    pdf.ln(4)

    # Meta info
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, safe_text(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=True)

    user_email = session.get("user_email", "guest")
    pdf.cell(0, 8, safe_text(f"User: {user_email}"), ln=True)

    if age or gender or duration:
        pdf.cell(0, 8, safe_text(f"Age: {age or 'N/A'}"), ln=True)
        pdf.cell(0, 8, safe_text(f"Gender: {gender or 'N/A'}"), ln=True)
        pdf.cell(0, 8, safe_text(f"Duration of symptoms (days): {duration or 'N/A'}"), ln=True)

    pdf.ln(3)

    # Selected symptoms
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, safe_text("Selected Symptoms:"), ln=True)
    pdf.set_font("Arial", "", 11)

    if selected_symptoms:
        for s in selected_symptoms:
            pdf.cell(0, 6, safe_text(f"- {s.replace('_', ' ')}"), ln=True)
    else:
        pdf.cell(0, 6, safe_text("None provided."), ln=True)

    pdf.ln(4)

    # Main prediction
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, safe_text("Predicted Condition:"), ln=True)
    pdf.set_font("Arial", "", 11)

    if confidence is not None:
        conf_str = f"{confidence*100:.2f}%"
    else:
        conf_str = "N/A"

    pdf.cell(0, 6, safe_text(f"Disease: {disease}"), ln=True)
    pdf.cell(0, 6, safe_text(f"Model Confidence: {conf_str}"), ln=True)

    # Risk level text (same logic as UI)
    risk_label = "Low Risk"
    if confidence is not None:
        if confidence >= 0.8:
            risk_label = "High Risk"
        elif confidence >= 0.5:
            risk_label = "Moderate Risk"

    pdf.cell(0, 6, safe_text(f"Risk Level: {risk_label}"), ln=True)
    pdf.ln(4)

    # Top-3 predictions
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, safe_text("Top 3 Probable Conditions:"), ln=True)
    pdf.set_font("Arial", "", 11)

    if top_predictions:
        for i, p in enumerate(top_predictions[:3], start=1):
            c = p.get("confidence")
            if c is not None:
                c_str = f"{c*100:.2f}%"
            else:
                c_str = "N/A"
            pdf.cell(0, 6, safe_text(f"{i}. {p.get('disease','Unknown')} - {c_str}"), ln=True)
    else:
        pdf.cell(0, 6, safe_text("No additional predictions available."), ln=True)

    pdf.ln(4)

    # About section from Gemini
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, safe_text("About This Condition (AI Summary):"), ln=True)
    pdf.set_font("Arial", "", 11)
    if about_text:
        pdf.multi_cell(0, 6, safe_text(about_text))
    else:
        pdf.cell(0, 6, safe_text("No additional information available."), ln=True)

    pdf.ln(3)

    # Precautions from Gemini
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, safe_text("Suggested Precautions (AI Generated):"), ln=True)
    pdf.set_font("Arial", "", 11)
    if precautions:
        for p in precautions:
            pdf.cell(0, 6, safe_text(f"- {p}"), ln=True)
    else:
        pdf.cell(0, 6, safe_text("No specific precautions listed."), ln=True)

    pdf.ln(4)

    # Disclaimer
    pdf.set_font("Arial", "I", 9)
    disclaimer_text = (
        "Disclaimer: This report is generated by an AI-based screening system and is not a "
        "substitute for professional medical diagnosis or treatment. Always consult a qualified "
        "healthcare provider for any medical concerns."
    )
    pdf.multi_cell(0, 5, safe_text(disclaimer_text))

    # Save to /reports folder
    reports_dir = os.path.join(BASE_DIR, "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    file_path = os.path.join(reports_dir, "ai_disease_report.pdf")
    pdf.output(file_path)

    return send_file(
        file_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="ai_disease_report.pdf"
    )

    # --- Save PDF to disk instead of using BytesIO ---

    # Create /reports folder one level above this file (if not exist)
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    file_path = os.path.join(reports_dir, "ai_disease_report.pdf")
    pdf.output(file_path)

    # Send the file as a download
    return send_file(
        file_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="ai_disease_report.pdf"
    )

@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    """
    Request JSON:
    {
        "question": "string",
        "prediction": { ... },   # lastPrediction from frontend (optional)
        "symptoms": ["fever", "cough", ...]
    }
    Response JSON:
    { "answer": "text..." } or { "error": "..." }
    """
    data = request.get_json() or {}
    user_question = (data.get("question") or "").strip()
    prediction = data.get("prediction") or {}
    symptoms = data.get("symptoms") or []

    # If prediction info is missing but we have symptoms, recompute:
    if (not prediction) and symptoms:
        prediction = make_prediction(symptoms)

    disease = prediction.get("disease", "Unknown")
    confidence = prediction.get("confidence", None)
    top_predictions = prediction.get("top_predictions", [])

    # Build prompt (same safety every time)
    lines = []
    lines.append("You are an AI medical information assistant.")
    lines.append("You must NOT give a confirmed diagnosis or prescribe medicine.")
    lines.append("Always encourage the user to consult a doctor for actual medical decisions.")
    lines.append("")
    lines.append("Here is an AI model based disease risk assessment:")
    lines.append(f"- Predicted disease: {disease}")
    if confidence is not None:
        lines.append(f"- Model confidence: {confidence*100:.2f}%")
    if top_predictions:
        lines.append("- Top possible conditions:")
        for p in top_predictions[:3]:
            c = p.get("confidence")
            if c is not None:
                conf_str = f"{c*100:.2f}%"
            else:
                conf_str = "N/A"
            lines.append(f"  - {p.get('disease','Unknown')} ({conf_str})")
    if symptoms:
        lines.append(f"- Reported symptoms: {', '.join(symptoms)}")
    lines.append("")
    lines.append("User question:")
    lines.append(user_question or "Explain this result in simple language.")
    lines.append("")
    lines.append("Answer in simple, calm language. Include only general information and lifestyle guidance.")
    lines.append("Do NOT sound like you are confirming a diagnosis. End by reminding them to visit a doctor.")

    prompt = "\n".join(lines)

    try:
        # Send to your Render Gemini API
        payload = {"message": prompt}

        resp = requests.post(GEMINI_API_URL, json=payload, timeout=20)

        # If HTTP error (e.g. 500, 404)
        try:
            resp.raise_for_status()
        except Exception as http_err:
            print("Gemini HTTP error:", http_err, "Status:", resp.status_code, "Body:", resp.text[:300])
            return jsonify({"error": "AI assistant is not available right now."}), 500

        # Try JSON first
        answer = None
        try:
            resp_json = resp.json()
            # Adjust based on your Render response structure
            answer = resp_json.get("reply") or resp_json.get("text") or resp_json.get("content")
        except Exception as json_err:
            # Not JSON? fall back to plain text
            print("Gemini JSON parse error:", json_err, "Raw text:", resp.text[:300])
            answer = None

        if not answer:
            # If still no answer, use raw text
            answer = resp.text.strip() or "AI response could not be parsed."

        return jsonify({"answer": answer})

    except Exception as e:
        print("Error calling Gemini API:", e)
        return jsonify({"error": "AI assistant is not available right now."}), 500

# --- Auth APIs: signup, login, logout, current user ---

@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not name or not email or not password:
        return jsonify({"success": False, "message": "All fields are required."}), 400

    if find_user_by_email(email):
        return jsonify({"success": False, "message": "Email is already registered."}), 400

    users = load_users()
    new_id = max([u.get("id", 0) for u in users] or [0]) + 1

    new_user = {
        "id": new_id,
        "name": name,
        "email": email,
        "password": password,  # plain text for project; mention hashing in future scope
        "role": "patient"
    }
    users.append(new_user)
    save_users(users)

    # Auto-login after signup
    session["user_email"] = email
    session["user_name"] = name
    session["user_role"] = "patient"

    return jsonify({
        "success": True,
        "message": "Signup successful.",
        "name": name,
        "role": "patient"
    })


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required."}), 400

    user = find_user_by_email(email)
    if not user or user.get("password") != password:
        return jsonify({"success": False, "message": "Invalid email or password."}), 400

    session["user_email"] = user["email"]
    session["user_name"] = user["name"]
    session["user_role"] = user.get("role", "patient")

    return jsonify({
        "success": True,
        "message": "Login successful.",
        "name": user["name"],
        "role": user.get("role", "patient")
    })


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/current-user", methods=["GET"])
def api_current_user():
    if "user_email" not in session:
        return jsonify({"logged_in": False})
    return jsonify({
        "logged_in": True,
        "email": session.get("user_email"),
        "name": session.get("user_name"),
        "role": session.get("user_role", "patient")
    })

@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Return prediction history for the currently logged-in user.
    Reads from data/predictions.csv and filters by user_email.
    Always returns JSON, even on error.
    """
    if "user_email" not in session:
        return jsonify({"success": False, "message": "Not logged in."}), 401

    user_email = session.get("user_email")
    records = []

    try:
        if os.path.exists(PREDICTIONS_CSV):
            with open(PREDICTIONS_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("user_email") == user_email:
                        # Parse symptoms into list
                        symptoms_str = row.get("symptoms", "")
                        symptoms_list = [
                            s.strip() for s in symptoms_str.split(",") if s.strip()
                        ]

                        # Parse confidence
                        conf_raw = row.get("confidence")
                        try:
                            conf = float(conf_raw)
                        except (TypeError, ValueError):
                            conf = None

                        records.append({
                            "timestamp": row.get("timestamp", ""),
                            "disease": row.get("disease", ""),
                            "confidence": conf,
                            "symptoms": symptoms_list,
                        })

        # Sort latest first (string sort works for "YYYY-MM-DD HH:MM:SS")
        records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

        return jsonify({"success": True, "records": records})

    except Exception as e:
        print("Error in /api/history:", e)
        return jsonify({
            "success": False,
            "message": "Server error while loading history."
        }), 500

@app.route("/api/admin-data", methods=["GET"])
def api_admin_data():
    """
    Return system-wide data for admin:
    - total_users
    - total_predictions
    - most_common_disease
    - full predictions list
    """
    # Auth check
    if "user_email" not in session or session.get("user_role") != "admin":
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    try:
        # Load users
        users = load_users()
        total_users = len(users)

        predictions = []

        # Load predictions from CSV
        if os.path.exists(PREDICTIONS_CSV):
            with open(PREDICTIONS_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    conf_raw = row.get("confidence")
                    try:
                        conf = float(conf_raw)
                    except (TypeError, ValueError):
                        conf = None

                    predictions.append({
                        "timestamp": row.get("timestamp", ""),
                        "user_email": row.get("user_email", ""),
                        "disease": row.get("disease", ""),
                        "confidence": conf,
                        "symptoms": row.get("symptoms", "")
                    })

        total_predictions = len(predictions)

        # Most common disease
        disease_count = {}
        for p in predictions:
            d = p["disease"]
            if d:
                disease_count[d] = disease_count.get(d, 0) + 1

        most_common_disease = None
        if disease_count:
            most_common_disease = max(disease_count, key=disease_count.get)

        return jsonify({
            "success": True,
            "total_users": total_users,
            "total_predictions": total_predictions,
            "most_common_disease": most_common_disease,
            "predictions": predictions
        })

    except Exception as e:
        print("ADMIN DATA ERROR:", e)
        return jsonify({"success": False, "message": "Server error while loading admin data."}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
