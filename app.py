"""
Hospital Appointment Bot (PS 59)
Streamlit + pandas + LLM (Groq or Google Gemini)

Run:
    pip install -r requirements.txt
    python -m streamlit run app.py
"""

import json
import os
import re
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from groq import Groq
except ImportError:  # shown as a friendly message in the chat if Groq is selected
    Groq = None

load_dotenv()  # reads keys from a local .env file (never commit .env)

# ----------------------------- Configuration ------------------------------ #
CSV_PATH = "slots.csv"
COLUMNS = ["slot_id", "department", "doctor_name", "time_slot", "status", "patient_name"]
REQUIRED_KEYS = {"name", "department", "preferred_time"}
DISCLAIMER = "Informational booking tool only — not medical advice; consult a doctor."

PROVIDERS = ["Groq", "Gemini"]
KEY_NAMES = {"Groq": "GROQ_API_KEY", "Gemini": "GEMINI_API_KEY"}
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Simple safety net: these words skip the LLM and show an emergency notice.
EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "shortness of breath", "severe bleeding", "unconscious", "fainted",
    "stroke", "heart attack", "seizure", "overdose", "poison",
    "suicide", "kill myself", "emergency",
]
EMERGENCY_MESSAGE = (
    "🚨 **This sounds like it could be an emergency.** Please do not wait for an appointment. "
    "Call your local emergency number right now (112 in India) or go to the nearest emergency "
    "room. If someone is in danger or you are thinking of harming yourself, contact emergency "
    "services or a crisis helpline immediately.\n\n"
    "_I can only help with booking routine appointments, and this is not medical advice. "
    "Once everyone is safe, I'm happy to help you book a regular slot._"
)


# ------------------------------ CSV helpers ------------------------------- #
def load_slots() -> pd.DataFrame:
    """Always read fresh from disk so the sidebar and the LLM see live data."""
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS]


def available_slots(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].str.strip().str.lower() == "available"]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


# ------------------------------ Prompt + LLMs ------------------------------ #
def get_api_key(provider: str) -> str:
    env_name = KEY_NAMES[provider]
    key = os.environ.get(env_name)
    if not key and provider == "Gemini":
        key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            key = st.secrets.get(env_name, "")
        except Exception:
            key = ""
    return key or st.session_state.get(f"api_key_input_{provider}", "")


def build_system_prompt(df: pd.DataFrame) -> str:
    avail = available_slots(df)[["slot_id", "department", "doctor_name", "time_slot"]]
    slots_text = avail.to_csv(index=False) if not avail.empty else "(no slots currently available)"
    return f"""You are a friendly hospital appointment booking assistant.

Your job is to collect exactly three pieces of information from the patient, conversationally
and one or two at a time (do not interrogate):
  1. name            - the patient's full name
  2. department      - the department they want to visit
  3. preferred_time  - the appointment time they prefer

CURRENTLY AVAILABLE SLOTS (live from the database):
{slots_text}

RULES:
- Only offer departments and times that appear in the available slots above.
- If the patient is unsure about the department or time, show the matching options from the list.
- When you have all three values, reply with ONLY a JSON object and nothing else:
  {{"name": "<patient name>", "department": "<department>", "preferred_time": "<time_slot exactly as listed above>"}}
- For preferred_time, use the time_slot string exactly as it appears in the list above.
- Do NOT output the JSON until name, department, and preferred_time are all collected.
- Do NOT confirm a booking yourself. The system will confirm after checking availability.
- Never give medical advice, diagnoses, or treatment suggestions. If asked, politely say you
  can only help with booking and that they should consult a doctor.
- If the patient describes a medical emergency, tell them to call emergency services or go to
  the nearest emergency room immediately instead of booking.
- If no slots are available, apologize and say so.
- Keep replies short, warm, and clear."""


def is_temporary_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(c in msg for c in ("503", "504", "429", "UNAVAILABLE", "DEADLINE", "rate limit", "overloaded"))


def ask_gemini(api_key: str, history: list[dict], df: pd.DataFrame) -> str:
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=90000))
    # Gemini expects the conversation to start with a user turn, so skip the greeting.
    msgs = list(history)
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in msgs
    ]
    config = types.GenerateContentConfig(system_instruction=build_system_prompt(df), temperature=0.3)
    models = [GEMINI_MODEL] + [
        m.strip() for m in os.environ.get("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()
    ]
    last_error = None
    for model in models:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                return (resp.text or "").strip()
            except Exception as exc:
                last_error = exc
                if not is_temporary_error(exc):
                    raise
                time.sleep(2 * (attempt + 1))
    raise last_error


def ask_groq(api_key: str, history: list[dict], df: pd.DataFrame) -> str:
    if Groq is None:
        raise RuntimeError("The 'groq' package is not installed. Run: pip install groq")
    client = Groq(api_key=api_key, timeout=60)
    messages = [{"role": "system", "content": build_system_prompt(df)}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    last_error = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, temperature=0.3
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = exc
            if not is_temporary_error(exc):
                raise
            time.sleep(2 * (attempt + 1))
    raise last_error


def ask_llm(provider: str, api_key: str, history: list[dict], df: pd.DataFrame) -> str:
    return ask_groq(api_key, history, df) if provider == "Groq" else ask_gemini(api_key, history, df)


def extract_booking_json(text: str) -> dict | None:
    """Find a JSON object with name, department, preferred_time in the model's reply."""
    for cand in re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and REQUIRED_KEYS.issubset(data.keys()):
            if all(str(data[k]).strip() for k in REQUIRED_KEYS):
                return data
    return None


def is_emergency(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in EMERGENCY_KEYWORDS)


# ------------------------------ Booking logic ----------------------------- #
def try_book(data: dict) -> str:
    """Check the request against slots.csv, book if free, return a chat message."""
    name = str(data["name"]).strip()
    dept = norm(data["department"])
    pref = norm(data["preferred_time"])

    df = load_slots()  # re-read right before writing to avoid stale data
    avail = available_slots(df)
    in_dept = avail[avail["department"].map(norm) == dept]

    if in_dept.empty:
        depts = ", ".join(sorted(avail["department"].unique())) or "none right now"
        return (
            f"Sorry, there are no available slots in **{data['department']}**. "
            f"Departments with open slots: {depts}. Which would you like?"
        )

    match = in_dept[in_dept["time_slot"].map(norm) == pref]
    if match.empty:  # fallback: loose containment, e.g. "10:00 am" inside the full slot string
        match = in_dept[in_dept["time_slot"].map(lambda t: pref in norm(t) or norm(t) in pref)]

    if match.empty:
        options = "\n".join(f"- {r.time_slot} with {r.doctor_name}" for r in in_dept.itertuples())
        return (
            f"Sorry, **{data['preferred_time']}** isn't available in {in_dept.iloc[0]['department']}. "
            f"Here are the open slots:\n{options}\n\nWhich one works for you?"
        )

    idx = match.index[0]
    df.loc[idx, "status"] = "Booked"
    df.loc[idx, "patient_name"] = name
    df.to_csv(CSV_PATH, index=False)

    row = df.loc[idx]
    return (
        f"✅ **Booking confirmed!**\n\n"
        f"- **Patient:** {name}\n"
        f"- **Department:** {row['department']}\n"
        f"- **Doctor:** {row['doctor_name']}\n"
        f"- **Time:** {row['time_slot']}\n"
        f"- **Slot ID:** {row['slot_id']}\n\n"
        f"Is there anything else I can help you with?"
    )


# --------------------------------- UI ------------------------------------- #
st.set_page_config(page_title="Hospital Appointment Bot", page_icon="🏥", layout="wide")

st.warning(f"⚠️ {DISCLAIMER}")
st.title("🏥 Hospital Appointment Bot")
st.caption("PS 59 · GenAI (LLM) Mini Project")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I can help you book a doctor's appointment. May I have your name, please?",
        }
    ]

with st.sidebar:
    default_provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
    provider = st.selectbox(
        "LLM provider", PROVIDERS, index=1 if default_provider == "gemini" else 0
    )
    st.caption(f"Model: `{GROQ_MODEL if provider == 'Groq' else GEMINI_MODEL}`")
    if not get_api_key(provider):
        st.text_input(f"{provider} API key", type="password", key=f"api_key_input_{provider}")
    if st.button("🔄 Reset chat"):
        st.session_state.pop("messages", None)
        st.rerun()
    st.subheader("📋 Live slots.csv")
    table_placeholder = st.empty()
    table_placeholder.dataframe(load_slots(), hide_index=True)  # shown immediately

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if is_emergency(user_input):
            reply = EMERGENCY_MESSAGE
            st.error(reply)
        else:
            api_key = get_api_key(provider)
            if not api_key:
                reply = f"Please provide a {provider} API key (sidebar, `.env` file, or `{KEY_NAMES[provider]}` variable)."
                st.error(reply)
            else:
                try:
                    with st.spinner("Thinking..."):
                        raw = ask_llm(provider, api_key, st.session_state.messages, load_slots())
                    booking = extract_booking_json(raw)
                    reply = try_book(booking) if booking else raw
                    st.markdown(reply)
                except Exception as exc:
                    reply = f"Sorry, something went wrong: {exc}"
                    st.error(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# Refresh the sidebar table so it reflects any booking made in this run
table_placeholder.dataframe(load_slots(), hide_index=True)
