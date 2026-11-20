# Hospital Appointment Bot (PS 59)

**GenAI (LLM) Mini Project** · Category: Health \& Wellness

* **Name:** Yugandhar Khedkar
* **Class:** LY AIA 4
* **Roll No:** 59 (ADT23OSCB1370)

> ⚠️ \*\*Informational booking tool only — not medical advice; consult a doctor.\*\*

## Problem Statement

Booking doctor appointments by phone is slow. This project is a chatbot that collects a
patient's details through conversation and books a slot from the available timings.
It helps patients who want a quick booking and hospital front desks that handle many calls.

## How It Works

1. Doctor slots are stored in `slots.csv` (`slot\_id, department, doctor\_name, time\_slot, status, patient\_name`).
2. The currently available slots are passed into the LLM system prompt on every turn.
3. The LLM (Groq or Gemini) chats with the patient and collects `name`, `department` and `preferred\_time`,
then returns them as a JSON object.
4. Python parses the JSON, checks the slot in `slots.csv`, marks it `Booked` with the patient's
name, saves the CSV and confirms the booking in the chat.
5. A sidebar shows the live `slots.csv` table. Emergency keywords (e.g. "chest pain") skip the
LLM and show an emergency notice instead.

## Tech Stack

Python 3.10+, Streamlit, pandas, Groq API (`groq`, model `llama-3.3-70b-versatile`) with Google Gemini (`google-genai`) as an alternative provider, python-dotenv.

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/llm-ps59-<your-full-roll-no>.git
cd llm-ps59-<your-full-roll-no>

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\\Scripts\\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key (Groq: https://console.groq.com/keys, Gemini: https://aistudio.google.com/apikey)
cp .env.example .env           # Windows: copy .env.example .env
# then edit .env and paste your key (set LLM\_PROVIDER=groq or gemini)

# 5. Start the app
python -m streamlit run app.py
```

You can also skip the `.env` file and paste the key into the sidebar field when the app opens.

## Screenshots

!\[Booking conversation](screenshots/booking.png)
!\[Sidebar with updated slots](screenshots/slots\_updated.png)

## Project Files

|File|Purpose|
|-|-|
|`app.py`|Streamlit app, Gemini prompt, JSON parsing and booking logic|
|`slots.csv`|Doctor slot database|
|`requirements.txt`|Python dependencies|
|`.env.example`|Template for the API key (real `.env` is git-ignored)|

## Limitations

* Uses a CSV file, so it is not safe for many simultaneous users.
* Matching of the preferred time depends on the LLM returning the exact slot text.

