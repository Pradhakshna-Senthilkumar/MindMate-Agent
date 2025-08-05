import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import warnings
import json
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

GOOGLE_API_KEY = "AIzaSyDv9fUd-1CjN4bhSnVMH6z2_Cknam_NyJY"  # Replace with your real key
genai.configure(api_key=GOOGLE_API_KEY)

# ---------- Persistent Chat History (JSON) ----------
HISTORY_FILE = "chat_history.json"

def save_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump(chat_history, f)

def load_history():
    global chat_history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            chat_history[:] = json.load(f)
    else:
        chat_history[:] = []

# ---------- Tools ----------
def emotional_support_tool(query):
    responses = [
        "I'm here for you. It's okay to feel this way; many students have similar struggles.",
        "Remember, reaching out for help is a sign of strength.",
        "Taking deep breaths and giving yourself a break can help when things feel overwhelming."
    ]
    return responses[hash(query) % len(responses)]

def wellbeing_advice_tool(query):
    tips = [
        "Try a short walk or breathing exercise to clear your mind.",
        "Connect with a friend or family member for a quick chat.",
        "Break big tasks into smaller steps—celebrate each small win!"
    ]
    return tips[hash(query) % len(tips)]

tools = [
    Tool(
        name="EmotionalSupport",
        func=emotional_support_tool,
        description="Provide empathetic support messages for stressed, anxious, or upset students."
    ),
    Tool(
        name="WellbeingAdvice",
        func=wellbeing_advice_tool,
        description="Offer practical tips to improve mental health and wellbeing."
    )
]

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False,
    max_iterations=8,
    max_execution_time=120
)

# ---------- Flask App ----------
app = Flask(__name__)
CORS(app)

chat_history = []
load_history()  # Load chat history from file at startup

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    chat_history.append(("You", user_message))
    response = agent.invoke({"input": user_message, "chat_history": chat_history})
    bot_reply = response["output"] if isinstance(response, dict) and "output" in response else str(response)
    chat_history.append(("MindMate", bot_reply))
    save_history()  # Save chat history after each exchange
    return jsonify({"reply": bot_reply})

@app.route("/history", methods=["GET"])
def history():
    # Return a list of {"sender":..., "text":...} objects for frontend compatibility
    return jsonify([{"sender": sender, "text": text} for sender, text in chat_history])

@app.route("/")
def index():
    return send_from_directory(".", "chat_page.html")

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)

if __name__ == "__main__":
    app.run(debug=True, port=8000)