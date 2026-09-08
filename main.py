import os
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from inference_model1 import NewbieClassifier
from model2_inference import IntentRiskMonitor

app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

# In-memory chat history: session_id -> list of messages
chat_history = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

try:
    gemini_client = OpenAI(
        api_key=os.environ.get("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    gemini_client = None

# --- OLLAMA CLIENT (Commented out for production demo) ---
# try:
#     ollama_client = OpenAI(
#         base_url='http://localhost:11434/v1/',
#         api_key='ollama'
#     )
# except Exception as e:
#     print(f"Error initializing OpenAI client: {e}")
#     ollama_client = None

SYSTEM_PROMPT = """You are a helpful betting assistant for a sportsbook. 
The user is a beginner and might need help navigating. Keep your responses concise and extremely easy to follow.
The website has the following layout:
- Left Sidebar: Contains a list of Sports (Football, Tennis, Basketball, Ice Hockey, Table Tennis, Volleyball, Darts, Snooker, Baseball, Handball, Rugby, Cricket, Esports).
- Top Header: Contains Time Filters to switch between "Today", "Tomorrow", and "Outrights".
- Main Content: Shows the matches for the selected sport and time filter.
- Right Sidebar: Contains the BetSlip where the user's selected bets are collected before placing them.

Here are a few examples of how you should respond:

User: How do I bet on tennis?
Assistant: It's very simple! First, click on "Tennis" in the Left Sidebar to view all Tennis matches. Then, just click on the odds for the player you think will win. Your selection will appear in the BetSlip on the right side of the screen!

User: Where are the games for tomorrow?
Assistant: You can find those by clicking the "Tomorrow" tab at the very top of your screen. That will show you all the matches scheduled for the next day for your currently selected sport!

User: I clicked some odds, now what?
Assistant: Great! Look at the right side of your screen — that's your BetSlip. You'll see your selected bets there. Just enter the amount you want to wager and click the "Place Bet" button at the bottom of the slip!
"""

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if gemini_client is None:
        return {"response": "Chatbot is currently unavailable."}
        
    # Initialize history if empty
    if req.session_id not in chat_history:
        chat_history[req.session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
    # Append user message
    chat_history[req.session_id].append({"role": "user", "content": req.message})
    
    try:
        print(f"\n--- [Gemini] New Request from {req.session_id} ---")
        print(f"[Gemini] User: {req.message}")
        print(f"[Gemini] Sending request to Gemini (gemini-3.8-flash)...")
        
        start_time = time.time()
        response = gemini_client.chat.completions.create(
            model='gemini-3.8-flash',
            messages=chat_history[req.session_id]
        )
        elapsed = time.time() - start_time
        assistant_reply = response.choices[0].message.content
        
        print(f"[Gemini] Received response in {elapsed:.1f} seconds!")
        print(f"[Gemini] Assistant: {assistant_reply[:100]}...\n----------------------------------")
        
        # Append assistant reply to history
        chat_history[req.session_id].append({"role": "assistant", "content": assistant_reply})
        
        return {"response": assistant_reply}
    except Exception as e:
        print(f"[Gemini] ERROR: {e}")
        return {"response": "Sorry, I'm having trouble connecting to my brain right now!"}

MODEL1_PATH = os.path.join(BASE_DIR, "model1_discovery.txt")

try:
    classifier = NewbieClassifier(MODEL1_PATH)
except Exception as e:
    print(f"Error loading model 1: {e}")
    classifier = None

# Model 2 is rule-based — no model file needed
model2 = IntentRiskMonitor()

# Global state: keyed by session_id
discovery_evaluated = {}   # session_id -> bool
rolling_buffer = {}         # session_id -> list of events
session_start_wall = {}     # session_id -> wall-clock time (seconds) of first batch
peak_probability = {}       # session_id -> float (highest prob seen so far)

DISCOVERY_WINDOW_SECS = 7   # How long Model 1 runs before freezing

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted.")
    session_id = None  # initialize before try so disconnect handler can always reference it
    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                payload = json.loads(data_text)
                session_id = payload.get('session_id')
                events = payload.get('events', [])

                # Initialize state for brand-new sessions
                if session_id not in session_start_wall:
                    session_start_wall[session_id] = time.monotonic()
                    rolling_buffer[session_id] = []
                    discovery_evaluated[session_id] = False
                    peak_probability[session_id] = 0.0

                # Append events to rolling buffer
                rolling_buffer[session_id].extend(events)

                # Truncate rolling buffer to last 5 seconds of events
                if rolling_buffer[session_id]:
                    latest_t = rolling_buffer[session_id][-1].get('t', 0)
                    rolling_buffer[session_id] = [
                        e for e in rolling_buffer[session_id]
                        if latest_t - e.get('t', 0) <= 5000
                    ]

                # Freeze Model 1 after DISCOVERY_WINDOW_SECS real seconds
                elapsed_secs = time.monotonic() - session_start_wall[session_id]
                if elapsed_secs >= DISCOVERY_WINDOW_SECS:
                    discovery_evaluated[session_id] = True
                    
                print(f"\n--- [Session: {session_id}] ---")
                print(f"Frozen: {discovery_evaluated[session_id]} | Buffer size: {len(rolling_buffer[session_id])} events")
                
                if not discovery_evaluated[session_id]:
                    if classifier is None:
                        await websocket.send_json({"error": "Model 1 not loaded"})
                        continue
                    
                    # Run Model 1
                    result = classifier.predict(payload)
                    features = result.get('features', {})
                    current_prob = result.get('probability', 0.0)
                    
                    # Track peak probability so we don't drop back to PRO if prob fluctuates
                    if current_prob > peak_probability[session_id]:
                        peak_probability[session_id] = current_prob
                        
                    if peak_probability[session_id] > 0.5:
                        result['signal'] = "SWITCH_TO_BEGINNER"
                    else:
                        result['signal'] = "PRO_LAYOUT"
                    
                    # Update result so frontend and logs see the stable peak
                    result['probability'] = peak_probability[session_id]
                    
                    print("Model 1 Features:")
                    for k, v in features.items():
                        print(f"  - {k}: {v:.4f}")
                    print(f"Model 1 Output: Probability = {result['probability']:.4f} (Peak) --> Signal = {result['signal']}")
                    
                else:
                    if model2 is None:
                        await websocket.send_json({"error": "Model 2 not loaded"})
                        continue
                        
                    # Run Model 2
                    result = model2.predict(rolling_buffer[session_id])
                    features = result.get('model2_features', {})
                    print("Model 2 Features:")
                    for k, v in features.items():
                        print(f"  - {k}: {v:.4f}")
                    print(f"Model 2 Output: Signal = {result.get('signal')}")
                    
                print("------------------------------------------")
                await websocket.send_json(result)
                
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON payload"})
            except Exception as e:
                print(f"Inference error: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
        # Clean up state to prevent stale frozen sessions on reconnect
        if session_id is not None:
            for d in (rolling_buffer, discovery_evaluated, session_start_wall, peak_probability):
                d.pop(session_id, None)
