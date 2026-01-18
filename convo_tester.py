from LLM import VeraAI   # make sure filename is llm.py
import sys
import os
from datetime import datetime

# =========================
# CONFIG
# =========================

MODEL_PATH = r"C:\Users\User\Documents\Fine_Tuning_Projects\LLAMA_LLM_3B"
CHAT_LOG_DIR = "chat_log"

# =========================
# INIT VERA
# =========================

vera = VeraAI(model_path=MODEL_PATH)

# =========================
# CHAT LOG SETUP
# =========================

os.makedirs(CHAT_LOG_DIR, exist_ok=True)

session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
chat_log_path = os.path.join(CHAT_LOG_DIR, f"chat_{session_id}.txt")

def log_line(prefix, text):
    with open(chat_log_path, "a", encoding="utf-8") as f:
        f.write(f"{prefix}: {text}\n\n")

# =========================
# CHAT STATE
# =========================

messages = [
    {
        "role": "system",
        "content": vera.base_system_prompt
    }
]

# Optional: log system prompt once (comment out if undesired)
log_line("SYSTEM", vera.base_system_prompt)

print("=== VERA AI Chat ===")
print("Type 'exit' to quit.\n")

# =========================
# CHAT LOOP
# =========================

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("VERA: Goodbye.")
        break

    # Add user message
    messages.append({
        "role": "user",
        "content": user_input
    })

    log_line("YOU", user_input)

    # Generate reply
    reply = vera.generate(messages)

    print("VERA:", reply, "\n")

    # Add assistant replydo
    messages.append({
        "role": "assistant",
        "content": reply
    })

    log_line("VERA", reply)
