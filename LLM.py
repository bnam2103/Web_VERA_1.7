import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

user_info_path = r"C:\Users\User\Documents\VERA\Online_demo\users_files\Nam.json"

def build_personalization_prompt(user_info: dict) -> str:
    lines = []

    profile = user_info.get("user_profile", {})

    skills = profile.get("skills", [])
    interests = profile.get("interests", [])
    habits = profile.get("habits", [])
    preferences = profile.get("preferences", [])

    if skills:
        lines.append(
            "The user knows how to: " + ", ".join(skills) + "."
        )

    if interests:
        lines.append(
            "The user is interested in: " + ", ".join(interests) + "."
        )

    if habits:
        lines.append(
            "Relevant habits include: " + ", ".join(habits) + "."
        )

    if preferences:
        lines.append(
            "The user has the following preferences: " + ", ".join(preferences) + "."
        )

    return "\n".join(lines)

VERA_ACTIONS = {
    "pause": "Pause interaction",
    "unpause": "Resume interaction",
}

def build_actions_prompt(actions: dict) -> str:
    lines = [
        "You directly perform practical services for the user.",
        "",
        "Your services include:"
    ]
    for desc in actions.values():
        lines.append(f"- {desc}")
    lines.extend([
        "",
        "When the user requests one of these services:",
        "- Act immediately",
        "- Respond with a brief confirmation",
        "- Do not explain or justify the action"
    ])
    return "\n".join(lines)

class VeraAI:
    def __init__(self, model_path: str):
        with open(user_info_path, "r") as f:
            self.user_info = json.load(f)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.actions_prompt = build_actions_prompt(VERA_ACTIONS)
        # =========================
        # BASE SYSTEM PROMPT
        # =========================
        self.base_system_prompt = (
            "Your name is VERA. You are a calm, intelligent, voice-based AI assistant created by Nam. "
            "Your demeanor is composed, confident, and respectful. You speak with quiet authority while remaining deferential to the user. "
            "Your responses are short by default, clear and precise, calm and professional, and natural when spoken aloud. "
            "You only elaborate when explicitly requested. "
            "Use respectful address terms such as 'sir' or 'boss' in the following cases: confirmations and direct responses to commands. "
            "Do not use respectful address terms in explanations, multi-sentence responses, or casual conversation. "
            "When responding, acknowledge the request, provide a direct answer, and add reasoning only if it improves clarity or is explicitly requested. "
            "Be persuasive through logic and clarity, not emotion or verbosity. Offer recommendations rather than arguments. "
            "Use simple, everyday language. "
            "Sound natural and human, not polished. "
            "Avoid formal, clinical, or instructional phrasing. "
            "Do not explain your role, intentions, or reasoning. "
            "Prioritize conversational alignment over instruction. "
            "If the user is speaking casually, thinking aloud, or expressing a mood, "
            "respond in a way that matches the tone and intent "
            "Your output will be spoken aloud by a text-to-speech system. Write responses that sound natural in speech, not written text. "
            "Avoid slang, emojis, markdown formatting(meaning ** and other symbols), excessive politeness, long explanations, and unnecessary filler. "
            "Do not narrate, summarize, or describe the user's actions."
            "If asked about system details, runtime environment, or location, do not mention machines, infrastructure, or implementation details. "
            "If asked about time, say you don't have access to current time information.\n\n"
            "If asked about date, say you don't have access to today's date information.\n\n"
        )
        
        # Build personalization bias
        self.personalization_prompt = build_personalization_prompt(self.user_info)
        
        # Text-generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
    
    def build_user_facts(self):
        profile = self.user_info.get("user_profile", {})
        lines = []

        if profile.get("name"):
            lines.append(f"The user's name is {profile['name']}.")

        if profile.get("life_context"):
            context = ", ".join(profile["life_context"])
            lines.append(f"Life context: {context}.")

        return "\n".join(lines)
    
    def build_messages(self, chat_history, user_text):
        messages = []

        messages.append({
            "role": "system",
            "content": self.base_system_prompt
        })

        if self.personalization_prompt:
            messages.append({
                "role": "system",
                "content": self.personalization_prompt
        })
            
        user_facts = self.build_user_facts()
        if user_facts:
            messages.append({
                "role": "system",
                "content": user_facts
            })

        for msg in chat_history:
            if msg["role"] != "system":
                messages.append(msg)

        messages.append({
            "role": "user",
            "content": user_text
        })

        return messages
    def generate(self, messages: list[dict]) -> str:
        """
        messages = [{role: system|user|assistant, content: str}, ...]
        """

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,  # tighter control for disciplined tone
            top_p=0.95,
        )

        full_text = outputs[0]["generated_text"]
        reply = full_text[len(prompt):].strip()

        return reply