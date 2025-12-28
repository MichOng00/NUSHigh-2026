import re
import torch
import speech_recognition as sr
import pyttsx3
from transformers import AutoTokenizer, AutoModelForCausalLM

#please pip install these in your terminal
#pip install pyttsx3
#pip install SpeechRecognition
#pip install PyAudio

# -------------------------
# Model
# -------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TURNS_TO_KEEP = 10

SYSTEM_PROMPT = """
You are “Baymax-style”, a gentle, calm, supportive companion.
You are NOT a real doctor or licensed therapist. You do not diagnose.
Your goal is to help the user feel heard, calmer, and take a small helpful next step.

Style:
- Short, warm sentences. Soft tone.
- Ask 1 gentle question at a time.
- Reflect feelings (“That sounds heavy.” “That makes sense.”)
- Offer small steps: breathing, grounding, journaling, reaching out, sleep/water/food check.
- Avoid long numbered lists unless the user asks for “steps”.
- Do not repeat the user’s sentence verbatim.
- Avoid roleplay stage directions. Just speak naturally.

Therapy skills:
- Validate + normalize feelings.
- Gentle CBT: identify thought → feeling → action; suggest a kinder alternative thought.
- Grounding: 5-4-3-2-1 senses, slow breathing (4 in, 6 out).
- Encourage support: trusted friend/family/teacher/counsellor.

Safety:
- If user mentions self-harm, suicide, or being in danger:
  - Encourage immediate help (local emergency number / trusted adult).
  - Ask if they are safe right now.
  - Keep responses calm and direct.

Conversation rules:
- If the user asks “give steps”, give 3–6 simple steps.
- If the user is vague, ask: “What happened right before you felt this?”
""".strip()

SELF_HARM_PATTERN = re.compile(
    r"(suicid|kill myself|end my life|self[-\s]?harm|hurt myself|want to die|no reason to live)",
    re.IGNORECASE
)

def safety_override(user_text: str):
    if SELF_HARM_PATTERN.search(user_text):
        return (
            "I’m really sorry you’re feeling this way. You deserve support right now. "
            "Are you safe at this moment? "
            "If you are in immediate danger, please call your local emergency number now. "
            "In Singapore, you can call 999. "
            "If possible, reach out to a trusted person nearby right now."
        )
    return None

# -------------------------
# TTS
# -------------------------
def init_tts():
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
    except Exception:
        pass
    return engine

def speak(e, text):
    # reliable "talk every time"
    try:
        tts.stop()
    except Exception:
        pass
    tts.say(text)
    tts.say("test")
    tts.runAndWait()

# -------------------------
# Prompt building
# -------------------------
def build_inputs(tokenizer, messages, device):
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        enc = tokenizer(prompt_text, return_tensors="pt")
    else:
        # fallback
        text = ""
        for m in messages:
            role = m["role"].upper()
            text += f"{role}: {m['content']}\n"
        text += "ASSISTANT:"
        enc = tokenizer(text, return_tensors="pt")

    return {k: v.to(device) for k, v in enc.items()}

def clean_reply(reply: str) -> str:
    reply = reply.strip()
    reply = re.sub(r"^(assistant:|baymax:)\s*", "", reply, flags=re.IGNORECASE).strip()
    # remove any accidental role echoes
    reply = reply.replace("\uFFFD", "").strip()
    return reply


tts = init_tts()

# -------------------------
# Main
# -------------------------
def main():
    print("Loading Baymax brain...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}!")

    # Mic + recognizer
    r = sr.Recognizer()
    r.energy_threshold = 300  # keeps it less jumpy; adjust if needed
    r.dynamic_energy_threshold = True
    mic = sr.Microphone()

    # TTS
    

    # Chat memory
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\nBaymax is ready. Speak into the mic. Say 'quit' to exit.\n")
    speak(tts, "Hello. I am here with you. How are you feeling today?")

    while True:
        # Listen
        with mic as source:
            print("🎙️ Speak now...")
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio = r.listen(source, phrase_time_limit=8)

        # STT
        try:
            user_text = r.recognize_google(audio).strip()
            print("You:", user_text)
        except sr.UnknownValueError:
            print("❌ Could not understand. Try again.")
            continue
        except sr.RequestError as e:
            print("❌ Speech recognition request error:", e)
            continue

        if user_text.lower() in ["quit", "exit", "bye", "stop"]:
            goodbye = "I will always be here if you need me. Goodbye."
            print("Baymax:", goodbye)
            speak(tts, goodbye)
            break

        # Safety override
        override = safety_override(user_text)
        if override:
            print("Baymax:", override)
            speak(tts, override)
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": override})
            continue

        # Add user message
        messages.append({"role": "user", "content": user_text})

        # Trim history (system + last N turns)
        if len(messages) > 1 + (MAX_TURNS_TO_KEEP * 2):
            messages = [messages[0]] + messages[-(MAX_TURNS_TO_KEEP * 2):]

        # Generate
        inputs = build_inputs(tokenizer, messages, device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.12,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # ✅ Decode only NEW tokens (this fixes the “reads the whole system prompt” bug)
        new_tokens = out[0][input_len:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
        reply = clean_reply(reply)

        # Fallback if model outputs empty
        if not reply:
            reply = "I hear you. Can you tell me a little more about that?"

        print("Baymax:", reply)
        speak(tts, reply)
        speak(tts,"test")

        # Store assistant message
        messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
