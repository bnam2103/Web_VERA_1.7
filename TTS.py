import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers.models.speecht5 import SpeechT5HifiGan
import re
from scipy.io.wavfile import write
import string
from num2words import num2words
import numpy as np
from datetime import datetime
from pathlib import Path

# ------------------------------
# Load model, processor, vocoder
# ------------------------------

processor = SpeechT5Processor.from_pretrained(
    r"C:\Users\User\Documents\Fine_Tuning_Projects\TTS_Project\First_american_accent\speecht5_finetuned_voxpopuli_nl\checkpoint-8648"
)

model = SpeechT5ForTextToSpeech.from_pretrained(
    r"C:\Users\User\Documents\Fine_Tuning_Projects\TTS_Project\First_american_accent\speecht5_finetuned_voxpopuli_nl\checkpoint-8648",
    use_safetensors=True,
    trust_remote_code=True
)

vocoder = SpeechT5HifiGan.from_pretrained(
    r"C:\Users\User\Documents\VERA\local_vocoder"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
vocoder = vocoder.to(device)

# ------------------------------
# Speaker embedding
# ------------------------------

speaker_embedding = torch.tensor([[-0.0743, -0.0103,  0.0260,  0.0237,  0.0045, -0.0173, -0.0721, -0.0579,
          0.0374,  0.0206, -0.0648, -0.0665,  0.0259,  0.0414,  0.0323,  0.0512,
         -0.0078,  0.0259,  0.0123,  0.0155,  0.0371,  0.0255, -0.0156, -0.0398,
         -0.0612, -0.0098, -0.0582, -0.0046,  0.0377,  0.0320, -0.0028,  0.0450,
          0.0136, -0.0471,  0.0584, -0.0672,  0.0124,  0.0591, -0.0767, -0.0775,
          0.0142, -0.0590,  0.0407,  0.0436,  0.0238, -0.1164, -0.0200,  0.0116,
         -0.0551,  0.0721,  0.0228,  0.0490,  0.0465,  0.0149, -0.0871, -0.0100,
          0.0324,  0.0294,  0.0441,  0.0122,  0.0189, -0.0091, -0.0154,  0.0116,
          0.0376,  0.0224,  0.0141, -0.0388, -0.0615, -0.0467,  0.0216,  0.0115,
          0.0205,  0.0136,  0.0419,  0.0258,  0.0181,  0.0173, -0.0678, -0.0821,
         -0.0862, -0.0480, -0.0566, -0.0387, -0.0345, -0.0636, -0.0453,  0.0519,
          0.0190,  0.0681,  0.0282, -0.0694, -0.0032, -0.0608,  0.0649, -0.0070,
          0.0200,  0.0304, -0.0486, -0.0640,  0.0396, -0.1017, -0.0794,  0.0478,
          0.0425, -0.0547,  0.0486,  0.0480,  0.0169,  0.0227, -0.0807,  0.0313,
          0.0611, -0.0058,  0.0498,  0.0242, -0.0534, -0.0267, -0.0341, -0.0348,
          0.0220, -0.0662,  0.0370,  0.0365, -0.0660,  0.0279, -0.0644,  0.0143,
          0.0326,  0.0500,  0.0300,  0.0072,  0.0336,  0.0345,  0.0276, -0.0646,
         -0.0484, -0.0059, -0.0605,  0.0012,  0.0081,  0.0036, -0.0033,  0.0463,
         -0.0506,  0.0270, -0.0066, -0.0609,  0.0493, -0.1155,  0.0447, -0.0371,
         -0.0567,  0.0285,  0.0146,  0.0203,  0.0108, -0.0639, -0.0762,  0.0279,
          0.0205,  0.0018,  0.0158, -0.0595, -0.0299,  0.0084,  0.0270,  0.0379,
          0.0132,  0.0510,  0.0261, -0.0636,  0.0276, -0.0498,  0.0167,  0.0027,
         -0.0372,  0.0067,  0.0527, -0.0707,  0.0391, -0.0644,  0.0172,  0.0347,
         -0.0643, -0.0093,  0.0371,  0.0346, -0.0542, -0.0589, -0.0141,  0.0344,
         -0.0659,  0.0478,  0.0131,  0.0165,  0.0172,  0.0042,  0.0322, -0.0516,
          0.0523,  0.0285, -0.0554,  0.0056, -0.0021,  0.0150,  0.0391, -0.0400,
          0.0248, -0.0332,  0.0047, -0.0792, -0.0429,  0.0398, -0.0565,  0.0409,
          0.0457, -0.0870,  0.0314,  0.0226, -0.0816,  0.0377, -0.0779, -0.0134,
          0.0412,  0.0425,  0.0585,  0.0799,  0.0527,  0.0279, -0.0557,  0.0240,
          0.0306, -0.0370,  0.0098, -0.0225,  0.0299,  0.0527, -0.0011,  0.0456,
         -0.0768,  0.0237, -0.0966,  0.0106, -0.0521,  0.0512,  0.0424,  0.0236,
          0.0301,  0.0044,  0.0502,  0.0307,  0.0095, -0.0570,  0.0166,  0.0166,
          0.0321,  0.0367, -0.0677,  0.0514,  0.0165, -0.0601,  0.0407,  0.0401,
          0.0020,  0.0015,  0.0574,  0.0310, -0.0053, -0.0610,  0.0391, -0.0212,
          0.0271, -0.0256, -0.0613,  0.0301,  0.0564,  0.0209, -0.0815, -0.0544,
         -0.0091,  0.0303,  0.0256, -0.0597, -0.0593,  0.0376,  0.0184,  0.0580,
         -0.1039,  0.0021,  0.0159,  0.0319, -0.0386,  0.0322, -0.0432,  0.0292,
          0.0096,  0.0047,  0.0127,  0.0264,  0.0627,  0.0366,  0.0212, -0.0772,
          0.0303,  0.0400,  0.0267,  0.0290,  0.0309,  0.0488,  0.0430,  0.0153,
         -0.0187,  0.0440, -0.0995, -0.0837, -0.0254,  0.0274, -0.0638,  0.0500,
         -0.0568,  0.0611, -0.0643,  0.0084,  0.0148, -0.0675,  0.0311,  0.0652,
         -0.0648, -0.0791, -0.0660,  0.0231,  0.0096,  0.0477, -0.0702,  0.0503,
          0.0446, -0.0523, -0.0305, -0.0593,  0.0238, -0.0557,  0.0130,  0.0067,
         -0.0756,  0.0354,  0.0289,  0.0261, -0.0466, -0.0584, -0.0441, -0.0355,
         -0.0699,  0.1035,  0.0268, -0.0459, -0.0062,  0.0283, -0.0462,  0.0247,
         -0.1061,  0.0222, -0.0052,  0.0058,  0.0479, -0.0126, -0.0533,  0.0160,
          0.0042,  0.0476,  0.0133,  0.0263, -0.0822,  0.0167, -0.0129, -0.0026,
          0.0359,  0.0130,  0.0528,  0.0397, -0.0638, -0.1078,  0.0214,  0.0292,
          0.0351, -0.0545,  0.0406, -0.0787,  0.0306,  0.0389,  0.0332,  0.0178,
         -0.0405,  0.0238,  0.0087,  0.0140,  0.0397, -0.0856, -0.0334, -0.0002,
         -0.0025, -0.0352,  0.0299, -0.0384,  0.0179,  0.0057,  0.0005, -0.0593,
         -0.0505, -0.0592, -0.0831,  0.0174,  0.0417, -0.0128,  0.0286, -0.0422,
         -0.0141, -0.0779, -0.1574, -0.0493, -0.0533, -0.0075,  0.0274, -0.0474,
         -0.0516,  0.0257,  0.0360,  0.0330,  0.0212, -0.0346, -0.0637, -0.0165,
         -0.0254,  0.0295,  0.0180,  0.0093,  0.0260,  0.0096, -0.0626, -0.0537,
          0.0172,  0.0479,  0.0311,  0.0023,  0.0482,  0.0456,  0.0232,  0.0089,
         -0.0030, -0.0109,  0.0400,  0.0059,  0.0046,  0.0122,  0.0007, -0.0109,
          0.0188, -0.0746, -0.0615,  0.0463, -0.0136,  0.0101,  0.0435,  0.0257,
         -0.0516,  0.0282,  0.0218, -0.0788,  0.0135,  0.0192, -0.0027,  0.0225,
          0.0103,  0.0045, -0.0529, -0.0672,  0.0158, -0.0058,  0.0440,  0.0572,
         -0.0373, -0.0386,  0.0256,  0.0211,  0.0453,  0.0515, -0.0624,  0.0371,
         -0.0205, -0.0121, -0.0542,  0.0136, -0.0411,  0.0284,  0.0219, -0.0009,
         -0.0469, -0.0276, -0.0797, -0.0664,  0.0094,  0.0443, -0.0661,  0.0388,
         -0.0244, -0.0143, -0.0674,  0.0379,  0.0583, -0.0234,  0.0413, -0.0651]]).to(device)

# ------------------------------
# Utilities
# ------------------------------

def chunk_text(text, max_chars=200):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""

    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current.strip():
        chunks.append(current.strip())

    return chunks

# ------------------------------
# MAIN TTS FUNCTION (UPDATED)
# ------------------------------

def speak_to_file(text: str, output_path: Path) -> Path:
    """
    Generate speech audio from text and write to output_path.
    output_path MUST be provided by the caller.
    Returns the output_path.
    """

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # =========================
    # TEXT NORMALIZATION
    # =========================

    text = re.sub(r"\bvera\b", "vairah", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnam\b", "nom", text, flags=re.IGNORECASE)
    # Replace e.g. with "for example" in all forms
    text = re.sub(r'e\.g\.', 'for example', text, flags=re.IGNORECASE)
    text = re.sub(r'i\.e\.', 'that is', text, flags=re.IGNORECASE)
    def replace_decimals(match):
        num = match.group()
        return num.replace('.', ' point ')

    # Replace decimal numbers first
    text = re.sub(r'\b\d+\.\d+\b', replace_decimals, text)
    def remove_commas_in_numbers(text: str) -> str:
    # Replace commas that are between digits with nothing
        return re.sub(r'(?<=\d),(?=\d)', '', text)
    text = remove_commas_in_numbers(text)

    def remove_hyphens_between_words(text: str) -> str:
    # Replace hyphens (with optional spaces around) that are between words with a single space
        return re.sub(r'(?<=[A-Za-z])\s*-\s*(?=[A-Za-z])', ' ', text)

    text = remove_hyphens_between_words(text)

    def replace_x_multiplication(text: str) -> str:
        # Replace lowercase x between two numbers with *
        return re.sub(r'(?<=\d)\s*[xX]\s*(?=\d)', ' * ', text)
    text = replace_x_multiplication(text)

    def fix_numeric_hyphens(text: str) -> str:
        # Case 1: numeric ranges like "250-500 calories" → "250 to 500 calories"
        text = re.sub(r'(\$?\d+)\s*-\s*(\$?\d+)', lambda m: f"{m.group(1)} to {m.group(2)}", text)

        # Case 2: pure math expression (digit - digit) → "minus"
        text = re.sub(r'(?<=\d)\s*-\s*(?=\d)', ' minus ', text)

        return text
    text = fix_numeric_hyphens(text)

    def replace_dollars(match):
        amount = match.group(1)
        words = num2words(int(amount))
        return f"{words} dollars"
    
    text = re.sub(r'\$(\d+)', replace_dollars, text)

    def normalize_subscript_numbers(text: str) -> str:
        subscript_map = {
            "₀": "0",
            "₁": "1",
            "₂": "2",
            "₃": "3",
            "₄": "4",
            "₅": "5",
            "₆": "6",
            "₇": "7",
            "₈": "8",
            "₉": "9"
        }
        # Replace any subscript characters with normal digits
        return ''.join(subscript_map.get(char, char) for char in text)
    text = normalize_subscript_numbers(text)

    def split_chemical_formulas(text: str) -> str:
        # Match pattern: one or more letters followed by one or more digits
        return re.sub(r'\b([A-Z]+)(\d+)\b', r'\1 \2', text)
    text = split_chemical_formulas(text)

    def replace_numbers(match):
        num = match.group()
        return num2words(int(num))
    
    text = re.sub(r'\b\d+\b', replace_numbers, text)

    letter_map = {
        'A': 'ay', 'B': 'bee', 'C': 'cee', 'D': 'dee', 'E': 'ee', 'F': 'ef', 'G': 'gee',
        'H': 'aitch', 'I': 'eye', 'J': 'jay', 'K': 'kay', 'L': 'el', 'M': 'em', 'N': 'en',
        'O': 'oh', 'P': 'pee', 'Q': 'cue', 'R': 'ar', 'S': 'ess', 'T': 'tee', 'U': 'you',
        'V': 'vee', 'W': 'double you', 'X': 'ex', 'Y': 'why', 'Z': 'zee'
    }

    known_acronyms = {
        "NASA": "NAHSA",
        "UNESCO": "Yoo Neh SKOW",
        "NATO": "NAYTOH",
        "RADAR": "RAYDAR",
        "SCUBA": "scuba",
        "≠": "does not equal",
        "°C": "degrees Celsius",
        "°F": "degrees Fahrenheit",
    }

    def replace_all_caps(match):
        word = match.group()
        if word in known_acronyms:
            return known_acronyms[word]  # use custom pronunciation
        else:
            # spell out each letter for unknown acronyms
            spelled = ' '.join(letter_map.get(c, c) for c in word if c in string.ascii_uppercase)
            return spelled
    text = re.sub(r'\b[A-Z]{2,}\b', replace_all_caps, text)

    mispronounced_words = {
        "radar": "raydar",
        "queue": "kyoo",
        "pizza": "peet zah",
        "buses": "buhses",
        "epitome": "ih pit uh me",
        "avocados": "av uh cah dough",
        "Python": "Pie thon",
        "dataset": "data set",
        "Mega": "MAY Gah",
        "Console": "Con sole",
        "Giga": "Gig Gah",
        "GigaBytes": "GigGa Bytes",
        "GigaByte": "GigGa Byte",
        "Pythagorean": "Pie thagorean",
        "Vietnam": "Vee et nom",
        "Saigon": "Sigh gon",
        "double bass": "double base",
        "base": "base",
        "Irvine": "Ir-vine",
        "Celcius": "Sell see us",
        "anime": "ah-knee-may",
        "email" : "e-mail",
        "parameters": "puhrammiters"
    }

    def replace_mispronounced(text: str) -> str:
        # Use regex to match whole words only, case-insensitive
        for word, replacement in mispronounced_words.items():
            pattern = r'\b' + re.escape(word) + r'\b'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    text = replace_mispronounced(text)
    
    def remove_double_stars(text: str) -> str:
        # Remove ** around any phrase
        return re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = remove_double_stars(text)

    symbol_map = {
        '+': 'plus',
        '*': 'times',
        '/': 'divided by',
        '=': 'equals',
        '@': 'at',
        '&': 'and',
        '%': 'percent',
        '#': 'hash',
        '^': 'to the power of',
        '<': 'is less than',
        '>': 'is greater than',
        '_': 'underscore',
    }

    for sym, word in symbol_map.items():
        text = text.replace(sym, f" {word} ")

    # =========================
    # TTS GENERATION
    # =========================

    chunks = chunk_text(text)
    audio_chunks = []

    for chunk in chunks:
        inputs = processor(text=chunk, return_tensors="pt").to(device)

        with torch.no_grad():
            waveform = model.generate_speech(
                inputs["input_ids"],
                speaker_embedding,
                vocoder=vocoder,
            )

        audio_chunks.append(waveform.squeeze().cpu().numpy())

    # Concatenate chunks
    final_audio = np.concatenate(audio_chunks)

    # Normalize to int16
    final_audio = np.clip(final_audio, -1.0, 1.0)
    final_audio = (final_audio * 32767).astype(np.int16)

    # Write WAV
    write(output_path, 16000, final_audio)

    return output_path

# base = Path(r"C:\Users\User\Documents\VERA\Online_demo\static\fillers")

# speak_to_file(
#     "Give me a moment, sir.",
#     base / "moment.wav"
# )

# speak_to_file(
#     "One second, sir.",
#     base / "one_second.wav"
# )

# speak_to_file(
#     "Give me a second, sir.",
#     base / "give_me_a_second.wav"
# )

# speak_to_file(
#     "One moment, sir.",
#     base / "one_moment.wav"
# )