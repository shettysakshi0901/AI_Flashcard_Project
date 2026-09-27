import re
import random

def generate_quiz(title, text, count=5):
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", str(text).replace("\n"," ")) if len(x.strip().split()) >= 7]
    if not parts:
        return []
    out = []
    for i, correct in enumerate(parts[:count]):
        others = [x for j,x in enumerate(parts) if j != i][:3]
        while len(others) < 3:
            others.append("This information is not stated in the article.")
        options = [correct] + others
        random.shuffle(options)
        out.append({
            "question": f"Which statement is supported by the article about {title}?",
            "options": options,
            "answer": correct
        })
    return out
