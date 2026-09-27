import re

def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", str(text).replace("\n"," "))
    return [p.strip() for p in parts if len(p.strip().split()) >= 5]

def generate_flashcards(title, text, count=5, difficulty="medium"):
    s = split_sentences(text)
    if not s:
        return [{"question": f"What is {title}?", "answer": str(text).strip()}]
    prompts = {
        "easy": ["What does this article explain?","What is one important fact about this topic?"],
        "medium": ["What important information is given about this topic?","What does the article state about this topic?"],
        "hard": ["What specific information does the article provide?","What is an important detail that should be remembered?"]
    }
    p = prompts.get(difficulty, prompts["medium"])
    cards = [{"question": f"What is {title}?", "answer": s[0]}]
    for i, sentence in enumerate(s[1:]):
        if len(cards) >= count:
            break
        cards.append({"question": p[i % len(p)], "answer": sentence})
    return cards[:count]
