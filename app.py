from flask import Flask, render_template, request, jsonify
from data_loader import load_dataset, dataset_stats
from retriever import Retriever
from flashcard_generator import generate_flashcards
from quiz_generator import generate_quiz

app = Flask(__name__)
DATASET_FILE = "enwiki_namespace_0_00015.parquet"

print("Loading dataset...")
df = load_dataset(DATASET_FILE)
print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", list(df.columns))

retriever = Retriever(df)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json() or {}
    topic = str(data.get("topic", "")).strip()
    count = max(1, min(int(data.get("count", 5)), 10))
    difficulty = str(data.get("difficulty", "medium"))

    if not topic:
        return jsonify(error="Please enter a topic."), 400

    results = retriever.search(topic, 5)
    if not results:
        return jsonify(error="No matching article found. Try another topic."), 404

    article = results[0]
    cards = generate_flashcards(article["title"], article["text"], count, difficulty)

    return jsonify(
        flashcards=cards,
        source=article["title"],
        stats={"words": len(article["text"].split()), "articles": len(df)}
    )

@app.route("/search", methods=["POST"])
def search():
    topic = str((request.get_json() or {}).get("topic", "")).strip()
    if not topic:
        return jsonify(error="Please enter a topic."), 400
    return jsonify(results=retriever.search(topic, 10))

@app.route("/summary", methods=["POST"])
def summary():
    topic = str((request.get_json() or {}).get("topic", "")).strip()
    results = retriever.search(topic, 1)
    if not results:
        return jsonify(error="No matching article found."), 404
    article = results[0]
    sentences = [x.strip() for x in article["text"].replace("\n", " ").split(".") if x.strip()]
    text = ". ".join(sentences[:3])
    if text and not text.endswith("."):
        text += "."
    return jsonify(title=article["title"], summary=text, source=article["title"])

@app.route("/quiz", methods=["POST"])
def quiz():
    data = request.get_json() or {}
    topic = str(data.get("topic", "")).strip()
    count = max(1, min(int(data.get("count", 5)), 10))
    results = retriever.search(topic, 1)
    if not results:
        return jsonify(error="No matching article found."), 404
    article = results[0]
    return jsonify(title=article["title"], quiz=generate_quiz(article["title"], article["text"], count))

@app.route("/stats")
def stats():
    return jsonify(dataset_stats(df))

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
