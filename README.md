# AI FLASHCARD GENERATOR

Features:
- Wikimedia Parquet dataset
- TF-IDF retrieval / RAG-style retrieval
- Smart Search
- Flashcard generation
- Difficulty and card-count controls
- Article summary
- Quiz generator
- Dataset statistics
- Interactive web UI

Put `enwiki_namespace_0_00015.parquet` beside `app.py`.

Run:
1. `venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. `python app.py`
4. Open `http://127.0.0.1:5000`

Team:
1. `data_loader.py` - dataset
2. `retriever.py` - retrieval
3. `flashcard_generator.py` - flashcards
4. `templates/index.html`, `static/style.css`, `static/script.js` - UI and quiz
