# ============================================================
# RETRIEVER.PY
# Wikipedia Dataset Retriever
# ============================================================

import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SEARCH RESULT
# ============================================================

class SearchResult(dict):

    def __init__(self, title, text, score):

        super().__init__(
            title=title,
            text=text,
            score=score
        )

        self.title = title
        self.text = text
        self.score = score


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:

    def __init__(
        self,
        dataset_file="enwiki_namespace_0_00015.parquet"
    ):

        print("Loading retrieval dataset...")

        self.dataset_file = dataset_file

        self.df = dataset_file.copy()

        # Make sure required columns exist
        if "name" not in self.df.columns:
            raise ValueError(
                "Dataset does not contain 'name' column."
            )

        if "abstract" not in self.df.columns:
            raise ValueError(
                "Dataset does not contain 'abstract' column."
            )

        # Clean missing values
        self.df["name"] = (
            self.df["name"]
            .fillna("")
            .astype(str)
        )

        self.df["abstract"] = (
            self.df["abstract"]
            .fillna("")
            .astype(str)
        )

        # Normalized title
        self.df["_title_normalized"] = (
            self.df["name"]
            .apply(self.normalize)
        )

        # Build searchable text
        self.df["_search_text"] = (
            self.df["name"] + " " +
            self.df["abstract"]
        )

        print(
            "Retrieval dataset loaded:",
            len(self.df),
            "articles"
        )

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        print("Building TF-IDF index...")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100000,
            sublinear_tf=True
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["_search_text"]
        )

        print("TF-IDF index ready!")

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    @staticmethod
    def normalize(text):

        text = str(text).lower()

        # Replace different dash characters
        text = text.replace("–", "-")
        text = text.replace("—", "-")

        # Replace punctuation with spaces
        text = re.sub(
            r"[^a-z0-9]+",
            " ",
            text
        )

        # Remove extra spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    # ========================================================
    # TOKENIZE
    # ========================================================

    @staticmethod
    def tokens(text):

        normalized = Retriever.normalize(text)

        return set(
            word
            for word in normalized.split()
            if len(word) > 1
        )

    # ========================================================
    # EXACT TITLE SEARCH
    # ========================================================

    def exact_title_search(self, query):

        normalized_query = self.normalize(query)

        matches = self.df[
            self.df["_title_normalized"] == normalized_query
        ]

        results = []

        for _, row in matches.head(5).iterrows():

            results.append(
                SearchResult(
                    title=row["name"],
                    text=row["abstract"],
                    score=1.0
                )
            )

        return results

    # ========================================================
    # TITLE CONTAINS QUERY
    # ========================================================

    def title_search(self, query):

        normalized_query = self.normalize(query)

        if not normalized_query:
            return []

        query_tokens = self.tokens(query)

        candidates = []

        for index, row in self.df.iterrows():

            title = row["_title_normalized"]

            if not title:
                continue

            title_tokens = self.tokens(title)

            # Query must have meaningful overlap
            common = query_tokens.intersection(
                title_tokens
            )

            if not common:
                continue

            # Calculate title overlap
            overlap = len(common) / max(
                len(query_tokens),
                1
            )

            # Exact phrase inside title
            phrase_bonus = 0.0

            if normalized_query in title:
                phrase_bonus = 0.5

            score = overlap + phrase_bonus

            candidates.append(
                (
                    index,
                    score
                )
            )

        # Highest score first
        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for index, score in candidates[:10]:

            row = self.df.iloc[index]

            results.append(
                SearchResult(
                    title=row["name"],
                    text=row["abstract"],
                    score=float(score)
                )
            )

        return results

    # ========================================================
    # TF-IDF SEARCH
    # ========================================================

    def tfidf_search(
        self,
        query,
        k=10
    ):

        query = str(query).strip()

        if not query:
            return []

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        # Get candidate indexes
        candidate_indexes = similarities.argsort()[
            ::-1
        ][:k * 5]

        query_tokens = self.tokens(query)

        scored = []

        for index in candidate_indexes:

            row = self.df.iloc[index]

            title = row["name"]

            title_tokens = self.tokens(title)

            # ----------------------------------------------
            # Title overlap
            # ----------------------------------------------

            common = query_tokens.intersection(
                title_tokens
            )

            title_overlap = len(common) / max(
                len(query_tokens),
                1
            )

            # ----------------------------------------------
            # Exact phrase bonus
            # ----------------------------------------------

            normalized_title = self.normalize(
                title
            )

            normalized_query = self.normalize(
                query
            )

            phrase_bonus = 0.0

            if normalized_query in normalized_title:

                phrase_bonus = 0.70

            # ----------------------------------------------
            # Exact title bonus
            # ----------------------------------------------

            exact_bonus = 0.0

            if normalized_title == normalized_query:

                exact_bonus = 2.0

            # ----------------------------------------------
            # Final score
            # ----------------------------------------------

            tfidf_score = float(
                similarities[index]
            )

            final_score = (
                tfidf_score * 0.45
                +
                title_overlap * 0.35
                +
                phrase_bonus
                +
                exact_bonus
            )

            scored.append(
                (
                    index,
                    final_score
                )
            )

        # Highest score first
        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        used_titles = set()

        for index, score in scored:

            row = self.df.iloc[index]

            title = str(row["name"])

            title_key = self.normalize(
                title
            )

            if title_key in used_titles:
                continue

            used_titles.add(title_key)

            results.append(
                SearchResult(
                    title=title,
                    text=row["abstract"],
                    score=float(score)
                )
            )

            if len(results) >= k:
                break

        return results

    # ========================================================
    # MAIN RETRIEVAL FUNCTION
    # ========================================================

    def retrieve(
        self,
        query,
        k=5
    ):

        query = str(query).strip()

        if not query:
            return []

        print()
        print("========================================")
        print("RETRIEVING WIKIPEDIA ARTICLES")
        print("QUERY:", query)
        print("========================================")

        # ----------------------------------------------------
        # STEP 1: Exact title
        # ----------------------------------------------------

        exact_results = self.exact_title_search(
            query
        )

        if exact_results:

            print(
                "Exact title match found!"
            )

            for result in exact_results[:k]:

                print(
                    f"RETRIEVED: {result.title}"
                )

                print(
                    f"SCORE: {result.score:.4f}"
                )

            return exact_results[:k]

        # ----------------------------------------------------
        # STEP 2: Strong title match
        # ----------------------------------------------------

        title_results = self.title_search(
            query
        )

        # If there is a strong title match,
        # prefer it over a weak TF-IDF match.

        strong_title_results = [
            result
            for result in title_results
            if result.score >= 0.70
        ]

        if strong_title_results:

            print(
                "Strong title match found!"
            )

            for result in strong_title_results[:k]:

                print(
                    f"RETRIEVED: {result.title}"
                )

                print(
                    f"SCORE: {result.score:.4f}"
                )

            return strong_title_results[:k]

        # ----------------------------------------------------
        # STEP 3: TF-IDF semantic retrieval
        # ----------------------------------------------------

        results = self.tfidf_search(
            query,
            k=k
        )

        print(
            "TF-IDF retrieval completed."
        )

        for result in results:

            print(
                f"RETRIEVED: {result.title}"
            )

            print(
                f"SCORE: {result.score:.4f}"
            )

        return results

    # ========================================================
    # SEARCH ALIAS
    # ========================================================

    def search(
        self,
        query,
        k=5
    ):

        return self.retrieve(
            query,
            k=k
        )

    # ========================================================
    # GET TOP RESULT
    # ========================================================

    def get_best_result(
        self,
        query
    ):

        results = self.retrieve(
            query,
            k=1
        )

        if results:

            return results[0]

        return None


# ============================================================
# TEST RETRIEVER DIRECTLY
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("      RETRIEVER TEST")
    print("========================================")

    retriever = Retriever()

    test_topics = [
        "Machine Learning",
        "Artificial Intelligence",
        "Python",
        "Database",
        "Hitchin Thorpe"
    ]

    for topic in test_topics:

        print()
        print("----------------------------------------")
        print("TEST TOPIC:", topic)
        print("----------------------------------------")

        results = retriever.retrieve(
            topic,
            k=5
        )

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{i}. {result.title}"
            )

            print(
                f"   Score: {result.score:.4f}"
            )

    print()
    print("========================================")
    print("        RETRIEVER TEST COMPLETE")
    print("========================================")