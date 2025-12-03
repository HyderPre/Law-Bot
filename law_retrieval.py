import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============== CONFIG ==============
CSV_PATH = "Laws.csv"   # change if your file name is different
TOP_K = 3                        # how many similar examples to aggregate laws from
# ====================================

# 1. Load CSV (handle BOM / encoding issues on Windows)
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding="latin1")

# Normalize column names: "Text", "Category", "Relevant Laws"
clean_cols = {c.strip().lower(): c for c in df.columns}

required = ["text", "category", "relevant laws"]
for col in required:
    if col not in clean_cols:
        raise ValueError(
            f"Expected columns: Text, Category, Relevant Laws (any case, with/without spaces). Found: {df.columns}"
        )

text_col = clean_cols["text"]
cat_col = clean_cols["category"]
laws_col = clean_cols["relevant laws"]

df = df[[text_col, cat_col, laws_col]].rename(
    columns={text_col: "text", cat_col: "category", laws_col: "laws"}
)

df["text"] = df["text"].astype(str)
df["category"] = df["category"].astype(str).str.strip().str.upper()
df["laws"] = df["laws"].astype(str)

print("✅ Loaded law examples:", len(df))
print("Categories in laws dataset:", df["category"].unique())


def get_relevant_laws_for_query(user_text: str, category: str, top_k: int = TOP_K):
    """
    Given a user query and its predicted category,
    find the most similar past examples in that category and return merged laws.
    """

    # Match category in uppercase to avoid mismatch
    category = category.upper()
    subset = df[df["category"] == category]
    if subset.empty:
        return ["No law mapping available for this category."]

    texts = subset["text"].tolist()
    laws_list = subset["laws"].tolist()

    # TF-IDF over the example texts of this category
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    user_vec = vectorizer.transform([user_text])

    sims = cosine_similarity(user_vec, X).flatten()
    top_idx = sims.argsort()[::-1][:top_k]

    collected_laws = []
    for idx in top_idx:
        laws_str = laws_list[idx]

        # Normalize newlines to ';' so multi-line cells also work
        laws_str = laws_str.replace("\r\n", ";").replace("\n", ";")

        # allow multiple laws inside one string, separated by ';'
        for law in laws_str.split(";"):
            law = law.strip()
            if law and law not in collected_laws:
                collected_laws.append(law)

    if not collected_laws:
        return ["No specific laws found."]

    return collected_laws


if __name__ == "__main__":
    # tiny self-test
    q = "My landlord is refusing to return my security deposit after I vacated the flat"
    cat = "TENANCY"
    print("\nTest query:", q)
    print("Category:", cat)
    print("Relevant laws:")
    for l in get_relevant_laws_for_query(q, cat):
        print(" -", l)
