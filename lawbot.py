import torch
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from law_retrieval import get_relevant_laws_for_query  # laws module

# =========================
# CONFIG
# =========================
MODEL_DIR = "./category_model_simple"   # trained category model
ISSUES_CSV = "issues_master.csv"        # sub-topic + advice data
MAX_LENGTH = 128

# =========================
# LOAD CATEGORY MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device for category model:", device)

category_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
category_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
category_model.to(device)
category_model.eval()

with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_category(text: str) -> str:
    """Predict main legal category for a given query."""
    enc = category_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = category_model(**enc)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()

    label = label_encoder.inverse_transform([pred_id])[0]
    return label

# =========================
# LOAD SUB-CATEGORY + ADVICE DATA
# =========================
issues = pd.read_csv(ISSUES_CSV, encoding="utf-8")
issues.columns = issues.columns.str.strip().str.lower()

required_cols = ["issue_id", "category", "example_text"]
for col in required_cols:
    if col not in issues.columns:
        raise ValueError(f"Column '{col}' not found in issues_master.csv. Found: {issues.columns}")

# advice_text is optional but recommended
has_advice = "advice_text" in issues.columns

issues["category"] = issues["category"].astype(str).str.strip().str.upper()
issues["example_text"] = issues["example_text"].astype(str)
if has_advice:
    issues["advice_text"] = issues["advice_text"].astype(str)


def predict_subtopic(user_text: str, category: str):
    """
    Return (subtopic_id, matched_example) for TENANCY / LABOUR,
    or (None, None) if no matching entries / other categories.
    """
    category = category.upper()
    subset = issues[issues["category"] == category]
    if subset.empty:
        return None, None

    texts = subset["example_text"].tolist()
    ids = subset["issue_id"].tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    user_vec = vectorizer.transform([user_text])

    sims = cosine_similarity(user_vec, X).flatten()
    best_idx = sims.argmax()

    return ids[best_idx], texts[best_idx]


def get_advice_for_issue(issue_id: str, category: str):
    """
    Look up advice_text for a given issue_id from issues_master.
    Only meaningful for TENANCY / LABOUR.
    """
    if not has_advice:
        # No advice column in CSV
        if category in ["TENANCY", "LABOUR"]:
            return "Detailed advice for this sub-topic is under preparation."
        else:
            return "Detailed advice is currently available only for Tenancy and Labour matters."

    row = issues[issues["issue_id"] == issue_id]
    if row.empty:
        if category in ["TENANCY", "LABOUR"]:
            return "Detailed advice for this sub-topic is under preparation."
        else:
            return "Detailed advice is currently available only for Tenancy and Labour matters."

    advice = str(row.iloc[0].get("advice_text", "")).strip()
    if advice:
        return advice

    # fallback messages
    if category in ["TENANCY", "LABOUR"]:
        return "Detailed advice for this sub-topic is under preparation."
    else:
        return "Detailed advice is currently available only for Tenancy and Labour matters."

# =========================
# MAIN PIPELINE
# =========================
def process_query(user_text: str):
    """
    Full pipeline:
    1. Predict category
    2. Predict sub-topic (if TENANCY / LABOUR)
    3. Retrieve relevant laws
    4. Retrieve advice (for TENANCY / LABOUR)
    """
    category = predict_category(user_text)

    subtopic_id = None
    matched_example = None
    advice_text = None

    if category in ["TENANCY", "LABOUR"]:
        subtopic_id, matched_example = predict_subtopic(user_text, category)
        if subtopic_id is not None:
            advice_text = get_advice_for_issue(subtopic_id, category)
        else:
            advice_text = "Detailed advice for this issue is under preparation."
    else:
        advice_text = "Detailed advice is currently available only for Tenancy and Labour matters."

    laws = get_relevant_laws_for_query(user_text, category)

    return category, subtopic_id, matched_example, laws, advice_text


if __name__ == "__main__":
    print("\n==== Category + Sub-category + Laws + Advice ====")
    print("Type 'quit' to exit.\n")

    while True:
        q = input("Enter legal query: ").strip()
        if q.lower() in ("quit", "exit"):
            break

        category, subtopic_id, matched_example, laws, advice = process_query(q)

        print("\nPredicted Category:", category)

        if subtopic_id is not None:
            print("Predicted Sub-category (issue_id):", subtopic_id)
            print("Matched Example from issues_master:")
            print("  ", matched_example)
        else:
            if category in ["TENANCY", "LABOUR"]:
                print("Sub-category: Not found (check issues_master.csv)")
            else:
                print("Sub-category: Not applicable for this category")

        print("\nRelevant Laws:")
        for l in laws:
            print("  -", l)

        print("\nAdvice:")
        print(advice)

        print("\n" + "-" * 80 + "\n")
