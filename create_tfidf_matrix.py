import os
import joblib
import pandas as pd
from scipy.sparse import save_npz


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "lexora_tfidf_vectorizer.pkl"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "lexora_embedding_metadata.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "lexora_tfidf_matrix.npz"
)


print("=" * 60)
print("LEXORA — CREATING TF-IDF SEARCH MATRIX")
print("=" * 60)


# ------------------------------------------------------------
# Load trained TF-IDF vectorizer
# ------------------------------------------------------------

print("\nLoading TF-IDF vectorizer...")

vectorizer = joblib.load(
    TFIDF_PATH
)


# ------------------------------------------------------------
# Load clause metadata
# ------------------------------------------------------------

print("Loading clause metadata...")

metadata = pd.read_csv(
    METADATA_PATH
)


if "Clause" not in metadata.columns:

    raise ValueError(
        "The metadata file does not contain a 'Clause' column."
    )


clauses = (
    metadata["Clause"]
    .fillna("")
    .astype(str)
    .tolist()
)


print(
    f"Found {len(clauses)} clauses."
)


# ------------------------------------------------------------
# Transform clauses using the EXISTING trained vectorizer
# ------------------------------------------------------------

print("\nCreating TF-IDF matrix...")

tfidf_matrix = vectorizer.transform(
    clauses
)


print(
    f"Matrix shape: {tfidf_matrix.shape}"
)


# ------------------------------------------------------------
# Save sparse matrix
# ------------------------------------------------------------

print("\nSaving matrix...")

save_npz(
    OUTPUT_PATH,
    tfidf_matrix
)


print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)

print(
    f"\nMatrix shape: {tfidf_matrix.shape}"
)

print("\nYou can now use this matrix for similarity search.")
