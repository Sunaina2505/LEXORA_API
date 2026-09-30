
"""
============================================================
LEXORA — CONTRACT CLAUSE RISK ANALYSIS API
============================================================

FastAPI service exposing the trained LEXORA ML pipeline.

Pipeline:

Input Clause
     |
     +--> TF-IDF --> Calibrated SVM --> Risk + Confidence
     |
     +--> Sentence Transformer --> Semantic Search
                                      |
                                      +--> Clause Type
                                      +--> Risk Reason
                                      +--> Recommended Action
                                      +--> Similar Clauses
============================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RISK_MODEL_PATH = os.path.join(
    BASE_DIR,
    "lexora_risk_model.pkl"
)

SVM_MODEL_PATH = os.path.join(
    BASE_DIR,
    "lexora_svm_model.pkl"
)

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "lexora_tfidf_vectorizer.pkl"
)

EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "lexora_clause_embeddings.npy"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "lexora_embedding_metadata.csv"
)

EMBEDDING_MODEL_PATH = os.path.join(
    BASE_DIR,
    "lexora_embedding_model"
)


# ============================================================
# GLOBAL MODEL OBJECTS
# ============================================================

risk_model = None
svm_model = None
tfidf_vectorizer = None

embedding_model = None
clause_embeddings = None
metadata = None


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    global risk_model
    global svm_model
    global tfidf_vectorizer
    global embedding_model
    global clause_embeddings
    global metadata

    print("=" * 70)
    print("LEXORA — LOADING ML MODELS")
    print("=" * 70)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required_files = [
        RISK_MODEL_PATH,
        SVM_MODEL_PATH,
        TFIDF_PATH,
        EMBEDDINGS_PATH,
        METADATA_PATH,
        EMBEDDING_MODEL_PATH
    ]

    for path in required_files:

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    # --------------------------------------------------------
    # Load classification artifacts
    # --------------------------------------------------------

    print("\nLoading TF-IDF vectorizer...")
    tfidf_vectorizer = joblib.load(
        TFIDF_PATH
    )

    print("Loading Logistic Regression model...")
    risk_model = joblib.load(
        RISK_MODEL_PATH
    )

    print("Loading calibrated SVM model...")
    svm_model = joblib.load(
        SVM_MODEL_PATH
    )

    # --------------------------------------------------------
    # Load embedding artifacts
    # --------------------------------------------------------

    print("Loading clause embeddings...")

    clause_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print("Loading metadata...")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print("Loading Sentence Transformer...")
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_PATH
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LEXORA — MODEL VERIFICATION")
    print("=" * 70)

    print(
        f"\nTF-IDF features       : "
        f"{len(tfidf_vectorizer.get_feature_names_out())}"
    )

    print(
        f"Embedding matrix      : "
        f"{clause_embeddings.shape}"
    )

    print(
        f"Metadata shape        : "
        f"{metadata.shape}"
    )

    print(
        f"Embedding dimensions  : "
        f"{clause_embeddings.shape[1]}"
    )

    print(
        f"Metadata columns      : "
        f"{metadata.columns.tolist()}"
    )

    print(
        f"SVM classes           : "
        f"{svm_model.classes_.tolist()}"
    )

    print("\n✓ All LEXORA models loaded successfully")


# ============================================================
# LAZY MODEL LOADING
# ============================================================

# IMPORTANT FOR RENDER:
# Do not load the 86 MB Sentence Transformer during FastAPI
# startup. Render must detect the HTTP port first. Models are
# loaded on the first /predict or /search request instead.

models_loaded = False


def ensure_models_loaded():

    global models_loaded

    if not models_loaded:
        load_models()
        models_loaded = True


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="LEXORA ML API",
    description=(
        "Machine Learning API for contract clause "
        "risk analysis, confidence estimation, "
        "and semantic clause search."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class ClauseRequest(BaseModel):

    clause: str = Field(
        ...,
        min_length=5,
        max_length=10000,
        description="Contract clause to analyze"
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class SimilarClause(BaseModel):

    clause: str
    clause_type: str
    risk_level: str
    similarity: float


class PredictionResponse(BaseModel):

    clause: str
    clause_type: str
    risk_level: str
    confidence: float
    risk_reason: str
    recommended_action: str
    similar_clauses: List[SimilarClause]


# ============================================================
# HELPER — SEMANTIC SEARCH
# ============================================================

def semantic_search(
    clause: str,
    top_k: int = 5
):

    # Generate query embedding
    query_embedding = embedding_model.encode(
        [clause],
        normalize_embeddings=True
    )

    # Ensure numpy array
    query_embedding = np.asarray(
        query_embedding
    )

    # Existing embeddings were created for the
    # same Sentence Transformer model.
    #
    # Normalize them for cosine similarity.
    database_embeddings = clause_embeddings

    norms = np.linalg.norm(
        database_embeddings,
        axis=1,
        keepdims=True
    )

    normalized_database = (
        database_embeddings /
        np.maximum(norms, 1e-12)
    )

    # Cosine similarity
    similarities = (
        normalized_database @
        query_embedding[0]
    )

    # Get highest similarity indices
    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        row = metadata.iloc[int(idx)]

        results.append(
            {
                "clause": str(
                    row["Clause"]
                ),
                "clause_type": str(
                    row["Clause_Type"]
                ),
                "risk_level": str(
                    row["Risk_Level"]
                ),
                "risk_reason": str(
                    row["Risk_Reason"]
                ),
                "recommended_action": str(
                    row["Recommended_Action"]
                ),
                "similarity": float(
                    similarities[idx]
                )
            }
        )

    return results


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "LEXORA",
        "service": "Contract Clause Risk Analysis API",
        "version": "1.0.0",
        "status": "running"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    # The service is healthy even before ML artifacts are loaded.
    # They are intentionally loaded lazily for Render compatibility.
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "embedding_count": (
            len(clause_embeddings)
            if clause_embeddings is not None
            else 0
        )
    }


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: ClauseRequest):

    clause = request.clause.strip()

    if not clause:
        raise HTTPException(
            status_code=400,
            detail="Clause cannot be empty."
        )

    try:

        # Load ML artifacts only when an actual prediction is requested.
        ensure_models_loaded()

        # ----------------------------------------------------
        # STEP 1 — TF-IDF TRANSFORMATION
        # ----------------------------------------------------

        X = tfidf_vectorizer.transform(
            [clause]
        )

        # ----------------------------------------------------
        # STEP 2 — SVM RISK PREDICTION
        # ----------------------------------------------------

        predicted_risk = svm_model.predict(
            X
        )[0]

        # ----------------------------------------------------
        # STEP 3 — CONFIDENCE
        # ----------------------------------------------------

        probabilities = svm_model.predict_proba(
            X
        )[0]

        confidence = float(
            np.max(probabilities)
        )

        # ----------------------------------------------------
        # STEP 4 — SEMANTIC SEARCH
        # ----------------------------------------------------

        similar = semantic_search(
            clause,
            top_k=5
        )

        # ----------------------------------------------------
        # STEP 5 — DERIVE METADATA FROM
        # MOST SIMILAR CLAUSE
        # ----------------------------------------------------

        best_match = similar[0]

        clause_type = best_match[
            "clause_type"
        ]

        risk_reason = best_match[
            "risk_reason"
        ]

        recommended_action = best_match[
            "recommended_action"
        ]

        # ----------------------------------------------------
        # STEP 6 — BUILD RESPONSE
        # ----------------------------------------------------

        similar_response = []

        for item in similar:

            similar_response.append(
                SimilarClause(
                    clause=item["clause"],
                    clause_type=item["clause_type"],
                    risk_level=item["risk_level"],
                    similarity=round(
                        item["similarity"],
                        4
                    )
                )
            )

        return PredictionResponse(

            clause=clause,

            clause_type=clause_type,

            risk_level=str(
                predicted_risk
            ),

            confidence=round(
                confidence,
                4
            ),

            risk_reason=risk_reason,

            recommended_action=recommended_action,

            similar_clauses=similar_response
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# SEMANTIC SEARCH ENDPOINT
# ============================================================

@app.post("/search")
def search(request: ClauseRequest):

    clause = request.clause.strip()

    if not clause:
        raise HTTPException(
            status_code=400,
            detail="Search clause cannot be empty."
        )

    try:

        # Load ML artifacts only when an actual search is requested.
        ensure_models_loaded()

        results = semantic_search(
            clause,
            top_k=5
        )

        return {
            "query": clause,
            "results": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {str(e)}"
        )
