import json
from pathlib import Path
from typing import Optional

import numpy as np
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

from fashion_retrieval.db_reader import get_connection


# ============================================================
# Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STYLE_KB_PATH = BASE_DIR / "style_kb_matched.jsonl"

EMBEDDING_MODEL = "BAAI/bge-m3"

# Number of final products returned for each category
TOP_M = 3

# Final ranking weights
USER_WEIGHT = 0.67
QUERY_WEIGHT = 0.33


# ============================================================
# Embedding model
# ============================================================

_model = None


def get_embedding_model():
    """
    Lazy-load BGE-M3.
    """
    global _model

    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")

        _model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    return _model


def encode_text(text: str) -> np.ndarray:
    """
    Encode text and L2-normalize the result.
    """

    model = get_embedding_model()

    embedding = model.encode(
        text,
        normalize_embeddings=True,
    )

    return np.asarray(
        embedding,
        dtype=np.float32,
    )


# ============================================================
# Utility
# ============================================================

def normalize_embedding(
    embedding,
) -> np.ndarray:
    """
    Convert an embedding into a normalized numpy vector.

    PostgreSQL may return embedding as a JSON string.
    """

    if isinstance(embedding, str):
        embedding = json.loads(
            embedding
        )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        raise ValueError(
            "Zero-length embedding encountered."
        )

    return embedding / norm

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(np.dot(a, b))


# ============================================================
# Load style candidates
# ============================================================

def load_style_candidates(
    style_name: str,
):
    """
    Find one style record from style_kb_matched.jsonl.

    Returns:
        {
            "top": [...],
            "bottom": [...]
        }
    """

    with open(
        STYLE_KB_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if record.get("style") != style_name:
                continue

            return {
                "top": record["items"]["top"].get(
                    "matches",
                    [],
                ),
                "bottom": record["items"]["bottom"].get(
                    "matches",
                    [],
                ),
            }

    raise ValueError(
        f"Style not found: {style_name}"
    )


# ============================================================
# Product embeddings
# ============================================================

def load_candidate_products(
    product_ids: list,
):
    """
    Load embeddings only for the K candidate products.
    """

    if not product_ids:
        return {}

    query = """
        SELECT
            product_id,
            category,
            embedding
        FROM products
        WHERE product_id = ANY(%s)
          AND embedding IS NOT NULL;
    """

    with get_connection() as conn:
        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            cursor.execute(
                query,
                (product_ids,),
            )

            rows = cursor.fetchall()

    products = {}

    for row in rows:

        products[row["product_id"]] = {
            "product_id": row["product_id"],
            "category": row["category"],
            "embedding": normalize_embedding(
                row["embedding"]
            ),
        }

    return products


# ============================================================
# User preference embeddings
# ============================================================

def load_user_preference_embeddings(
    user_id: str,
    category: str,
):
    """
    Load garment embeddings extracted from Instagram
    outfits that the user sent to the LINE bot.

    IMPORTANT:
    The exact table name / schema may need to be changed
    once the IG database is finalized.

    Expected database fields:
        user_id
        category
        embedding
    """

    query = """
        SELECT
            embedding
        FROM user_fashion_preferences
        WHERE user_id = %s
          AND category = %s
          AND embedding IS NOT NULL;
    """

    with get_connection() as conn:
        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            cursor.execute(
                query,
                (
                    user_id,
                    category,
                ),
            )

            rows = cursor.fetchall()

    embeddings = []

    for row in rows:

        embeddings.append(
            normalize_embedding(
                row["embedding"]
            )
        )

    return embeddings


# ============================================================
# S_user
# ============================================================

def calculate_user_score(
    product_embedding: np.ndarray,
    user_embeddings: list[np.ndarray],
) -> Optional[float]:
    """
    S_user(x) =
        average cosine similarity between the candidate
        product and all user preference embeddings.

    Returns None if the user has no preference history.
    """

    if not user_embeddings:
        return None

    similarities = [
        cosine_similarity(
            product_embedding,
            user_embedding,
        )
        for user_embedding in user_embeddings
    ]

    return float(
        np.mean(similarities)
    )


# ============================================================
# S_query
# ============================================================

def calculate_query_score(
    product_embedding: np.ndarray,
    query_embedding: np.ndarray,
) -> float:
    """
    S_query(x) =
        cosine similarity between product and current query.
    """

    return cosine_similarity(
        product_embedding,
        query_embedding,
    )


# ============================================================
# Final score
# ============================================================

def calculate_final_score(
    user_score: Optional[float],
    query_score: float,
) -> float:
    """
    With user preference:

        S_final =
            0.6 * S_user
            + 0.4 * S_query

    Without user preference:

        S_final = S_query
    """

    if user_score is None:
        return query_score

    return (
        USER_WEIGHT * user_score
        + QUERY_WEIGHT * query_score
    )


# ============================================================
# Rank one category
# ============================================================

def rank_category(
    candidates: list,
    products: dict,
    user_embeddings: list[np.ndarray],
    query_embedding: np.ndarray,
    top_m: int = TOP_M,
):
    """
    Re-rank the K style candidates and return Top-M.
    """

    ranked = []

    for candidate in candidates:

        product_id = candidate["product_id"]

        product = products.get(product_id)

        if product is None:
            print(
                f"Warning: product {product_id} "
                "not found in database."
            )
            continue

        product_embedding = (
            product["embedding"]
        )

        # ----------------------------------------
        # S_user
        # ----------------------------------------

        user_score = calculate_user_score(
            product_embedding,
            user_embeddings,
        )

        # ----------------------------------------
        # S_query
        # ----------------------------------------

        query_score = calculate_query_score(
            product_embedding,
            query_embedding,
        )

        # ----------------------------------------
        # S_final
        # ----------------------------------------

        final_score = calculate_final_score(
            user_score,
            query_score,
        )

        ranked.append({
            "product_id": product_id,

            # Keep the lookup score for debugging.
            "style_score": candidate.get(
                "score"
            ),

            "user_score": (
                round(user_score, 6)
                if user_score is not None
                else None
            ),

            "query_score": round(
                query_score,
                6,
            ),

            "final_score": round(
                final_score,
                6,
            ),
        })

    ranked.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    return ranked[:top_m]


# ============================================================
# Main recommendation function
# ============================================================

def rank_products(
    user_id: str,
    style_name: str,
    query_text: str,
    top_m: int = TOP_M,
):
    """
    Complete second-stage ranking.

    Pipeline:

        Style
          ↓
        K candidates
          ↓
        S_user + S_query
          ↓
        weighted ranking
          ↓
        Top-M tops + Top-M bottoms
    """

    # --------------------------------------------------------
    # 1. Load K candidates generated by style lookup
    # --------------------------------------------------------

    candidates = load_style_candidates(
        style_name
    )

    # --------------------------------------------------------
    # 2. Collect candidate product IDs
    # --------------------------------------------------------

    product_ids = []

    for category in ["top", "bottom"]:

        product_ids.extend([
            item["product_id"]
            for item in candidates[category]
        ])

    # --------------------------------------------------------
    # 3. Load candidate product embeddings
    # --------------------------------------------------------

    products = load_candidate_products(
        product_ids
    )

    # --------------------------------------------------------
    # 4. Encode current user query
    # --------------------------------------------------------

    query_embedding = encode_text(
        query_text
    )

    # --------------------------------------------------------
    # 5. Load user's long-term preferences
    # --------------------------------------------------------

    user_top_embeddings = (
        load_user_preference_embeddings(
            user_id=user_id,
            category="top",
        )
    )

    user_bottom_embeddings = (
        load_user_preference_embeddings(
            user_id=user_id,
            category="bottom",
        )
    )

    # --------------------------------------------------------
    # 6. Rank tops
    # --------------------------------------------------------

    top_results = rank_category(
        candidates=candidates["top"],
        products=products,
        user_embeddings=user_top_embeddings,
        query_embedding=query_embedding,
        top_m=top_m,
    )

    # --------------------------------------------------------
    # 7. Rank bottoms
    # --------------------------------------------------------

    bottom_results = rank_category(
        candidates=candidates["bottom"],
        products=products,
        user_embeddings=user_bottom_embeddings,
        query_embedding=query_embedding,
        top_m=top_m,
    )

    # --------------------------------------------------------
    # 8. Return final result
    # --------------------------------------------------------

    return {
        "user_id": user_id,
        "style": style_name,
        "query": query_text,

        "weights": {
            "user": USER_WEIGHT,
            "query": QUERY_WEIGHT,
        },

        "top": top_results,
        "bottom": bottom_results,
    }

# ============================================================
# Test manually
# ============================================================

if __name__ == "__main__":

    result = rank_products(
        user_id="test_user",
        style_name="old_money",

        # This should be the semantic description
        # produced by the LLM.
        query_text=(
            "A clean and slightly formal outfit "
            "for a casual date in warm weather."
        ),

        top_m=3,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )