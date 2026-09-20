import json
from pathlib import Path
from typing import Optional

import numpy as np
from psycopg2.extras import RealDictCursor

from fashion_retrieval.db_reader import get_connection
from fashion_retrieval.fashion_encoder import encode_text


# ============================================================
# Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STYLE_KB_PATH = (
    BASE_DIR
    / "style_kb_matched.jsonl"
)

# Number of final products returned
# for each category.
TOP_M = 3

# Final ranking weights.
USER_WEIGHT = 0.67
QUERY_WEIGHT = 0.33


# ============================================================
# Embedding utility
# ============================================================

def normalize_embedding(
    embedding,
) -> np.ndarray:
    """
    Convert an embedding into a normalized numpy vector.

    PostgreSQL may return the embedding as:

        "[0.1, 0.2, ...]"

    so string embeddings are parsed first.
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
    """
    Both vectors are assumed to be normalized.

    Therefore:

        cosine similarity = dot product
    """

    return float(
        np.dot(a, b)
    )


# ============================================================
# Load style candidates
# ============================================================

def load_style_candidates(
    style_name: str,
):
    """
    Read the Top-K candidate products generated
    by build_style_lookup.py.

    Returns:

        {
            "top": [...],
            "bottom": [...]
        }
    """

    if not STYLE_KB_PATH.exists():
        raise FileNotFoundError(
            f"Matched style KB not found: "
            f"{STYLE_KB_PATH}"
        )

    with open(
        STYLE_KB_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            if (
                record.get("style")
                != style_name
            ):
                continue

            return {
                "top": (
                    record["items"]
                    ["top"]
                    .get(
                        "matches",
                        [],
                    )
                ),
                "bottom": (
                    record["items"]
                    ["bottom"]
                    .get(
                        "matches",
                        [],
                    )
                ),
            }

    raise ValueError(
        f"Style not found: {style_name}"
    )


# ============================================================
# Load candidate products
# ============================================================

def load_candidate_products(
    product_ids: list,
):
    """
    Load embeddings for the candidate products.

    Product embeddings are already stored in
    PostgreSQL and are NOT recomputed here.
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

        try:
            embedding = (
                normalize_embedding(
                    row["embedding"]
                )
            )

        except ValueError:

            print(
                f"Warning: product "
                f"{row['product_id']} "
                f"has zero embedding."
            )

            continue

        products[
            row["product_id"]
        ] = {
            "product_id":
                row["product_id"],

            "category":
                row["category"],

            "embedding":
                embedding,
        }

    return products


# ============================================================
# Load user preference embeddings
# ============================================================

def load_user_preference_embeddings(
    user_id: str,
    category: str,
):
    """
    Load garment embeddings extracted from
    outfits that the user previously liked
    and sent to the LINE bot.

    Expected database schema:

        user_fashion_preferences

        user_id
        category
        embedding

    category should be:

        top
        bottom

    NOTE:
    Change the SQL here if the actual preference
    table/schema is different.
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

        try:
            embedding = (
                normalize_embedding(
                    row["embedding"]
                )
            )

        except ValueError:
            continue

        embeddings.append(
            embedding
        )

    return embeddings


# ============================================================
# S_user
# ============================================================

def calculate_user_score(
    product_embedding: np.ndarray,
    user_embeddings: list,
) -> Optional[float]:
    """
    S_user(x)

    = average similarity between the candidate
      product and the user's historical
      preference embeddings.

    Returns None if the user does not yet have
    preference data.
    """

    if not user_embeddings:
        return None

    similarities = [
        cosine_similarity(
            product_embedding,
            user_embedding,
        )
        for user_embedding
        in user_embeddings
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
    S_query(x)

    = cosine similarity between the candidate
      product and the current user query.
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
    If user preference exists:

        S_final =
            USER_WEIGHT * S_user
            +
            QUERY_WEIGHT * S_query

    Otherwise:

        S_final = S_query
    """

    if user_score is None:
        return query_score

    return (
        USER_WEIGHT
        * user_score
        +
        QUERY_WEIGHT
        * query_score
    )


# ============================================================
# Rank one category
# ============================================================

def rank_category(
    candidates: list,
    products: dict,
    user_embeddings: list,
    query_embedding: np.ndarray,
    top_m: int = TOP_M,
):
    """
    Re-rank the K style candidates.

    Returns the Top-M products.
    """

    ranked = []

    for candidate in candidates:

        product_id = (
            candidate["product_id"]
        )

        product = products.get(
            product_id
        )

        if product is None:

            print(
                f"Warning: product "
                f"{product_id} "
                f"not found in database."
            )

            continue

        product_embedding = (
            product["embedding"]
        )

        # ----------------------------------------
        # S_user
        # ----------------------------------------

        user_score = (
            calculate_user_score(
                product_embedding,
                user_embeddings,
            )
        )

        # ----------------------------------------
        # S_query
        # ----------------------------------------

        query_score = (
            calculate_query_score(
                product_embedding,
                query_embedding,
            )
        )

        # ----------------------------------------
        # S_final
        # ----------------------------------------

        final_score = (
            calculate_final_score(
                user_score,
                query_score,
            )
        )

        ranked.append({
            "product_id":
                product_id,

            # First-stage score.
            # Kept only for debugging.
            "style_score":
                candidate.get(
                    "score"
                ),

            "user_score":
                (
                    round(
                        user_score,
                        6,
                    )
                    if user_score
                    is not None
                    else None
                ),

            "query_score":
                round(
                    query_score,
                    6,
                ),

            "final_score":
                round(
                    final_score,
                    6,
                ),
        })

    ranked.sort(
        key=lambda x:
            x["final_score"],
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
    Second-stage ranking pipeline.

    Input:
        user_id
        style_name
        query_text

    Pipeline:

        style_name
            ↓
        style_kb_matched.jsonl
            ↓
        K top candidates
        K bottom candidates
            ↓
        S_user
        S_query
            ↓
        weighted ranking
            ↓
        Top-M top
        Top-M bottom
    """

    # --------------------------------------------------------
    # 1. Load first-stage candidates
    # --------------------------------------------------------

    candidates = (
        load_style_candidates(
            style_name
        )
    )

    # --------------------------------------------------------
    # 2. Collect candidate product IDs
    # --------------------------------------------------------

    product_ids = []

    for category in [
        "top",
        "bottom",
    ]:

        product_ids.extend([
            item["product_id"]
            for item
            in candidates[category]
        ])

    # Remove duplicates.
    product_ids = list(
        dict.fromkeys(
            product_ids
        )
    )

    # --------------------------------------------------------
    # 3. Load product embeddings
    # --------------------------------------------------------

    products = (
        load_candidate_products(
            product_ids
        )
    )

    # --------------------------------------------------------
    # 4. Encode current query
    # --------------------------------------------------------

    query_embedding = np.asarray(
        encode_text(
            query_text
        ),
        dtype=np.float32,
    )

    # Safety normalization.
    query_embedding = (
        normalize_embedding(
            query_embedding
        )
    )

    # --------------------------------------------------------
    # 5. Check embedding dimensions
    # --------------------------------------------------------

    if products:

        first_product = next(
            iter(
                products.values()
            )
        )

        product_dimension = (
            first_product[
                "embedding"
            ].shape[0]
        )

        query_dimension = (
            query_embedding.shape[0]
        )

        if (
            product_dimension
            != query_dimension
        ):

            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"product={product_dimension}, "
                f"query={query_dimension}"
            )

    # --------------------------------------------------------
    # 6. Load user preference embeddings
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
    # 7. Rank top candidates
    # --------------------------------------------------------

    top_results = (
        rank_category(
            candidates=
                candidates["top"],

            products=
                products,

            user_embeddings=
                user_top_embeddings,

            query_embedding=
                query_embedding,

            top_m=
                top_m,
        )
    )

    # --------------------------------------------------------
    # 8. Rank bottom candidates
    # --------------------------------------------------------

    bottom_results = (
        rank_category(
            candidates=
                candidates["bottom"],

            products=
                products,

            user_embeddings=
                user_bottom_embeddings,

            query_embedding=
                query_embedding,

            top_m=
                top_m,
        )
    )

    # --------------------------------------------------------
    # 9. Return result
    # --------------------------------------------------------

    return {
        "user_id":
            user_id,

        "style":
            style_name,

        "query":
            query_text,

        "weights": {
            "user":
                USER_WEIGHT,

            "query":
                QUERY_WEIGHT,
        },

        "top":
            top_results,

        "bottom":
            bottom_results,
    }


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":

    result = rank_products(
        user_id="test_user",

        style_name="old_money",

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