"""
test_fashion_encoder.py

Tests for fashion_encoder.py
"""

import math

from fashion_retrieval.fashion_encoder import (
    MODEL_NAME,
    encode_text,
    encode_item,
    encode_items,
)


# ============================================================
# Helper
# ============================================================

def vector_norm(
    vector: list[float],
) -> float:
    """
    Compute L2 norm of a vector.
    """

    return math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )


# ============================================================
# Test encode_text
# ============================================================

def test_encode_text():

    print(
        "\n========================================"
    )

    print(
        "TEST 1: encode_text"
    )

    print(
        "========================================"
    )

    text = (
        "A relaxed black oversized T-shirt "
        "with short sleeves, a crew neckline, "
        "and a smooth cotton-like appearance."
    )

    embedding = encode_text(
        text
    )

    print(
        f"Embedding dimension: "
        f"{len(embedding)}"
    )

    print(
        f"L2 norm: "
        f"{vector_norm(embedding):.6f}"
    )

    print(
        f"First 10 values: "
        f"{embedding[:10]}"
    )

    assert isinstance(
        embedding,
        list,
    )

    assert len(embedding) == 1024

    assert abs(
        vector_norm(embedding) - 1.0
    ) < 1e-5

    print(
        "[PASS] encode_text"
    )


# ============================================================
# Test encode_item
# ============================================================

def test_encode_item():

    print(
        "\n========================================"
    )

    print(
        "TEST 2: encode_item"
    )

    print(
        "========================================"
    )

    item = {
        "source":
            "instagram",

        "source_item_id":
            "test_001",

        "category":
            "pants",

        "text_description":
            (
                "Black wide-leg cargo pants with "
                "a relaxed silhouette, full length, "
                "utility pockets, and a smooth "
                "structured fabric appearance."
            ),

        "display_tags": [
            "Black",
            "Wide-leg",
            "Cargo",
        ],

        "outfit_tags": [
            "Casual",
            "Streetwear",
        ],

        "price_twd":
            None,
    }

    result = encode_item(
        item
    )

    assert (
        result["source"]
        ==
        item["source"]
    )

    assert (
        result["source_item_id"]
        ==
        item["source_item_id"]
    )

    assert (
        result["category"]
        ==
        item["category"]
    )

    assert (
        result["text_description"]
        ==
        item["text_description"]
    )

    assert (
        result["display_tags"]
        ==
        item["display_tags"]
    )

    assert (
        result["outfit_tags"]
        ==
        item["outfit_tags"]
    )

    assert (
        result["embedding_model"]
        ==
        MODEL_NAME
    )

    assert isinstance(
        result["embedding"],
        list,
    )

    assert (
        len(result["embedding"])
        ==
        1024
    )

    assert abs(
        vector_norm(
            result["embedding"]
        ) - 1.0
    ) < 1e-5

    # Old field must no longer exist
    assert "text_vec" not in result

    print(
        f"Category: "
        f"{result['category']}"
    )

    print(
        f"Embedding model: "
        f"{result['embedding_model']}"
    )

    print(
        f"Embedding dimension: "
        f"{len(result['embedding'])}"
    )

    print(
        "[PASS] encode_item"
    )


# ============================================================
# Test batch encoding
# ============================================================

def test_encode_items():

    print(
        "\n========================================"
    )

    print(
        "TEST 3: encode_items"
    )

    print(
        "========================================"
    )

    items = [
        {
            "source":
                "product",

            "source_item_id":
                "product_001",

            "category":
                "top",

            "text_description":
                (
                    "A fitted white sleeveless "
                    "tank top with a scoop neckline "
                    "and smooth athletic fabric."
                ),

            "price_twd":
                1200,
        },

        {
            "source":
                "product",

            "source_item_id":
                "product_002",

            "category":
                "pants",

            "text_description":
                (
                    "Black relaxed athletic shorts "
                    "with a high-rise waist and "
                    "lightweight smooth fabric."
                ),

            "price_twd":
                1500,
        },
    ]

    results = encode_items(
        items
    )

    assert len(results) == 2

    for result in results:

        assert "embedding" in result

        assert (
            "embedding_model"
            in result
        )

        assert (
            len(result["embedding"])
            ==
            1024
        )

        assert abs(
            vector_norm(
                result["embedding"]
            ) - 1.0
        ) < 1e-5

    print(
        f"Encoded items: "
        f"{len(results)}"
    )

    print(
        "[PASS] encode_items"
    )


# ============================================================
# Test invalid input
# ============================================================

def test_invalid_input():

    print(
        "\n========================================"
    )

    print(
        "TEST 4: invalid input"
    )

    print(
        "========================================"
    )

    try:

        encode_text("")

        raise AssertionError(
            "Empty text should raise ValueError."
        )

    except ValueError:

        pass

    try:

        encode_item(
            {
                "category": "top"
            }
        )

        raise AssertionError(
            "Missing text_description "
            "should raise ValueError."
        )

    except ValueError:

        pass

    print(
        "[PASS] invalid input handling"
    )


# ============================================================
# Main
# ============================================================

def main():

    test_encode_text()

    test_encode_item()

    test_encode_items()

    test_invalid_input()

    print(
        "\n========================================"
    )

    print(
        "ALL FASHION ENCODER TESTS PASSED"
    )

    print(
        "========================================\n"
    )


if __name__ == "__main__":
    main()