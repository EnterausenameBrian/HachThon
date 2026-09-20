import numpy as np

from build_style_lookup import (
    build_product_index,
    find_top_k,
    match_styles,
)


# ============================================================
# Mock data
# ============================================================

def make_mock_products():
    """
    Create fake products with simple normalized embeddings.

    The vectors are intentionally simple so that
    expected similarity rankings are obvious.
    """

    return [
        {
            "product_id": "top_001",
            "category": "top",
            "embedding": np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        },
        {
            "product_id": "top_002",
            "category": "top",
            "embedding": np.array(
                [0.8, 0.6, 0.0],
                dtype=np.float32,
            ),
        },
        {
            "product_id": "top_003",
            "category": "top",
            "embedding": np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            ),
        },
        {
            "product_id": "bottom_001",
            "category": "bottom",
            "embedding": np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        },
        {
            "product_id": "bottom_002",
            "category": "bottom",
            "embedding": np.array(
                [0.0, 0.6, 0.8],
                dtype=np.float32,
            ),
        },
    ]


# ============================================================
# Test product index
# ============================================================

def test_build_product_index():

    products = make_mock_products()

    index = build_product_index(products)

    assert "top" in index
    assert "bottom" in index

    assert len(index["top"]["products"]) == 3
    assert len(index["bottom"]["products"]) == 2

    assert index["top"]["embeddings"].shape == (3, 3)
    assert index["bottom"]["embeddings"].shape == (2, 3)


# ============================================================
# Test category separation
# ============================================================

def test_category_separation():

    products = make_mock_products()

    index = build_product_index(products)

    top_ids = [
        product["product_id"]
        for product in index["top"]["products"]
    ]

    bottom_ids = [
        product["product_id"]
        for product in index["bottom"]["products"]
    ]

    assert "top_001" in top_ids
    assert "top_002" in top_ids
    assert "top_003" in top_ids

    assert "bottom_001" not in top_ids
    assert "bottom_002" not in top_ids

    assert "bottom_001" in bottom_ids
    assert "bottom_002" in bottom_ids


# ============================================================
# Test Top-K ranking
# ============================================================

def test_find_top_k(monkeypatch):

    products = make_mock_products()

    index = build_product_index(products)

    # Fake query embedding:
    #
    # [1, 0, 0]
    #
    # Therefore:
    #
    # top_001 score = 1.0
    # top_002 score = 0.8
    # top_003 score = 0.0

    def fake_encode_text(text):

        return np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        "build_style_lookup.encode_text",
        fake_encode_text,
    )

    matches = find_top_k(
        description="test top",
        category="top",
        index=index,
        k=2,
    )

    assert len(matches) == 2

    assert matches[0]["product_id"] == "top_001"
    assert matches[1]["product_id"] == "top_002"

    assert np.isclose(
        matches[0]["score"],
        1.0,
    )

    assert np.isclose(
        matches[1]["score"],
        0.8,
    )


# ============================================================
# Test K larger than database
# ============================================================

def test_top_k_larger_than_database(monkeypatch):

    products = make_mock_products()

    index = build_product_index(products)

    def fake_encode_text(text):

        return np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        "build_style_lookup.encode_text",
        fake_encode_text,
    )

    matches = find_top_k(
        description="test",
        category="top",
        index=index,
        k=100,
    )

    print("\nTop-K matches:")
    for match in matches:
        print(
            f"  {match['product_id']}: "
            f"{match['score']:.6f}"
        )

    # There are only 3 top products.
    assert len(matches) == 3


# ============================================================
# Test complete style matching
# ============================================================

def test_match_styles(monkeypatch):

    products = make_mock_products()

    index = build_product_index(products)

    def fake_encode_text(text):

        if "shirt" in text:
            return np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )

        return np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        "build_style_lookup.encode_text",
        fake_encode_text,
    )

    records = [
        {
            "style": "test_style",
            "style_zh": "測試風格",
            "items": {
                "top": {
                    "description": "white shirt",
                },
                "bottom": {
                    "description": "black pants",
                },
            },
        }
    ]

    result = match_styles(
        records,
        index,
    )

    top_matches = (
        result[0]["items"]["top"]["matches"]
    )

    bottom_matches = (
        result[0]["items"]["bottom"]["matches"]
    )

    # Best top should be top_001.
    assert (
        top_matches[0]["product_id"]
        == "top_001"
    )

    # Best bottom should be bottom_001.
    assert (
        bottom_matches[0]["product_id"]
        == "bottom_001"
    )

    # Make sure categories never get mixed.
    assert all(
        match["product_id"].startswith("top_")
        for match in top_matches
    )

    assert all(
        match["product_id"].startswith("bottom_")
        for match in bottom_matches
    )

    import json

    print("\nMatched style result:")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )