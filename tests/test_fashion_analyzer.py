'''
# 測單張本機圖片

python -m tests.test_fashion_analyzer image path/to/image.jpg

# 測 IG Post 整條 parser → filter → analyzer

python -m tests.test_fashion_analyzer post "IG_URL"

# 測 Reel

python -m tests.test_fashion_analyzer reel "IG_URL"

# 測 DB 商品

python -m tests.test_fashion_analyzer product 3
'''

"""
tests/test_fashion_analyzer.py

Integration tests for fashion_analyzer.py

Usage
-----

Test one local image:

    python -m tests.test_fashion_analyzer \
        image path/to/image.jpg

Test Instagram Post:

    python -m tests.test_fashion_analyzer \
        post "https://www.instagram.com/p/..."

Test Instagram Reel:

    python -m tests.test_fashion_analyzer \
        reel "https://www.instagram.com/reel/..."

Test products from database:

    python -m tests.test_fashion_analyzer \
        product 3
"""

import sys
import json

from fashion_retrieval.fashion_analyzer import (
    analyze_image,
    analyze_post,
    analyze_reel,
    analyze_product,
)


# ============================================================
# Utilities
# ============================================================

def print_json(data):
    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )


def print_separator(title):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Test single image
# ============================================================

def test_image(image_path: str):

    print_separator(
        "TEST: SINGLE IMAGE"
    )

    print(f"Image: {image_path}")

    result = analyze_image(
        image=image_path,
        generate_tags=True,
    )

    print_separator(
        "RESULT"
    )

    print_json(result)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    assert "garments" in result

    assert isinstance(
        result["garments"],
        list,
    )

    for garment in result["garments"]:

        assert garment["category"] in {
            "top",
            "pants",
        }

        assert isinstance(
            garment["text_description"],
            str,
        )

        assert (
            garment["text_description"].strip()
        )

    print("\n[PASS] Single image test passed.")


# ============================================================
# Test Instagram Post
# ============================================================

def test_post(url: str):

    print_separator(
        "TEST: INSTAGRAM POST"
    )

    from fashion_retrieval.post_parser import (
        parse_post,
    )

    from fashion_retrieval.image_filter import (
        filter_post,
    )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    print("\n[1/3] Parsing Instagram Post...")

    parsed = parse_post(url)

    print(
        f"Parsed images: "
        f"{len(parsed.get('items', []))}"
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    print("\n[2/3] Filtering images...")

    filtered = filter_post(parsed)

    print(
        f"Filtered images: "
        f"{len(filtered.get('items', []))}"
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    print("\n[3/3] Analyzing garments...")

    result = analyze_post(
        filtered
    )

    print_separator(
        "POST ANALYSIS RESULT"
    )

    print_json(result)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    assert result["source"] == "instagram"
    assert result["type"] == "post"

    assert isinstance(
        result["samples"],
        list,
    )

    for sample in result["samples"]:

        assert sample["category"] in {
            "top",
            "pants",
        }

        assert sample[
            "text_description"
        ].strip()

        assert isinstance(
            sample.get(
                "display_tags",
                []
            ),
            list,
        )

    print(
        "\n[PASS] Instagram Post test passed."
    )


# ============================================================
# Test Instagram Reel
# ============================================================

def test_reel(url: str):

    print_separator(
        "TEST: INSTAGRAM REEL"
    )

    from fashion_retrieval.reel_parser import (
        parse_reel,
    )

    from fashion_retrieval.image_filter import (
        filter_reel,
    )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    print("\n[1/3] Parsing Reel...")

    parsed = parse_reel(url)

    print(
        f"Original frames: "
        f"{len(parsed.get('items', []))}"
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    print("\n[2/3] Filtering Reel frames...")

    filtered = filter_reel(
        parsed
    )

    print(
        f"Filtered frames: "
        f"{len(filtered.get('items', []))}"
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    print("\n[3/3] Analyzing garments...")

    result = analyze_reel(
        filtered
    )

    print_separator(
        "REEL ANALYSIS RESULT"
    )

    print_json(result)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    assert result["source"] == "instagram"
    assert result["type"] == "reel"

    assert isinstance(
        result["samples"],
        list,
    )

    for sample in result["samples"]:

        assert sample["category"] in {
            "top",
            "pants",
        }

        assert sample[
            "text_description"
        ].strip()

        assert "timestamp" in sample

    print(
        "\n[PASS] Instagram Reel test passed."
    )


# ============================================================
# Test Product Database
# ============================================================

def test_products(limit: int = 3):

    print_separator(
        "TEST: PRODUCT DATABASE"
    )

    from fashion_retrieval.db_reader import (
        fetch_products,
    )

    print(
        f"\nFetching {limit} product(s)..."
    )

    products = fetch_products(
        limit=limit
    )

    if not products:
        raise RuntimeError(
            "No products returned from database."
        )

    results = []

    for index, product in enumerate(
        products
    ):

        print_separator(
            f"PRODUCT {index + 1}/{len(products)}"
        )

        print(
            f"ID: {product['product_id']}"
        )

        print(
            f"Title: {product['title']}"
        )

        print(
            f"Category: {product['category']}"
        )

        print(
            f"Price: {product['price_twd']}"
        )

        result = analyze_product(
            product
        )

        results.append(result)

        print("\nAnalysis:")

        print_json(result)

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        assert result["category"] in {
            "top",
            "pants",
        }

        assert (
            result["item_id"]
            == str(product["product_id"])
        )

        if result["text_description"] is not None:

            assert isinstance(
                result["text_description"],
                str,
            )

            assert (
                result[
                    "text_description"
                ].strip()
            )

    print_separator(
        "ALL PRODUCT RESULTS"
    )

    print_json(results)

    print(
        f"\n[PASS] "
        f"{len(results)} product test(s) passed."
    )


# ============================================================
# CLI
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            """
Usage:

Single image:
    python -m tests.test_fashion_analyzer \
        image path/to/image.jpg

Instagram Post:
    python -m tests.test_fashion_analyzer \
        post "https://www.instagram.com/p/..."

Instagram Reel:
    python -m tests.test_fashion_analyzer \
        reel "https://www.instagram.com/reel/..."

Product DB:
    python -m tests.test_fashion_analyzer \
        product 3
"""
        )

        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "image":

        if len(sys.argv) < 3:
            raise ValueError(
                "Please provide an image path."
            )

        test_image(
            sys.argv[2]
        )

    elif mode == "post":

        if len(sys.argv) < 3:
            raise ValueError(
                "Please provide an Instagram Post URL."
            )

        test_post(
            sys.argv[2]
        )

    elif mode == "reel":

        if len(sys.argv) < 3:
            raise ValueError(
                "Please provide an Instagram Reel URL."
            )

        test_reel(
            sys.argv[2]
        )

    elif mode == "product":

        limit = 3

        if len(sys.argv) >= 3:
            limit = int(
                sys.argv[2]
            )

        test_products(
            limit=limit
        )

    else:

        raise ValueError(
            f"Unknown test mode: {mode}"
        )


if __name__ == "__main__":
    main()