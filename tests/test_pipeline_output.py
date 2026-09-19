"""
Test final fashion item output from:

1. Product DB
2. Instagram Post
3. Instagram Reel

The final item should be ready to pass into:

    DB_write(item)
"""

import argparse

from fashion_retrieval.db_reader import fetch_products
from fashion_retrieval.post_parser import parse_post
from fashion_retrieval.reel_parser import parse_reel
from fashion_retrieval.image_filter import filter_post, filter_reel
from fashion_retrieval.fashion_analyzer import (
    analyze_post,
    analyze_reel,
    analyze_product,
)
from fashion_retrieval.fashion_formatter import (
    build_product_item,
    build_instagram_item,
)


# ============================================================
# Display
# ============================================================

def print_item(item: dict):

    print("\n========================================")
    print("FINAL FASHION ITEM")
    print("========================================")

    for key, value in item.items():

        # Don't print thousands of binary bytes
        if key == "image_data":
            print(
                f"{key}: <binary {len(value)} bytes>"
            )

        # Don't print all 1024 embedding values
        elif key == "embedding":
            print(
                f"{key}: <vector dim={len(value)}>"
            )

        else:
            print(f"{key}: {value}")

    print("========================================")

    print(
        "image_data type:",
        type(item["image_data"]),
    )

    print(
        "embedding dimension:",
        len(item["embedding"]),
    )


# ============================================================
# Product DB
# ============================================================

def test_product():

    print("\n[TEST] Product DB")

    products = fetch_products(limit=1)

    if not products:
        raise RuntimeError(
            "No product found in database."
        )

    product = products[0]

    print(
        "Product:",
        product["product_id"],
        product["title"],
    )

    # Claude
    analysis = analyze_product(product)

    print(
        "[OK] Claude analysis completed."
    )

    # Formatter + BGE
    item = build_product_item(
        product,
        analysis,
    )

    print_item(item)

    return item


# ============================================================
# Instagram Post
# ============================================================

def test_post(url: str):

    print("\n[TEST] Instagram Post")

    # Parse
    post = parse_post(url)

    print(
        f"[OK] Parsed "
        f"{len(post['items'])} images."
    )

    # Filter
    post = filter_post(post)

    print(
        f"[OK] Filtered to "
        f"{len(post['items'])} images."
    )

    # Claude
    result = analyze_post(post)

    samples = result["samples"]

    if not samples:
        raise RuntimeError(
            "No garments detected."
        )

    print(
        f"[OK] Detected "
        f"{len(samples)} garments."
    )

    # For now only test the first garment
    garment = samples[0]

    item = build_instagram_item(
        garment,

        instagram_url=url,
        instagram_type="post",

        shortcode=post["shortcode"],

        source_item_id=(
            f"{post['shortcode']}_"
            f"0_"
            f"{garment['category']}"
        ),
    )

    print_item(item)

    return item


# ============================================================
# Instagram Reel
# ============================================================

def test_reel(url: str):

    print("\n[TEST] Instagram Reel")

    # Parse
    reel = parse_reel(url)

    print(
        f"[OK] Extracted "
        f"{len(reel['items'])} frames."
    )

    # Filter + dHash
    reel = filter_reel(reel)

    print(
        f"[OK] Filtered to "
        f"{len(reel['items'])} frames."
    )

    # Claude
    result = analyze_reel(reel)

    samples = result["samples"]

    if not samples:
        raise RuntimeError(
            "No garments detected."
        )

    print(
        f"[OK] Detected "
        f"{len(samples)} garments."
    )

    # For now only test first garment
    garment = samples[0]

    timestamp = garment.get(
        "timestamp",
        0,
    )

    item = build_instagram_item(
        garment,

        instagram_url=url,
        instagram_type="reel",

        shortcode=reel["shortcode"],

        source_item_id=(
            f"{reel['shortcode']}_"
            f"{timestamp}_"
            f"{garment['category']}"
        ),
    )

    print_item(item)

    return item


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=[
            "product",
            "post",
            "reel",
        ],
    )

    parser.add_argument(
        "url",
        nargs="?",
    )

    args = parser.parse_args()

    if args.mode == "product":

        test_product()

    elif args.mode == "post":

        if not args.url:
            raise ValueError(
                "Instagram Post URL required."
            )

        test_post(
            args.url
        )

    elif args.mode == "reel":

        if not args.url:
            raise ValueError(
                "Instagram Reel URL required."
            )

        test_reel(
            args.url
        )


if __name__ == "__main__":
    main()