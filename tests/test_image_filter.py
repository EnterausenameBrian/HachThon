"""
tests/test_image_filter.py

Integration test for image_filter.py.

Usage
-----
Post:
    python -m tests.test_image_filter post <instagram_url>

Reel:
    python -m tests.test_image_filter reel <instagram_url>

Examples:
    python -m tests.test_image_filter post \
        "https://www.instagram.com/p/XXXXXXXX/"

    python -m tests.test_image_filter reel \
        "https://www.instagram.com/reel/XXXXXXXX/"
"""

import sys
import json

from fashion_retrieval.post_parser import parse_post
from fashion_retrieval.reel_parser import parse_reel

from fashion_retrieval.image_filter import (
    filter_post,
    filter_reel,
)


# ============================================================
# Pretty print
# ============================================================

def print_result(result: dict):
    """
    Print filtered result without dumping large/unnecessary data.
    """

    print("\n")
    print("=" * 70)
    print("FILTER RESULT")
    print("=" * 70)

    print("source:   ", result.get("source"))
    print("type:     ", result.get("type"))
    print("url:      ", result.get("url"))
    print("shortcode:", result.get("shortcode"))

    items = result.get("items", [])

    print("\n")
    print(f"Kept {len(items)} image(s)")
    print("-" * 70)

    for index, item in enumerate(items):

        print(f"\n[{index}]")

        print(
            "image_path:",
            item.get("image_path")
        )

        if "timestamp" in item:
            print(
                "timestamp:",
                item.get("timestamp")
            )

        text = item.get("text")

        if text:
            # Avoid printing extremely long captions
            preview = text[:150]

            if len(text) > 150:
                preview += "..."

            print(
                "caption:",
                preview
            )


# ============================================================
# Test Post
# ============================================================

def test_post(url: str):

    print("\n")
    print("=" * 70)
    print("TESTING INSTAGRAM POST")
    print("=" * 70)

    print("\n[1/2] Parsing Instagram Post...")

    parsed = parse_post(url)

    original_count = len(
        parsed.get("items", [])
    )

    print(
        f"[Test] Parser returned "
        f"{original_count} image(s)."
    )

    print("\n[2/2] Filtering images...")

    filtered = filter_post(
        parsed
    )

    filtered_count = len(
        filtered.get("items", [])
    )

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Original images : {original_count}"
    )

    print(
        f"Filtered images : {filtered_count}"
    )

    print(
        f"Removed images  : "
        f"{original_count - filtered_count}"
    )

    print_result(filtered)

    return filtered


# ============================================================
# Test Reel
# ============================================================

def test_reel(
    url: str,
    batch_size: int = 10,
):

    print("\n")
    print("=" * 70)
    print("TESTING INSTAGRAM REEL")
    print("=" * 70)

    print("\n[1/2] Parsing Instagram Reel...")

    parsed = parse_reel(url)

    original_count = len(
        parsed.get("items", [])
    )

    print(
        f"[Test] Parser returned "
        f"{original_count} frame(s)."
    )

    print("\n[2/2] Filtering Reel frames...")

    filtered = filter_reel(parsed)

    filtered_count = len(
        filtered.get("items", [])
    )

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Original frames : {original_count}"
    )

    print(
        f"Filtered frames : {filtered_count}"
    )

    print(
        f"Removed frames  : "
        f"{original_count - filtered_count}"
    )

    print_result(filtered)

    return filtered


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) < 3:

        print(
            "\nUsage:\n"
            "\n"
            "Post:\n"
            "  python -m tests.test_image_filter "
            "post <instagram_url>\n"
            "\n"
            "Reel:\n"
            "  python -m tests.test_image_filter "
            "reel <instagram_url>\n"
        )

        sys.exit(1)

    content_type = (
        sys.argv[1]
        .strip()
        .lower()
    )

    url = sys.argv[2].strip()

    if content_type == "post":

        test_post(url)

    elif content_type == "reel":

        test_reel(url)

    else:

        raise ValueError(
            "First argument must be "
            "'post' or 'reel'."
        )


if __name__ == "__main__":
    main()