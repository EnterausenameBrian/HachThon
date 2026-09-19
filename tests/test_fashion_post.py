from fashion_retrieval.post_parser import parse_post
from fashion_retrieval.fashion_analyzer import analyze_post


URL = "https://www.instagram.com/p/DdbHdbAEtnJ/"


# ============================================================
# Parse
# ============================================================

print("\n===== Parse Post =====")

parsed = parse_post(URL)

print(
    f"\nDownloaded images: "
    f"{len(parsed['items'])}"
)


# ============================================================
# Analyze
# ============================================================

print("\n===== Analyze Fashion =====")

result = analyze_post(parsed)


# ============================================================
# Result
# ============================================================

print("\n================================")
print("Fashion Analysis Result")
print("================================")

print(
    f"\nNumber of fashion samples: "
    f"{len(result['samples'])}"
)


for index, sample in enumerate(
    result["samples"],
    start=1
):

    print(
        f"\n--- Fashion Sample "
        f"{index} ---"
    )

    print("\nImage:")
    print(
        sample["image_path"]
    )

    print("\nDescription:")
    print(
        sample["description"]
    )

    print("\nGarments:")

    for garment in sample[
        "garments"
    ]:

        print(
            f"- {garment}"
        )