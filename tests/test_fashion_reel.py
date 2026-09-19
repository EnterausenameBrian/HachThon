from fashion_retrieval.reel_parser import parse_reel
from fashion_retrieval.fashion_analyzer import analyze_reel


# ============================================================
# Test Reel
# ============================================================

URL = "https://www.instagram.com/reel/Dc3AKkcBPju/"


# ============================================================
# 1. Parse Reel
# ============================================================

print("\n================================")
print("1. Parse Reel")
print("================================")

parsed = parse_reel(URL)


print("\nReel information:")

print("ShortCode:")
print(parsed["shortcode"])

print("\nVideo:")
print(parsed["video_path"])

print("\nNumber of extracted frames:")
print(len(parsed["items"]))


# Show extracted frames

print("\nExtracted frames:")

for index, item in enumerate(
    parsed["items"]
):

    print(
        f"{index:02d} | "
        f"{item['timestamp']:5.1f}s | "
        f"{item['image_path']}"
    )


# ============================================================
# 2. Analyze Fashion
# ============================================================

print("\n================================")
print("2. Analyze Reel Fashion")
print("================================")

result = analyze_reel(
    parsed,
    batch_size=10
)


# ============================================================
# 3. Show Results
# ============================================================

print("\n================================")
print("Fashion Analysis Result")
print("================================")


print(
    f"\nNumber of detected outfits: "
    f"{len(result['samples'])}"
)


for index, sample in enumerate(
    result["samples"],
    start=1
):

    print(
        f"\n--------------------------------"
    )

    print(
        f"Outfit {index}"
    )

    print(
        f"--------------------------------"
    )

    print("\nTimestamp:")

    print(
        f"{sample['timestamp']} s"
    )

    print("\nRepresentative image:")

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