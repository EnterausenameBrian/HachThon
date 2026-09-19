from fashion_retrieval.reel_parser import parse_reel


URL = "https://www.instagram.com/reel/Dc3AKkcBPju/"


result = parse_reel(URL)


print("\n================================")
print("Reel Parser Result")
print("================================")

print("Source:")
print(result["source"])

print("\nType:")
print(result["type"])

print("\nShortCode:")
print(result["shortcode"])

print("\nVideo:")
print(result["video_path"])

print("\nNumber of frames:")
print(len(result["items"]))


for item in result["items"]:

    print(
        f'{item["timestamp"]:5.1f}s -> '
        f'{item["image_path"]}'
    )