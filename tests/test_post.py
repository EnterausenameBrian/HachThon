from fashion_retrieval.post_parser import parse_post


# 換成你要測試的 Instagram Post / Carousel
URL = "https://www.instagram.com/p/DdbHdbAEtnJ/"


result = parse_post(URL)


print("\n================================")
print("Post Parser Result")
print("================================")

print("Source:")
print(result["source"])

print("\nType:")
print(result["type"])

print("\nShortCode:")
print(result["shortcode"])

print("\nNumber of items:")
print(len(result["items"]))


for i, item in enumerate(
    result["items"],
    start=1
):

    print(f"\n--- Item {i} ---")

    print("Image:")
    print(item["image_path"])

    print("Text:")
    print(item["text"])