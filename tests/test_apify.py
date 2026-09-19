from fashion_retrieval.apify_client import fetch_instagram_post


URL = "https://www.instagram.com/reel/Dc3AKkcBPju/"


result = fetch_instagram_post(URL)


print("\n==============================")
print("Apify Result")
print("==============================")

print("Type:")
print(result.get("type"))

print("\nShortCode:")
print(result.get("shortCode"))

print("\nCaption:")
print(result.get("caption"))

print("\nImages:")
print(result.get("images"))

print("\nVideo URL:")
print("Available" if result.get("videoUrl") else "Not available")

print("\nVideo Duration:")
print(result.get("videoDuration"))