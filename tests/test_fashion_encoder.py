import numpy as np

from fashion_retrieval.fashion_encoder import (
    encode_image,
    encode_text
)


# ============================================================
# Test data
# ============================================================

IMAGE_PATH = (
    "outputs/posts/"
    "DdbHdbAEtnJ/"
    "image_01.jpg"
)

TEXT = (
    "A casual outfit with an oversized top "
    "and dark wide-leg pants."
)


# ============================================================
# Image embedding
# ============================================================

print("\n================================")
print("Image Embedding")
print("================================")


image_embedding = encode_image(
    IMAGE_PATH
)


print(
    "Dimension:",
    len(image_embedding)
)

print(
    "First 10 values:"
)

print(
    image_embedding[:10]
)


# ============================================================
# Text embedding
# ============================================================

print("\n================================")
print("Text Embedding")
print("================================")


text_embedding = encode_text(
    TEXT
)


print(
    "Dimension:",
    len(text_embedding)
)

print(
    "First 10 values:"
)

print(
    text_embedding[:10]
)


# ============================================================
# Cosine similarity
# ============================================================

image_vector = np.array(
    image_embedding
)

text_vector = np.array(
    text_embedding
)


similarity = np.dot(
    image_vector,
    text_vector
)


print("\n================================")
print("Image / Text Similarity")
print("================================")


print(
    "Cosine similarity:",
    similarity
)