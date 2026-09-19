from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"

print(f"[Fashion Encoder] Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("[Fashion Encoder] Model loaded.")


def encode_text(text: str) -> list[float]:
    """
    Encode standardized fashion text into the shared semantic space.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()


def encode_item(item: dict) -> dict:
    """
    Encode one garment.
    """

    return {
        **item,
        "text_vec": encode_text(item["text_description"]),
        "embedding_model": MODEL_NAME
    }


def encode_outfit(outfit: dict) -> dict:
    """
    Encode an outfit-level description.
    """

    return {
        **outfit,
        "text_vec": encode_text(outfit["text_description"]),
        "embedding_model": MODEL_NAME
    }