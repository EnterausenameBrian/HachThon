# Fashion Retrieval Pipeline

A fashion understanding and retrieval pipeline developed for a personalized outfit recommendation system.

The system converts fashion content from Instagram and e-commerce product databases into standardized semantic garment representations that can later be used for fashion retrieval, preference learning, and outfit recommendation.

## Overview

The current pipeline supports two main data sources:

1. **Instagram Posts / Reels**
2. **E-commerce Product Database**

Fashion images are analyzed using a Vision LLM to generate standardized garment descriptions. These descriptions are then encoded into semantic embeddings using BGE-M3.

```text
Instagram URL
      │
      ▼
    Apify
      │
      ├── Post ──► Post Parser
      │
      └── Reel ──► Reel Parser
                        │
                        ▼
                  Image Filter
               (Claude + dHash)
                        │
                        ▼
                Fashion Analyzer
                    (Claude)
                        │
                        ▼
              Garment Description
                        │
                        ▼
                BGE-M3 Encoder
                        │
                        ▼
               Semantic Embedding
```

For e-commerce products:

```text
Product Database
      │
      ▼
Product Image + Metadata
      │
      ▼
Fashion Analyzer
    (Claude)
      │
      ▼
Garment Description
      │
      ▼
BGE-M3 Encoder
      │
      ▼
Semantic Embedding
```

---

## Current MVP Scope

The current MVP supports two garment categories:

### `top`

Includes:

- T-shirts
- Shirts
- Blouses
- Sweaters
- Hoodies
- Sweatshirts
- Tank tops
- Crop tops
- Similar upper-body garments

### `pants`

Includes both long pants and shorts:

- Trousers
- Jeans
- Cargo pants
- Sweatpants
- Shorts
- Athletic shorts
- Similar two-leg lower-body garments

Other garment categories are currently ignored.

---

## Semantic Representation

Instead of directly embedding raw images, the system first converts each garment into a detailed natural-language description.

Example:

```json
{
  "category": "pants",
  "text_description": "Black wide-leg cargo pants with a relaxed silhouette, full length, multiple utility pockets, and a smooth structured fabric appearance."
}
```

The description is then encoded using:

```text
BAAI/bge-m3
```

This produces a normalized semantic embedding that can be used for downstream retrieval and compatibility modeling.

### Why text-based embeddings?

Using a standardized text representation creates a shared semantic space for:

- Instagram fashion content
- E-commerce products
- User queries
- Fashion compatibility examples

The Vision LLM acts as the visual-semantic information extraction layer, while BGE-M3 provides the common embedding space.

---

## Instagram Pipeline

### 1. Instagram Data Collection

Instagram URLs are processed using Apify.

```text
Instagram URL
      ↓
apify_client.py
      ↓
Instagram metadata
```

---

### 2. Post Parsing

`post_parser.py` processes Instagram posts and carousels.

It:

- extracts post images
- downloads images locally
- preserves the Instagram caption as supporting context

Example output:

```json
{
  "source": "instagram",
  "type": "post",
  "url": "...",
  "shortcode": "...",
  "items": [
    {
      "image_path": "...",
      "text": "Instagram caption"
    }
  ]
}
```

---

### 3. Reel Parsing

`reel_parser.py` processes Instagram Reels.

It:

- downloads the Reel video
- extracts frames at fixed time intervals
- preserves timestamps
- preserves the Reel caption as supporting context

Example:

```json
{
  "image_path": "...",
  "timestamp": 6.0,
  "text": "Instagram caption"
}
```

---

### 4. Image Filtering

`image_filter.py` removes images that do not provide useful fashion information.

Claude is used to reject:

- blurry images
- transition frames
- text-only slides
- advertisements
- unrelated images
- images without a useful top or pants garment

For Reels, local **difference hashing (dHash)** is additionally used to remove near-identical consecutive frames.

```text
Reel Frames
     ↓
Semantic Filtering
   (Claude)
     ↓
Near-Duplicate Filtering
    (dHash)
     ↓
Useful Frames
```

---

### 5. Fashion Analysis

`fashion_analyzer.py` uses Claude Vision to convert images into standardized garment descriptions.

Each supported garment is represented independently.

Example:

```json
{
  "garments": [
    {
      "category": "top",
      "text_description": "A fitted dark brown ribbed tank top with a sleeveless cut and scoop neckline.",
      "display_tags": [
        "Dark Brown",
        "Tank Top",
        "Ribbed",
        "Fitted"
      ]
    },
    {
      "category": "pants",
      "text_description": "Black relaxed cargo shorts with a loose silhouette and utility pocket details.",
      "display_tags": [
        "Black",
        "Cargo",
        "Relaxed-fit",
        "Shorts"
      ]
    }
  ],
  "outfit_tags": [
    "Casual",
    "Minimal"
  ]
}
```

For Instagram content, lightweight tags are also generated for UI purposes.

These tags are **not used as the semantic embedding input**.

---

## Product Database Pipeline

Product images can also be analyzed using the same semantic representation.

The product database provides information such as:

```text
product_id
title
price_twd
product_url
image_data
image_mime
category
```

Database categories are normalized as:

```text
top     → top
bottom  → pants
```

The database category is treated as authoritative when analyzing product images.

Product images stored as PostgreSQL `bytea` can be sent directly to the Vision model without first writing them to disk.

---

## Fashion Encoder

`fashion_encoder.py` converts garment descriptions into semantic vectors using BGE-M3.

```python
from fashion_retrieval.fashion_encoder import encode_item

encoded_item = encode_item(item)
```

Input:

```json
{
  "category": "top",
  "text_description": "A relaxed cream long-sleeve shirt..."
}
```

Output:

```json
{
  "category": "top",
  "text_description": "A relaxed cream long-sleeve shirt...",
  "embedding": ["..."],
  "embedding_model": "BAAI/bge-m3"
}
```

Only `text_description` is encoded.

Information such as category, price, tags, image paths, and URLs remains metadata.

---

## Project Structure

```text
HackThon/
│
├── fashion_retrieval/
│   ├── apify_client.py
│   ├── db_reader.py
│   ├── post_parser.py
│   ├── reel_parser.py
│   ├── image_filter.py
│   ├── fashion_analyzer.py
│   ├── fashion_encoder.py
│   └── pipeline.py
│
├── tests/
│   ├── test_apify.py
│   ├── test_post.py
│   ├── test_reel.py
│   ├── test_image_filter.py
│   ├── test_fashion_analyzer.py
│   └── test_fashion_encoder.py
│
├── outputs/
│
├── .env
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/EnterausenameBrian/HachThon.git
cd HachThon
```

Create and activate a Python environment.

Then install the required packages.

Example:

```bash
pip install anthropic
pip install python-dotenv
pip install sentence-transformers
pip install pillow
pip install numpy
pip install opencv-python
pip install apify-client
pip install psycopg2-binary
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
APIFY_TOKEN=
ANTHROPIC_API_KEY=

DB_HOST=
DB_PORT=5432
DB_NAME=
DB_USER=
DB_PASSWORD=
DB_SSLMODE=require
```

**Never commit `.env` to GitHub.**

---

## Testing

### Test Instagram / Apify

```bash
python -m tests.test_apify
```

### Test Post Pipeline

```bash
python -m tests.test_post
```

### Test Reel Pipeline

```bash
python -m tests.test_reel
```

### Test Image Filtering

```bash
python -m tests.test_image_filter
```

### Test Fashion Analyzer

Single image:

```bash
python -m tests.test_fashion_analyzer image path/to/image.jpg
```

Instagram Post:

```bash
python -m tests.test_fashion_analyzer post "INSTAGRAM_POST_URL"
```

Instagram Reel:

```bash
python -m tests.test_fashion_analyzer reel "INSTAGRAM_REEL_URL"
```

### Test Fashion Encoder

```bash
python -m tests.test_fashion_encoder
```

---

## Current Architecture

The system deliberately separates different responsibilities:

```text
Parser
  │
  │ obtains images / metadata
  ▼
Image Filter
  │
  │ removes unusable content
  ▼
Fashion Analyzer
  │
  │ image → semantic description
  ▼
Fashion Encoder
  │
  │ description → embedding
  ▼
Retrieval / Recommendation
```

This separation allows individual components to be modified without changing the entire pipeline.

---

## Future Work

Planned extensions include:

- Product preprocessing and persistent embedding storage
- Garment-level deduplication across Reel frames
- Semantic product retrieval
- User preference representation
- Outfit compatibility modeling
- Natural-language outfit queries
- Personalized recommendation
- Feedback-based preference updates

---

## Tech Stack

- **Python**
- **Anthropic Claude** — visual garment understanding
- **BAAI/BGE-M3** — semantic text embeddings
- **Apify** — Instagram content extraction
- **PostgreSQL** — product database
- **OpenCV / Pillow** — image and video processing
- **NumPy** — local image similarity processing
