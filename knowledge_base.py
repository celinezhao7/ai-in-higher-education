import requests
from bs4 import BeautifulSoup
import pickle
import re
from sentence_transformers import SentenceTransformer
import numpy as np

# ----------------------------
# 1. Your sources
# ----------------------------
sources = [
    {
        "title": "Swan 2025 - Avoiding AI",
        "url": "https://dailynexus.com/2025-07-03/could-avoiding-ai-prepare-you-for-an-ai-integrated-world/"
    },
    {
        "title": "Giant 2025 - Replace Students with AI",
        "url": "https://dailynexus.com/2025-07-29/ucsb-to-replace-one-third-of-students-with-ai/"
    },
    {
        "title": "Reinke et al. 2025",
        "url": "https://doaj.org/article/7178f7cf0518403da9c1cb04ccce7a0d"
    },
    {
        "title": "Bond et al. 2024",
        "url": "https://research.ebsco.com/c/fml6nw/viewer/pdf/puh37vw5pf?route=details"
    },
    {
        "title": "Kosmyna et al. 2025",
        "url": "https://arxiv.org/abs/2506.08872"
    },
    {
        "title": "UCSB AI Guidelines",
        "url": "https://cio.ucsb.edu/artificial-intelligence/ai-use-guidelines"
    },
    {
        "title": "UCSB AI in Classes",
        "url": "https://otl.ucsb.edu/resources/assessing-learning/ai-in-classes"
    }
]

# ----------------------------
# 2. Scrape function
# ----------------------------
def scrape_page(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.extract()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------------------
# 3. Chunk text
# ----------------------------
def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

all_chunks = []
metadata = []

for src in sources:
    try:
        print(f"Scraping {src['title']} ...")
        text = scrape_page(src["url"])
        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append(c)
            metadata.append({"title": src["title"], "url": src["url"]})
    except Exception as e:
        print(f"Failed {src['title']}: {e}")

print(f"Total chunks: {len(all_chunks)}")

# ----------------------------
# 4. Save chunks + metadata
# ----------------------------
with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Knowledge base ready!")