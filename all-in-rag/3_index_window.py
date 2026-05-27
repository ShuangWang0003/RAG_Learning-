from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np
import re

# =========================
# 1. 读取 PDF
# =========================

pdf_path = "../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# =========================
# 2. 句子切分
# =========================

sentences = re.split(r'(?<=[.!?])\s+', full_text)

sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

print(f"总句子数: {len(sentences)}")

# =========================
# 3. sentence window
# =========================

window_size = 3

data = []

for i in range(len(sentences)):

    start = max(0, i - window_size)
    end = min(len(sentences), i + window_size + 1)

    window_text = " ".join(sentences[start:end])

    data.append({
        "sentence": sentences[i],
        "window": window_text
    })

# =========================
# 4. embedding model
# =========================

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

sentence_embeddings = model.encode(
    [x["sentence"] for x in data],
    normalize_embeddings=True,
    show_progress_bar=True
)

# =========================
# 5. 建立 FAISS index
# =========================

dimension = sentence_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(
    np.array(sentence_embeddings).astype("float32")
)

print("FAISS index 构建完成")

# =========================
# 6. 查询
# =========================

query = "What are the concerns surrounding the AMOC?"

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

top_k = 2

scores, indices = index.search(
    np.array(query_embedding).astype("float32"),
    top_k
)

# =========================
# 7. 输出 retrieval 结果
# =========================

print(f"\n查询: {query}")

for rank, idx in enumerate(indices[0]):

    print("\n" + "=" * 80)
    print(f"Top {rank+1}")
    print("=" * 80)

    print("\n[中心句子]")
    print(data[idx]["sentence"])

    print("\n[上下文窗口]")
    print(data[idx]["window"])

    print("\n[相似度]")
    print(scores[0][rank])
