from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from pypdf import PdfReader

import faiss

# =========================================================
# 1. 读取 PDF
# =========================================================

pdf_path = "IPCC_AR6_WGII_Chapter03.pdf"

print("Loading PDF...")

reader = PdfReader(pdf_path)

print(f"Total pages: {len(reader.pages)}")

# =========================================================
# 2. 提取文本
# =========================================================

full_text = ""

for i, page in enumerate(reader.pages):

    print(f"Processing page {i+1}/{len(reader.pages)}")

    text = page.extract_text()

    if text:
        full_text += text + "\n"

print("\nFinished PDF extraction.")
print(f"Total text length: {len(full_text)}")

# =========================================================
# 3. 创建 LangChain Document
# =========================================================

docs = [Document(page_content=full_text)]

# =========================================================
# 4. Embedding model
# =========================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Embedding model loaded.")

# =========================================================
# 5. Parent / Child splitters
# =========================================================

# 小 chunk：用于 retrieval
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

# 大 chunk：用于 generation / context recovery
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# =========================================================
# 6. 创建 FAISS vector store
# =========================================================

# bge-small-en-v1.5 embedding dimension = 384
embedding_dimension = 384

index = faiss.IndexFlatL2(embedding_dimension)

vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# =========================================================
# 7. Parent document store
# =========================================================

store = InMemoryStore()

# =========================================================
# 8. ParentDocumentRetriever
# =========================================================

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# =========================================================
# 9. 添加文档
# =========================================================

print("\nAdding documents to retriever...")

retriever.add_documents(docs)

print("Documents added.")

# =========================================================
# 10. 查询
# =========================================================

query = "What are the concerns surrounding the AMOC?"

print(f"\nQuery: {query}")

results = retriever.get_relevant_documents(query)

# =========================================================
# 11. 输出结果
# =========================================================

for i, doc in enumerate(results):

    print("\n" + "=" * 80)
    print(f"Result {i+1}")
    print("=" * 80)

    print(doc.page_content[:2000])
