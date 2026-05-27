from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from pypdf import PdfReader

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
# 2. 创建 Document
# =========================

docs = [Document(page_content=full_text)]

# =========================
# 3. embedding model
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# =========================
# 4. parent / child splitters
# =========================

# 小 chunk 用于 retrieval
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

# 大 chunk 用于返回上下文
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# =========================
# 5. vector store
# =========================

vectorstore = FAISS.from_documents(
    [],
    embedding=embeddings
)

store = InMemoryStore()

# =========================
# 6. ParentDocumentRetriever
# =========================

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)

# =========================
# 7. 查询
# =========================

query = "What are the concerns surrounding the AMOC?"

results = retriever.get_relevant_documents(query)

# =========================
# 8. 输出
# =========================

print(f"\n查询: {query}")

for i, doc in enumerate(results):

    print("\n" + "=" * 80)
    print(f"Result {i+1}")
    print("=" * 80)

    print(doc.page_content[:2000])
