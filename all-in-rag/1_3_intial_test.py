import torch

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate


# =====================================================
# 1. Load Document
# =====================================================

markdown_path = "easy-rl-chapter1.md"

loader = TextLoader(
    markdown_path,
    encoding="utf-8"
)

docs = loader.load()

print(f"\nNumber of documents: {len(docs)}")


# =====================================================
# 2. Split Text into Chunks
# =====================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

texts = text_splitter.split_documents(docs)

print(f"\nNumber of chunks: {len(texts)}")


# =====================================================
# 3. Load Embedding Model
# =====================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

print("\nEmbedding model loaded successfully")


# =====================================================
# 4. Build FAISS Vector Store
# =====================================================

vectorstore = FAISS.from_documents(
    texts,
    embeddings
)

print("\nFAISS index built successfully")


# =====================================================
# 5. User Question
# =====================================================

question = "What reinforcement learning examples are mentioned in the document?"


# =====================================================
# 6. Retrieval
# =====================================================

retrieved_docs = vectorstore.similarity_search(
    question,
    k=3
)

print("\n================ Retrieval Results ================\n")

for i, doc in enumerate(retrieved_docs):

    print(f"\n===== Chunk {i+1} =====\n")

    print(doc.page_content[:500])

    print("\n--------------------------------------------------\n")


# =====================================================
# 7. Build Context
# =====================================================

docs_content = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)


# =====================================================
# 8. Prompt Template
# =====================================================

prompt = ChatPromptTemplate.from_template(
"""
Answer the question based only on the provided context.

Requirements:
1. Only use information from the context
2. Do not hallucinate or invent information
3. If the answer is unknown, clearly say so
4. Keep the answer concise and clear

Context:
{context}

Question:
{question}

Answer:
"""
)


# =====================================================
# 9. Load Local Qwen Model
# =====================================================

model_name = "Qwen/Qwen2.5-7B-Instruct"

print("\nLoading local LLM...\n")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

llm = HuggingFacePipeline(
    pipeline=pipe
)

print("\nLocal LLM loaded successfully")


# =====================================================
# 10. Construct Final Prompt
# =====================================================

final_prompt = prompt.format(
    question=question,
    context=docs_content
)


# =====================================================
# 11. Generate Final Answer
# =====================================================

print("\n================ FINAL ANSWER ================\n")

answer = llm.invoke(final_prompt)

print(answer)
