# kb_setup.py

import os
from sentence_transformers import SentenceTransformer
import chromadb

# -------------------------------
# 1. Load embedding model
# -------------------------------
print("🔄 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 2. Initialize ChromaDB
# -------------------------------
print("🔄 Initializing ChromaDB...")
client = chromadb.Client()
collection = client.create_collection(name="course_assistant")

# -------------------------------
# 3. Load documents from /data
# -------------------------------
print("🔄 Loading documents from data folder...")

data_path = "data"
documents = []

for file in os.listdir(data_path):
    if file.endswith(".txt"):
        file_path = os.path.join(data_path, file)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

            # Split topic and text
            lines = content.split("\n", 1)
            topic = lines[0].strip()
            text = lines[1].strip() if len(lines) > 1 else ""

            documents.append({
                "id": file.replace(".txt", ""),
                "topic": topic,
                "text": text
            })

print(f"✅ Loaded {len(documents)} documents")

# -------------------------------
# 4. Prepare data for ChromaDB
# -------------------------------
texts = [doc["text"] for doc in documents]
ids = [doc["id"] for doc in documents]
metadatas = [{"topic": doc["topic"]} for doc in documents]

# -------------------------------
# 5. Create embeddings
# -------------------------------
print("🔄 Creating embeddings...")
embeddings = embedder.encode(texts).tolist()

# -------------------------------
# 6. Store in ChromaDB
# -------------------------------
collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

print("✅ Documents stored in ChromaDB")

# -------------------------------
# 7. Retrieval Function
# -------------------------------
def retrieve(query):
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    return results

# -------------------------------
# 8. Test Retrieval
# -------------------------------
def test_retrieval():
    test_queries = [
        "What is LangGraph?",
        "Explain RAG",
        "What does router node do?",
        "How does memory work?",
        "What is ChromaDB?"
    ]

    for query in test_queries:
        print("\n" + "="*50)
        print("🔍 Query:", query)

        results = retrieve(query)

        for i in range(len(results["documents"][0])):
            print(f"\nResult {i+1}")
            print("Topic:", results["metadatas"][0][i]["topic"])
            print("Text:", results["documents"][0][i][:150], "...")

# -------------------------------
# 9. Run test
# -------------------------------
if __name__ == "__main__":
    test_retrieval()