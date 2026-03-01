import chromadb


class ChromaStore:

    def __init__(self):

        # ✅ TRUE DISK PERSISTENCE
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        print("ChromaDB initialized successfully")

        self.collection = self.client.get_or_create_collection(
            name="offline_memory"
        )

    # -------------------------
    # ADD DOCUMENTS
    # -------------------------
    def add_documents(self, ids, documents, embeddings, metadatas):

        # Ensure each metadata dict has 'source' and 'chunk'
        safe_metadatas = []
        for i, meta in enumerate(metadatas):
            safe_meta = {
                "source": meta.get("source", f"unknown_{i}"),
                "chunk": meta.get("chunk", i)
            }
            safe_metadatas.append(safe_meta)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=safe_metadatas
        )

        print("Documents stored in ChromaDB ✅")

    # -------------------------
    # QUERY
    # -------------------------
    def query(self, embedding, k=3):
        """
        Returns top-k relevant documents and metadatas as unwrapped lists.
        """

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )

        # Unwrap batch results (ChromaDB returns lists of lists)
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []

        return {
            "documents": docs,
            "metadatas": metas
        }