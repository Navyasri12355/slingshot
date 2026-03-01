from onnx_embedder import ONNXEmbedder
from chroma_store import ChromaStore
from ollama_llm import OllamaLLM  # your Ollama wrapper

# -------------------------
# Initialize embedder, database, and LLM
# -------------------------
embedder = ONNXEmbedder()
store = ChromaStore()
llm = OllamaLLM(model="mistral")  # replace with your Ollama model

# Keep chat history for multi-turn reasoning
chat_history = []


def retrieve_context(query: str, k: int = 3):
    """
    Retrieves top-k relevant chunks from ChromaDB.
    Returns context text and a set of used sources.
    """
    embedding = embedder.embed(query)[0]
    results = store.query(embedding, k=k)

    docs = results["documents"]
    metas = results["metadatas"]

    context_chunks = []
    used_sources = set()

    for doc, meta in zip(docs, metas):
        source = meta.get("source", "unknown")
        chunk_idx = meta.get("chunk", 0)
        used_sources.add(source)
        context_chunks.append(f"[Source: {source} | Chunk: {chunk_idx}]\n{doc}")

    context_text = "\n\n".join(context_chunks)
    return context_text, used_sources


def ask(query: str, k: int = 3):
    """
    Sends query to the offline AI agent and returns the AI response.
    """
    context, used_sources = retrieve_context(query, k=k)

    history_text = "\n".join(chat_history[-6:])  # last 3 turns
    prompt = f"{history_text}\n\nContext:\n{context}\n\nQuestion:\n{query}"

    response = llm.generate(prompt)

    # Update chat history
    chat_history.append(f"User: {query}")
    chat_history.append(f"AI: {response}")

    # Append sources for traceability
    if used_sources:
        sources_text = ", ".join(sorted(used_sources))
        response += f"\n\n[Sources used: {sources_text}]"

    return response


# -------------------------
# Exportable function for UI / integration
# -------------------------
def search_memory(query: str) -> str:
    """
    Clean exportable function that returns AI response for a given query.
    Can be imported by FastAPI backend or other modules.
    """
    return ask(query)


# -------------------------
# Interactive test
# -------------------------
if __name__ == "__main__":
    print("=== Offline AI Agent Ready ===")
    print("Type your question below. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("\nExiting Offline AI Agent. Goodbye!")
            break

        try:
            answer = search_memory(query)
            print(f"\nAI: {answer}\n")
        except Exception as e:
            print(f"\nError: {e}\n")