from langchain_community.embeddings import HuggingFaceEmbeddings


def mod_query(query: str, embedding_model: HuggingFaceEmbeddings) -> dict:
  
    query_cleaned = query.replace("El usuario busca ", "").strip()
    
    # Utiliza el método embed_query para generar el embedding
    query_embedding = embedding_model.embed_query(query_cleaned)
    
    return {
        "query_cleaned": query_cleaned,
        "query_embedding": query_embedding,
    }
