from numpy import array
def hybrid_retrieve(vdb, query, k, rank_weight=0.3, sim_weight=0.7, filter=None):
    docs_with_scores = vdb.similarity_search_with_score(query, k=k, filter=filter or None)
    docs = []
    for doc, distance in docs_with_scores:
        doc.metadata["similarity_score"] = 1 - distance
        docs.append(doc)
    similarities = array([d.metadata.get("similarity_score", 0.0) for d in docs])
    ranks = array([d.metadata.get("rank", 0.0) for d in docs])
    ranks_norm = ranks / ranks.max() if ranks.max() > 0 else ranks
    hybrid_scores = sim_weight * similarities + rank_weight * ranks_norm
    for d, s in zip(docs, hybrid_scores):
        d.metadata["hybrid_score"] = float(s)
    return sorted(docs, key=lambda d: d.metadata["hybrid_score"], reverse=True)