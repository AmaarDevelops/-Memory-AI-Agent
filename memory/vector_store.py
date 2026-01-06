import faiss
import os
import numpy as np
from typing import List,Dict,Tuple


class VectorStore:
    """
    FAISS Based vectore storage for embedding similarity search
    """

    def __init__(self,dim : int, index_path : str = "data/memory_store/faiss.index"):
        self.dim = dim
        self.index_path = index_path

        os.makedirs(os.path.dirname(index_path),exist_ok=True)

        if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
            try:
                print(f"Loading existing index from {index_path}...")
                self.index = faiss.read_index(index_path)
            except Exception as e:
                print(f"Error reading index: {e}. Starting fresh.")
                self.index = faiss.IndexFlatL2(dim)
        else:
            print("No existing memory found. Initializing new FAISS index...")
            # This creates the empty 'container' in RAM
            self.index = faiss.IndexFlatL2(dim)
            

    def _normalize(self,vectors:np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity
        """

        return vectors / np.linalg.norm(vectors,axis=1,keepdims=True)


    def add(self,embeddings : List[List[float]]) -> List[int]:
        """
        Add embeddings to the index.
        Returns FAISS internal ids.
        """

        vectors = np.array(embeddings).astype('float32')
        vectors = self._normalize(vectors)

        start_id = self.index.ntotal
        self.index.add(vectors)
        self.save()

        return list(range(start_id,start_id + len(vectors)))


    def search(self,qeury_embeddings: List[float],top_k : int = 5) -> Tuple[List[int], List[float]]:
        """
        Search for nearest vectors.
        Returns (ids,cosine_similarity)
        """

        query = np.array([qeury_embeddings]).astype('float32')
        query = self._normalize(query)

        scores,ids = self.index.search(query,top_k)

        return ids[0].tolist(),scores[0].tolist()

    def save(self):
        """
        Persist Faiss index to disk
        """
        faiss.write_index(self.index,self.index_path)



