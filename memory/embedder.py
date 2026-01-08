from sentence_transformers import SentenceTransformer

class GeminiEmbed:
    def __init__(self):
        # This downloads a tiny model (30MB) that runs on your CPU
        # No API keys required, no limits, 100% free
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("--- [EMBEDDER] Local Model Loaded (No Quota Limits) ---")

    def embed_text(self, text: str):
        # Returns a list of floats (384 dimensions)
        embedding = self.model.encode(text)
        return embedding.tolist()
