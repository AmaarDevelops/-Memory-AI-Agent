import json
import os
from typing import List
from memory.schemas import Memory
from memory.vector_store import VectorStore
from memory.embedder import GeminiEmbed
from memory.episodic_store import EpisodicMemoryStore
from memory.summarizer import EpisodicSummarizer
from llm.llm_client import GeminiClient
from datetime import datetime
import numpy as np


class MemoryManager:
    """
    Handles long term and episodic memory
    """
    def __init__(self,vector_store : VectorStore, embedder : GeminiEmbed,
                 llm_client : GeminiClient,
                 embedding_dim = 768,faiss_path = './data/memory_store/faiss.index',
                 metadata_path='./data/memory_store/metadata.json'):

        self.embedder = embedder
        self.store = vector_store
        self.metadata_path = metadata_path

        self.llm_client = llm_client
        self.episodic_store = EpisodicMemoryStore(max_events=2)
        self.summarizer = EpisodicSummarizer(self.llm_client)


        try:
            if os.path.exists(self.metadata_path) and os.path.getsize(self.metadata_path) >0:
                with open(self.metadata_path,'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except json.JSONDecodeError:
            print("Metadata was empty or corrupting, starting fresh.")
            self.metadata = {}


        if self.metadata:
            # Sort keys so they match the FAISS index order (0, 1, 2...)
            sorted_keys = sorted(self.metadata.keys(), key=int)
            self.store.memories = [self.metadata[k]['text'] for k in sorted_keys]
            print(f"--- [SYSTEM] Synced {len(self.store.memories)} memories to VectorStore ---")



    def save_metadata(self):
        os.makedirs(os.path.dirname(self.metadata_path),exist_ok=True)
        with open(self.metadata_path,"w",encoding="utf-8") as f:
            json.dump(self.metadata,f,indent=4)


    def add_memory(self,memory:Memory) -> str:
        """
        Add a memory object:
           - Embed text
           - Add to FAISS
           - Save metadata
        Returns memory UUID
        """

        embeddings = self.embedder.embed_text(memory.text)
        faiss_ids = self.store.add([embeddings],[memory.text])
        faiss_id = faiss_ids[0]

        # Link FAISS ID -> Memory
        self.metadata[str(faiss_id)] = memory.to_dict()
        self.save_metadata()
        return memory.id


    def retrieve_memories(self, query: str, top_k: int = 3):

      embedding = self.embedder.embed_text(query)
      ids, scores = self.store.search(embedding, top_k=top_k)

      if len(ids) > 0 and isinstance(ids[0],list):
          ids = ids[0]

      retrieved_texts = []

      for idx in ids:
        if idx != -1 and idx < len(self.store.memories):
            retrieved_texts.append(self.store.memories[idx])

      print(f"--- [DEBUG] Retrieved {len(retrieved_texts)} memories ---")

      return retrieved_texts


    def add_interactions(self,text:str):
         # Always capture episodic memory
        self.episodic_store.add_events(text)


        # Consolidate if episode closed
        completed = self.episodic_store.pop_completed()

        print(f"--- [DEBUG] Completed episodes found: {len(completed)} ---")

        for episode in completed:
            summary = self.summarizer.summarize_episode(episode)
            new_mem = Memory(
                text=summary,
                memory_type="episodic_memory",
                importance=0.7,
                metadata={
                    "start_time": episode.start_time.isoformat(),
                    "end_time": episode.end_time.isoformat()
                }
            )

            self.add_memory(new_mem) # This is much cleaner

            self.store.save()
            print(f"--- [SYSTEM] Research consolidated: {summary[:50]}... ---")








