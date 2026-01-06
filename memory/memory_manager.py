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
        faiss_ids = self.store.add([embeddings])
        faiss_id = faiss_ids[0]

        # Link FAISS ID -> Memory
        self.metadata[str(faiss_id)] = memory.to_dict()
        self.save_metadata()
        return memory.id


    def retrieve_memories(self,query : str , top_k : int = 5) -> List[Memory]:
        """
        Retrieve relevant memories for a query
        """

        embedding = self.embedder.embed_text(query)

        ids,scores = self.store.search(embedding,top_k=top_k)

        results = []

        if isinstance(ids, np.ndarray) and ids.ndim > 1:
            ids_to_process = ids[0]
        elif isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids_to_process = ids[0]
        else:
            ids_to_process = ids

        for idx in ids_to_process:
            # Important: FAISS often returns -1 if no match is found
            if idx == -1:
                continue



            mem_dict = self.metadata.get(str(int(idx)))

            if mem_dict:
                mem = Memory(
                    id = mem_dict['id'],
                    text = mem_dict['text'],
                    memory_type = mem_dict['memory_type'],
                    importance = mem_dict['importance'],
                    timestamp = mem_dict['timestamp'],
                    metadata = mem_dict['metadata']
                )
                results.append(mem)

        return results


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








