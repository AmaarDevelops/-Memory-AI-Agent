from llm.llm_client import GeminiClient
from memory.embedder import GeminiEmbed
from memory.vector_store import VectorStore
from memory.memory_manager import MemoryManager
from memory.decay import apply_memory_decay
from dotenv import load_dotenv
import os
import requests


load_dotenv()


def main():
    keys = [k.strip() for k in os.getenv("GEMINI_API_KEY").split(',')]

    # Removed genai configure line from here


    llm =   GeminiClient(api_keys = keys)
    embedder = GeminiEmbed()
    vector_store = VectorStore(
        index_path="./data/memory_store/faiss.index",
        dim = 384
    )

    memory_manager = MemoryManager(
        vector_store=vector_store,
        embedder=embedder,
        llm_client=llm
    )

    print("Memory-Augmented Agent (Episodic Mode)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        # 1. Store episodic interaction
        memory_manager.add_interactions(user_input)

        # 2. Retrieve relevant long-term memory
        memories = memory_manager.retrieve_memories(user_input)

        if memories:
            if not isinstance(memories[0], str):
                # Only apply decay if these are Memory objects
                memories = apply_memory_decay(memories)[:3]
                memory_list = [m.text for m in memories]
            else:
                # If they are already strings, use them directly
                memories = memories[:3]
                memory_list = memories
        else:
            memory_list = []

        # --- DEBUG PRINT ---
        for m in memories:
            if isinstance(m, str):
                print(f"[DEBUG] Memory: {m}")
            else:
                print(f"[DEBUG] Memory: {m.text} (Score: {getattr(m, 'score', 'N/A')})")

        # 3. Generate response
        system_instructions = (
            "You are a concise AI assistant for smart glasses research. "
            "Use the provided context to answer personal questions about the user."
        )

        # 4. Generate response with the correct arguments
        response = llm.generate(
            system_prompt=system_instructions,
            user_prompt=user_input,
            memories=memory_list
        )

        print(f'Agent : {response} \n')
        

if __name__ == "__main__":
    main()
