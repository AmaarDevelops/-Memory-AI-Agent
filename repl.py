from llm.llm_client import GeminiClient
from memory.embedder import GeminiEmbed
from memory.vector_store import VectorStore
from memory.memory_manager import MemoryManager
from memory.decay import apply_memory_decay
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()


def main():
    keys = [k.strip() for k in os.getenv("GEMINI_API_KEY").split(',')]

    genai.configure(api_key=keys[0])


    llm =   GeminiClient(api_keys = keys)
    embedder = GeminiEmbed()
    vector_store = VectorStore(
        index_path="./data/memory_store/faiss.index",
        dim = 768
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
        memories = apply_memory_decay(memories)[:3]

        for m in memories:
         print(
        "[DEBUG]",
        m.text,
        "score:", round(m.score, 4),
        "effective:", round(m.effective_score, 4)
        )



        memory_context = ""
        if memories:
            memory_context = "\n".join(
                f"- {m.text}" for m in memories
            )

        # 3. Generate response
        system_instructions = "You are a concise AI assistant for smart glasses research."
        # The 'user_input' is what you actually typed
        # The 'memories' should be passed as a list of strings

        memory_list = [m.text for m in memories] if memories else []

        # 4. Generate response with the correct arguments
        response = llm.generate(
         system_prompt=system_instructions,
         user_prompt=user_input,
         memories=memory_list
        )

        print(f'Agent : {response} \n')


if __name__ == "__main__":
    main()
