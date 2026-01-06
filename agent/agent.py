from typing import List
from memory.memory_manager import MemoryManager
from memory.schemas import Memory
from llm.llm_client import GeminiClient


SYSTEM_PROMPT = """
You are a memory-augmented AI agent.

You have access to past memories about the user.
Use them ONLY if relevant.
Do not invent memories.
Be concise, clear, and grounded.
""".strip()


class MemoryAgent:
    """
    Main cognitive loop for the memory agent.
    """

    def __init__(self):
        self.memory_manager = MemoryManager()
        self.llm = GeminiClient()
        self.short_term_memory : List[str] = []


    def should_store(self,user_input : str, response : str) -> bool:
        """
        Decide whether the information should be stores in memory.
        simple heuristic for now.
        """

        keywords = [
            "i like",
            "i love",
            "my goal",
            "i want",
            "i am",
            "remember",
            "always",
            "never"
        ]

        return any(k in user_input.lower() for k in keywords)


    def run(self,user_input : str) -> str:
        """
        Run One Agent interaction step.
        """

        # Retrieve long term memories
        retrieved_memories = self.memory_manager.retrieve_memories(
            query=user_input,
            top_k=5
        )

        memory_texts = [m.text for m in retrieved_memories]

        # Generate response
        response = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_input,
            memories=memory_texts
        )


        # Update short term memories
        self.short_term_memory.append(f"User : {user_input}")
        self.short_term_memory.append(f'Assistant : {response}')

        # Limit STM size
        if len(self.short_term_memory) > 10:
            self.short_term_memory = self.short_term_memory[-10:]



        # Decide whether to store long-term memory
        if self.should_store(user_input,response):
            memory = Memory(
                text = user_input,
                memory_type = "long_term",
                importance = 0.8
            )

            self.memory_manager.add_memory(memory)

        return response









