from typing import List
from llm.llm_client import GeminiClient
from memory.episodic_store import Episode

class EpisodicSummarizer:
    """
    Compresses episodic memory into semantic memory.
    """
    def __init__(self,llm : GeminiClient):
        self.llm = llm

    def summarize_episode(self,episode : Episode) -> str:
        prompt = (
            "Summarize the following interaction into a concise, factual memory.\n"
            "Focus on user goals, preferences, and important context.\n\n"
            f"Events:\n" + "\n".join(f"- {e}" for e in episode.events)
        )

        response = self.llm.generate(
            system_prompt="You are a research assistant. Summarize the following interaction into a single, factual memory string.",
            user_prompt=prompt,
            memories=None  # We don't need past memories to summarize the current ones
        )

        return response.strip()

