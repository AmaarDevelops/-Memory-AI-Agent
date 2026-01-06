from typing import List,Dict
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Episode:
    id: str = field(default_factory=lambda:str(uuid.uuid4()))
    events : List[str] = field(default_factory=list)
    start_time : datetime = field(default_factory=datetime.utcnow)
    end_time : datetime | None = None
    metadata : Dict = field(default_factory=dict)

    def add_events(self,text:str):
        self.events.append(text)
        self.end_time = datetime.utcnow()



class EpisodicMemoryStore:
    """
    Temporary high resolution memory
    """

    def __init__(self,max_events : int = 2):
        self.current_episodes = Episode()
        self.completed_episodes: List[Episode] = []
        self.max_events = max_events


    def add_events(self,text: str):
        self.current_episodes.add_events(text)

        if len(self.current_episodes.events) >= self.max_events:
            print(f"--- [DEBUG] Limit {self.max_events} hit. Closing episode. ---")
            self._close_episode()


    def _close_episode(self):
        self.completed_episodes.append(self.current_episodes)
        self.current_episodes = Episode()


    def pop_completed(self) -> List[Episode]:
        episodes = list(self.completed_episodes)
        self.completed_episodes = []
        return episodes




