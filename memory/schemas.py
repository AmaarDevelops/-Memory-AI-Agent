# Schemas.py

# --------------            Memory Schema for the LLM Agent        -----------

from dataclasses import dataclass,field
from typing import Dict,Any
from datetime import datetime
import uuid


@dataclass
class Memory:
    """
    Base memory unit for the agent.
    """

    id : str = field(default_factory=lambda: str(uuid.uuid4()))
    text : str = ""
    memory_type : str = "long_term"     # Long term / Short term / episodic
    importance : float = 0.5            # 0.0 -> 1.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)



    def to_dict(self) -> Dict[str, Any]:
        return {
            "id" : self.id,
            "text" : self.text,
            "memory_type" : self.memory_type,
            "importance" : self.importance,
            "timestamp" : self.timestamp,
            "metadata" : self.metadata
        }











