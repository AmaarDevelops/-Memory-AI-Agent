from datetime import datetime
import math

def apply_memory_decay(memories,decay_lambda=1e-6):
    """
    Adjust memory based on time and importance
    """

    now = datetime.utcnow()
    decayed = []

    for mem in memories:
        metadata = mem.metadata
        created_at = metadata.get("created_at")
        importance = metadata.get("importance",1.0)

        if not created_at:
            continue


        created_time = datetime.fromisoformat(created_at)
        delta_seconds = (now - created_time).total_seconds()

        decay_factor = math.exp(-decay_lambda * delta_seconds)

        effective_score = (
            mem['score'] * importance * decay_factor
        )

        mem['effective_score'] = effective_score
        decayed.append(mem)


    return sorted(decayed,
                  key=lambda m : m['effective_score'],
                  reverse=True)








