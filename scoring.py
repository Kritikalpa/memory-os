import numpy as np
from datetime import datetime, timedelta

def calculate_memory_score(
    semantic_similarity: float, 
    memory_date: datetime, 
    emotional_intensity: float = 1.0, 
    decay_rate: float = 0.05
) -> float:
    """
    Calculates the relevance of a memory based on semantic match, time decay, and emotion.
    
    :param semantic_similarity: Cosine similarity from Vector DB (0.0 to 1.0)
    :param memory_date: When the memory happened
    :param emotional_intensity: 1.0 (neutral) to 2.0 (highly emotional/important)
    :param decay_rate: How fast memories fade (lower = slower fade)
    """
    
    # Calculate days since the memory occurred
    days_old = (datetime.now() - memory_date).days
    if days_old < 0: days_old = 0
    
    # 1. Temporal Decay (Exponential decay)
    # The higher the emotional intensity, the slower it decays
    adjusted_decay_rate = decay_rate / emotional_intensity
    time_weight = np.exp(-adjusted_decay_rate * days_old)
    
    # 2. Final Score Calculation
    # Combines vector search, emotion, and time
    final_score = semantic_similarity * emotional_intensity * time_weight
    
    return final_score

# --- TEST IT OUT ---
if __name__ == "__main__":
    now = datetime.now()
    
    # Scenario A: A neutral note from 30 days ago
    score_a = calculate_memory_score(
        semantic_similarity=0.85, 
        memory_date=now - timedelta(days=30), 
        emotional_intensity=1.0 # Neutral
    )
    
    # Scenario B: A highly emotional diary entry from 30 days ago
    score_b = calculate_memory_score(
        semantic_similarity=0.85, 
        memory_date=now - timedelta(days=30), 
        emotional_intensity=1.8 # High emotion (e.g., stress after meeting X)
    )

    print(f"Neutral Memory Score (30 days old): {score_a:.4f}")
    print(f"Emotional Memory Score (30 days old): {score_b:.4f}")
    # You will see the emotional memory scores significantly higher, 
    # meaning it will be retrieved by the LLM more easily!