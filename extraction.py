from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

class MemoryData(BaseModel):
    summary: str = Field(description="A concise, 1-sentence summary of the memory.")
    entities: List[str] = Field(description="List of people, projects, or concepts mentioned (e.g., ['Alex', 'Side Project']).")
    sentiment_score: float = Field(description="Score from -1.0 (highly negative) to 1.0 (highly positive).")
    emotional_intensity: float = Field(description="Score from 1.0 (boring/neutral) to 2.0 (highly emotional/impactful).")
    primary_emotion: str = Field(description="The primary emotion felt (e.g., Joy, Frustration, Burnout, Excitement).")
    action_items: List[str] = Field(description="Any implied or explicit tasks to do later.")

async def extract_memory_data(raw_text: str) -> MemoryData:
    """Takes a raw text string and returns structured JSON/Pydantic object."""
    
    model = GoogleModel("gemini-3.1-pro-preview")
    
    prompt = f"""
    You are the extraction engine for a personal 'Memory OS'. 
    Analyze the following journal entry/brain dump.
    Extract the entities, sentiment, emotional intensity, and action items.
    
    Raw Memory: "{raw_text}"
    """

    agent = Agent(
        model=model,
        system_prompt=prompt,
        output_type=MemoryData
    )
    
    # We pass the Pydantic model to response_schema to guarantee the output format
    response = await agent.run()
    
    # Parse the returned JSON string back into our Pydantic model
    return response.output

# --- 4. TEST IT ---
if __name__ == "__main__":
    import asyncio
    # Test Scenario
    brain_dump = (
        "Just finished the sync with Sarah. Honestly, I feel completely burnt out "
        "and frustrated. We keep talking in circles about the database migration. "
        "I need to email the DevOps team tomorrow to get their opinion, but I "
        "really just want to abandon this whole feature right now."
    )
    
    print("Processing memory...")
    structured_memory = asyncio.run(extract_memory_data(brain_dump))
    
    print("\n--- EXTRACTED MEMORY DATA ---")
    print(f"Summary:    {structured_memory.summary}")
    print(f"Entities:   {structured_memory.entities}")
    print(f"Emotion:    {structured_memory.primary_emotion} (Intensity: {structured_memory.emotional_intensity})")
    print(f"Sentiment:  {structured_memory.sentiment_score}")
    print(f"Tasks:      {structured_memory.action_items}")