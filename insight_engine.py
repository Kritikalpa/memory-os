import os
from datetime import datetime, timedelta
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic import BaseModel, Field
from typing import List

# Import our existing database connection
from database import MemoryDB

# Setup Gemini
model = GoogleModel("gemini-3.1-pro-preview", settings=dict(temperature=0.4))

# --- 1. DEFINE INSIGHT SCHEMA ---
class WeeklyInsight(BaseModel):
    theme_of_the_week: str = Field(description="A short title for the overall vibe of the memories.")
    behavioral_patterns: List[str] = Field(description="List of observed patterns (e.g., 'You get frustrated when working with X', 'You are avoiding project Y').")
    key_entities_involved: List[str] = Field(description="Exact names of the entities (people/projects) driving these patterns.")

# --- 2. FETCH RECENT MEMORIES ---
def fetch_recent_memories(db: MemoryDB, days: int = 7) -> str:
    """Fetches memories from the last N days to analyze."""
    # Note: In a real system we filter by date. For this test, 
    # we'll just grab the last 20 memories so it works immediately.
    query = """
    MATCH (m:Memory)-[:MENTIONS]->(e:Entity)
    RETURN m.raw_text AS text, m.created_at AS date, collect(e.name) AS entities, m.primary_emotion AS emotion
    ORDER BY m.created_at DESC LIMIT 20
    """
    with db.driver.session() as session:
        result = session.run(query)
        records = [record.data() for record in result]
        
    if not records:
        return ""
        
    # Format for the LLM
    dump = "RECENT MEMORIES:\n"
    for r in records:
        dump += f"- [{r['date'][:10]}] Emotion: {r['emotion']} | Entities: {r['entities']} | Note: {r['text']}\n"
    return dump

# --- 3. GENERATE & SAVE INSIGHT ---
async def run_weekly_insight_job(db: MemoryDB):
    print("1. Fetching recent brain dumps from Graph...")
    memory_dump = fetch_recent_memories(db)
    
    if not memory_dump:
        print("Not enough data for insights.")
        return

    print("2. Gemini is analyzing your psychology and patterns...")
    
    prompt = f"""
    You are an analytical background processor for a personal Memory OS.
    Look at the user's recent memories. Find underlying behavioral patterns, 
    recurring stressors, or positive trends. 
    Be direct, objective, and insightful.
    
    {memory_dump}
    """

    agent = Agent(
        model=model,
        system_prompt=prompt,
        output_type=WeeklyInsight
    )
    
    response = await agent.run()
    
    insight_data = response.output
    
    print("\n💡 NEW INSIGHT GENERATED:")
    print(f"Theme: {insight_data.theme_of_the_week}")
    for pattern in insight_data.behavioral_patterns:
        print(f"- {pattern}")
    
    print("\n3. Saving Insight back to the Knowledge Graph...")
    # --- THIS IS THE MAGIC ---
    # We create an [Insight] node and connect it to the Entities!
    save_query = """
    CREATE (i:Insight {
        theme: $theme,
        patterns: $patterns,
        created_at: $created_at
    })
    WITH i
    UNWIND $entities AS entity_name
    MATCH (e:Entity {name: entity_name}) // Find existing entities
    MERGE (i)-[:DERIVED_FROM]->(e)       // Connect insight to entity
    """
    
    parameters = {
        "theme": insight_data.theme_of_the_week,
        "patterns": insight_data.behavioral_patterns,
        "entities": insight_data.key_entities_involved,
        "created_at": datetime.now().isoformat()
    }
    
    with db.driver.session() as session:
        session.run(save_query, parameters)
        
    print("✅ Insight successfully woven into the Graph!")

# --- RUN THE JOB ---
if __name__ == "__main__":
    URI = os.environ.get("NEO4J_URI")
    USER = os.environ.get("NEO4J_USERNAME")
    PASSWORD = os.environ.get("NEO4J_PASSWORD")
    
    db = MemoryDB(URI, USER, PASSWORD)
    import asyncio
    asyncio.run(run_weekly_insight_job(db))
    db.close()