from scoring import calculate_memory_score
import os
import numpy as np
from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from neo4j import GraphDatabase

# Import your functions from previous files
from embeddings import generate_embedding
from database import MemoryDB

class InsightEngine:
    def __init__(self):
        self.model = GoogleModel("gemini-3.1-pro-preview")

    async def synthesize(self, user_query: str, retrieved_memories: list) -> str:
        """
        Uses Gemini to read the memories and answer the user's question.
        """
        # Format the memories for the LLM
        context_blocks = []
        for i, mem in enumerate(retrieved_memories):
            context_blocks.append(
                f"Memory {i+1}:\n"
                f"- Date: {mem['created_at'][:10]}\n"
                f"- Emotion: {mem['emotion']} (Intensity: {mem['emotional_intensity']})\n"
                f"- Entities Involved: {', '.join(mem['entities'])}\n"
                f"- Content: {mem['text']}"
            )
        context_str = "\n\n".join(context_blocks)

        # Define the System Prompt (The Persona)
        system_prompt = f"""
        You are the insight engine for my personal Memory OS. 
        Based ONLY on the retrieved memories below, answer my question. 
        Act like a helpful, analytical assistant. Point out patterns if you see them.
        
        Retrieved Memories:
        {context_str}
        """

        # Define the User Prompt (The Task)
        user_prompt = f"My Question: {user_query}"

        # Run the Agent
        agent = Agent(
            model=self.model,
            system_prompt=system_prompt
        )
        
        response = await agent.run(user_prompt)
        return response.output


    async def search_memories(self, db: MemoryDB, user_query: str):
        print(f"User Query: '{user_query}'")
        
        # 1. Embed the user's question
        print("Generating query vector...")
        query_vector = generate_embedding(user_query)
        
        # 2. Cypher Query: Vector Search + Graph Traversal
        # We fetch the top 10 semantic matches and pull their connected entities
        search_query = """
        CALL db.index.vector.queryNodes('memory_embedding', 10, $query_vector)
        YIELD node AS m, score AS semantic_similarity
        
        // Traverse the graph to get attached entities
        OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
        
        RETURN m.raw_text AS text, 
            m.created_at AS created_at,
            m.emotional_intensity AS emotional_intensity,
            m.primary_emotion AS emotion,
            collect(e.name) AS entities,
            semantic_similarity
        """
        
        print("Querying Neo4j Vector Index...")
        with db.driver.session() as session:
            result = session.run(search_query, {"query_vector": query_vector})
            records = [record.data() for record in result]
            
        if not records:
            return "I don't have any memories related to this."

        # 3. Apply Custom Re-Ranking
        print("Applying Temporal + Emotional re-ranking...")
        scored_memories = []
        for r in records:
            # Parse the ISO date string back to a Python datetime object
            memory_date = datetime.fromisoformat(r["created_at"])
            
            final_score = calculate_memory_score(
                semantic_similarity=r["semantic_similarity"],
                memory_date=memory_date,
                emotional_intensity=r["emotional_intensity"]
            )
            r["final_score"] = final_score
            scored_memories.append(r)
            
        # Sort by our custom score, highest first, and take the top 3
        scored_memories.sort(key=lambda x: x["final_score"], reverse=True)
        top_memories = scored_memories[:3]
        
        # 4. LLM Synthesis (The Final Output)
        print("Synthesizing response with Gemini...")
        
        response = await self.synthesize(user_query, top_memories)
        return response

# --- TEST IT OUT ---
if __name__ == "__main__":
    URI = os.environ.get("NEO4J_URI")
    USER = os.environ.get("NEO4J_USERNAME")
    PASSWORD = os.environ.get("NEO4J_PASSWORD")
    
    db = MemoryDB(URI, USER, PASSWORD)
    
    # Ask it a question that relates to the memory we just saved
    question = "What's been stressing me out lately, and who is involved?"
    
    import asyncio
    engine = InsightEngine()
    final_answer = asyncio.run(engine.search_memories(db, question))
    
    print("\n==================================")
    print("🤖 MEMORY OS INSIGHT:")
    print("==================================")
    print(final_answer)
    print("==================================")
    
    db.close()