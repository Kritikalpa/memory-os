import os
from pydantic_ai.models.google import GoogleModel
from google import genai
from typing import List, Dict, Any
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Import the extraction function and model we built in the last step
# (Make sure extraction.py is in the same folder)
from extraction import extract_memory_data, MemoryData 

client = genai.Client()

def generate_embedding(text: str) -> List[float]:
    """
    Calls Gemini's embedding model to convert text into a 3072-dimensional vector.
    """

    response = client.models.embed_content(
        model="gemini-embedding-001",  # stable model name
        contents=text
    )

    return response.embeddings[0].values

def process_new_memory(raw_text: str) -> Dict[str, Any]:
    """
    The Master Pipeline: 
    1. Extracts structured data (Entities, Emotion, Tasks)
    2. Constructs a dense string for embedding
    3. Generates the vector
    """
    print("1. Extracting structured data via LLM...")
    structured_data = asyncio.run(extract_memory_data(raw_text))
    
    # 2. Build a "Rich Context" string to embed.
    # We combine the raw text with the LLM's summary and entities to make 
    # the vector highly dense and searchable.
    rich_text_to_embed = f"""
    Summary: {structured_data.summary}
    Entities: {', '.join(structured_data.entities)}
    Primary Emotion: {structured_data.primary_emotion}
    Raw Content: {raw_text}
    """
    
    print("2. Generating vector embedding...")
    vector = generate_embedding(rich_text_to_embed)
    
    # Return the complete package ready for the Database
    return {
        "raw_text": raw_text,
        "metadata": structured_data.model_dump(), # Converts Pydantic to Dict
        "embedding": vector,
        "embedding_dimensions": len(vector)
    }

# --- TEST IT ---
if __name__ == "__main__":
    brain_dump = (
        "Just finished the sync with Sarah. Honestly, I feel completely burnt out "
        "and frustrated. We keep talking in circles about the database migration. "
        "I need to email the DevOps team tomorrow to get their opinion, but I "
        "really just want to abandon this whole feature right now."
    )
    
    memory_package = process_new_memory(brain_dump)
    
    print("\n--- FINAL MEMORY PACKAGE FOR DB ---")
    print(f"Summary: {memory_package['metadata']['summary']}")
    print(f"Entities to become Graph Nodes: {memory_package['metadata']['entities']}")
    print(f"Vector length: {memory_package['embedding_dimensions']} dimensions")
    print(f"First 5 vector numbers: {memory_package['embedding'][:5]}...")