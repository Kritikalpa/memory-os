import os
from neo4j import GraphDatabase
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class MemoryDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()

    def setup_database(self):
        """Creates the Vector Index for Gemini Embeddings (3072 dimensions)"""
        setup_query = """
        CREATE VECTOR INDEX memory_embedding IF NOT EXISTS
        FOR (m:Memory)
        ON (m.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 3072,
          `vector.similarity_function`: 'cosine'
        }}
        """
        # Also create a constraint so we don't duplicate entities
        constraint_query = """
        CREATE CONSTRAINT entity_name IF NOT EXISTS 
        FOR (e:Entity) REQUIRE e.name IS UNIQUE
        """
        
        with self.driver.session() as session:
            session.run(setup_query)
            session.run(constraint_query)
            print("Database setup complete: Vector Index and Constraints verified.")

    def insert_memory(self, memory_package: dict):
        """
        Takes the package from embeddings.py and stores it as a Graph + Vector.
        """
        query = """
        // 1. Create the Memory Node
        CREATE (m:Memory {
            id: randomUUID(),
            raw_text: $raw_text,
            summary: $summary,
            emotional_intensity: $emotional_intensity,
            sentiment_score: $sentiment_score,
            primary_emotion: $primary_emotion,
            created_at: $created_at
        })
        
        // 2. Set the embedding vector
        // Neo4j stores vectors directly as list properties
        SET m.embedding = $embedding
        
        // 3. Connect to Entities (People, Projects, Topics)
        // UNWIND is like a for-loop in Cypher
        WITH m
        UNWIND $entities AS entity_name
        // MERGE means "Create if it doesn't exist, otherwise fetch it"
        MERGE (e:Entity {name: entity_name})
        // Create the connection
        MERGE (m)-[:MENTIONS]->(e)
        
        RETURN m.id AS memory_id
        """
        
        # Flatten the data for the query
        parameters = {
            "raw_text": memory_package["raw_text"],
            "summary": memory_package["metadata"]["summary"],
            "emotional_intensity": memory_package["metadata"]["emotional_intensity"],
            "sentiment_score": memory_package["metadata"]["sentiment_score"],
            "primary_emotion": memory_package["metadata"]["primary_emotion"],
            "created_at": datetime.now().isoformat(),
            "embedding": memory_package["embedding"],
            "entities": memory_package["metadata"]["entities"]
        }
        
        with self.driver.session() as session:
            result = session.run(query, parameters)
            record = result.single()
            return record["memory_id"]

# --- TEST IT OUT ---
if __name__ == "__main__":
    # In a real app, import this from your embeddings.py file.
    # For testing, we simulate the package generated in Step 2:
    from embeddings import process_new_memory
    
    URI = os.environ.get("NEO4J_URI")
    USER = os.environ.get("NEO4J_USERNAME")
    PASSWORD = os.environ.get("NEO4J_PASSWORD")
    
    if not URI:
        print("Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")
        exit()

    # 1. Connect
    db = MemoryDB(URI, USER, PASSWORD)
    
    # 2. Prepare DB (Run once)
    db.setup_database()
    
    # 3. Generate a new memory
    brain_dump = (
        "Just finished the sync with Sarah. Honestly, I feel completely burnt out "
        "and frustrated. We keep talking in circles about the database migration. "
        "I need to email the DevOps team tomorrow to get their opinion, but I "
        "really just want to abandon this whole feature right now."
    )
    
    print("Processing memory through Gemini...")
    memory_pkg = process_new_memory(brain_dump)
    
    print("Saving to Neo4j Graph Database...")
    memory_id = db.insert_memory(memory_pkg)
    
    print(f"✅ Success! Memory saved with Graph ID: {memory_id}")
    db.close()