from neo4j import GraphDatabase
import os

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "1234567890")

class Neo4jConnector:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASS))

    def run(self, cypher: str, params: dict = None):
        try:
            with self.driver.session() as session:
                return [dict(record) for record in session.run(cypher, params or {})]
        except Exception as e:
            print(f"Neo4j query failed: {str(e)}")
            return []

    def close(self):
        self.driver.close()