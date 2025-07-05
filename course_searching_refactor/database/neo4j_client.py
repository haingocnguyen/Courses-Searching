import logging
import streamlit as st
from neo4j import GraphDatabase
from config import NEO4J_CONFIG

logger = logging.getLogger(__name__)

class Neo4jConnection:
    """Neo4j database connection handler with caching support"""
    
    def __init__(self, uri=None, user=None, pwd=None, max_connection_pool_size=25):
        config = NEO4J_CONFIG
        self._driver = GraphDatabase.driver(
            uri or config["uri"], 
            auth=(user or config["user"], pwd or config["password"]),
            max_connection_pool_size=max_connection_pool_size,
            connection_acquisition_timeout=config["connection_acquisition_timeout"],
            max_transaction_retry_time=config["max_transaction_retry_time"]
        )
    
    @st.cache_data(ttl=600, max_entries=50)
    def execute_query_cached(_self, query_hash, q, p=None):
        """Execute cached query with read transaction for better performance"""
        try:
            with _self._driver.session(default_access_mode="READ") as s:
                result = s.run(q, p or {})
                return [dict(r) for r in result]
        except Exception as e:
            logger.error("Neo4j query failed: %s", e)
            return []
    
    def execute_query(self, q, p=None):
        """Execute query without caching for real-time data"""
        try:
            with self._driver.session(default_access_mode="READ") as s:
                result = s.run(q, p or {})
                return [dict(r) for r in result]
        except Exception as e:
            logger.error("Neo4j query failed: %s", e)
            return []
        
    def close(self): 
        self._driver.close()

@st.cache_resource
def get_neo4j_connection():
    """Cached Neo4j connection instance"""
    return Neo4jConnection()