import logging, sys

# Xoá toàn bộ handler hiện có
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

# Tạo handler mới ghi ra stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s"))

# Gắn vào root logger, và bật mức DEBUG
root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.addHandler(handler)
logger = logging.getLogger(__name__)
import streamlit as st
from neo4j import GraphDatabase
import ollama
import numpy as np
from functools import lru_cache
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from neo4j import GraphDatabase, WRITE_ACCESS
from neo4j.graph import Node as GraphNode
import time
from datetime import datetime


class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def run_query(self, query, parameters=None):
        try:
            with self._driver.session(
                default_access_mode=WRITE_ACCESS
            ) as session:
                result = session.run(query, parameters or {})
                return [dict(r.items()) for r in result]
        except Exception as e:
            logging.error(f"Neo4j query failed: {e}")
            logging.debug(f"Failed query: {query}")
            logging.debug(f"Parameters: {parameters}")
            return []

    def close(self):
        self.driver.close()

    # def run_query(self, query, parameters=None):
    #     with self.driver.session() as session:
    #         result = session.run(query, parameters or {})
    #         records = [r.data() for r in result]


    #     cleaned = []
    #     for record in records:
    #         clean = {}
    #         for k, v in record.items():
    #             new_key = k.strip().strip('"').strip()
    #             clean[new_key] = v
    #         cleaned.append(clean)

    #     return cleaned
    # def run_query(self, query, parameters=None):
    #     try:
    #         with self.driver.session() as session:
    #             return session.execute_write(
    #                 lambda tx: list(tx.run(query, parameters or {})))
    #     except Exception as e:
    #         logging.error(f"Query failed: {str(e)}")
    #         return []

    def store_embedding(self, node_id, embedding):
        query = """
        MATCH (c:Course {url: $node_id})
        SET c.embedding = $embedding
        """
        self.run_query(query, {"node_id": node_id, "embedding": embedding.tolist()})

# LSTM Embedding Model
class LSTMEmbeddingModel:
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=200):
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.max_length = max_length
        self.model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            LSTM(64, return_sequences=False),
            Dense(embedding_dim, activation='tanh')
        ])

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        self.model.compile(optimizer='adam', loss='mse')
        dummy_labels = np.zeros((len(texts), 128))
        self.model.fit(padded, dummy_labels, epochs=1, verbose=0)

    def get_embedding(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        return self.model.predict(padded, verbose=0)[0]

# Query Processor with Qwen/deepseek
class QueryProcessor:
    def __init__(self, qwen_model):
        self.qwen_model = qwen_model
        # Improved regex pattern to better capture JSON array
        self.json_pattern = re.compile(r'\[\s*{.*}\s*\]', re.DOTALL)

    def extract_json_from_text(self, text):
        """
        Extract a JSON array from the text, handling multiple possible formats and cleaning up the input.
        Args:
            text (str): The raw response text from the Qwen model.
        Returns:
            list: A parsed JSON array containing query steps.
        Raises:
            ValueError: If no valid JSON array found or parsing fails.
        """
        # 1) Tìm ```json[...]``` block
        m = re.search(r'```json\s*(\[\s*{.*?}\s*])\s*```', text, re.DOTALL|re.IGNORECASE)
        if not m:
            # 2) Fallback tìm <json>[…]</json>
            m = re.search(r'<json>\s*(\[\s*{.*?}\s*])\s*</json>', text, re.DOTALL|re.IGNORECASE)
        if not m:
            # 3) Fallback dùng general regex
            m = self.json_pattern.search(text)
        if not m:
            raise ValueError("No valid JSON array found in response")

        json_str = m.group(1)

        # 4) Nếu JSON block dùng single-quotes, normalize; còn không thì giữ nguyên
        if "'" in json_str and '"' not in json_str[:json_str.find('\n')]:
            json_str = (
                json_str
                .replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
            )

        # 5) Parse luôn
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"Failed parsing JSON: {e}")
            logging.error(f"JSON was: {json_str}")
            raise ValueError(f"Invalid JSON format: {e}")

        # 6) (Tuỳ chọn) validate structure
        for idx, step in enumerate(parsed):
            if not isinstance(step, dict):
                raise ValueError(f"Step {idx+1} is not a dict")
            for key in ("description","cypher","return"):
                if key not in step:
                    raise ValueError(f"Step {idx+1} missing required key: {key}")
            if step["return"]=="intermediate" and "bind" not in step:
                raise ValueError(f"Intermediate step {idx+1} missing 'bind'")

        return parsed

    def analyze_query(self, user_query):
        prompt = """
        You are an expert in creating Cypher queries for Neo4j to answer complex, multihop questions about course data. The knowledge graph contains the following nodes and relationships:

        - Nodes:
        - Course: Represents a course in the dataset, with properties like url (unique identifier), name (course title), duration (length in hours), rating (user rating), description (course summary), embedding (vector representation).
        - Skill: Represents a skill taught by a course, with properties like name (e.g., 'AWS Lambda', 'Python').
        - Level: Represents the difficulty level of a course, with properties like name (e.g., 'Beginner', 'Intermediate', 'Advanced').
        - Organization: Represents the organization offering the course, with properties like name (e.g., 'Coursera', 'Udemy').
        - Instructor: Represents the instructor teaching the course, with properties like name (e.g., 'John Doe').
        - Career: Represents a career path, with properties like name (e.g., 'Cloud Computing', 'Data Science').

        - Relationships:
        - TEACHES: Connects Course to Skill (Course -> Skill), indicating the course teaches the skill.
        - HAS_LEVEL: Connects Course to Level (Course -> Level), indicating the course's difficulty level.
        - OFFERED_BY: Connects Course to Organization (Course -> Organization), indicating the course is offered by the organization (to find relationship about organization)
        - TAUGHT_BY: Connects Course to Instructor (Course -> Instructor), indicating the course is taught by the instructor.
        - REQUIRES: Connects Career to Skill (Career -> Skill), indicating the career requires the skill.

        Your task is to analyze the natural language query below and **decompose** it into a **minimal**, **sequential** chain of Cypher sub-queries.  

        1. First output a '<think> ... </think>' block with your step-by-step reasoning in plain English, describing:
        - Which entity or relationship you'll target.
        - Which filters you'll apply.
        - How intermediate results will flow into the next step.

        2. Then, immediately after '</think>', output **only** a JSON array wrapped in ```json and ``` or <json> and </json>. Each element must be an object with exactly these keys:
        - "description": one-sentence summary of this step.
        - "cypher": the full Cypher query.
        - "parameters": a dict of named parameters (use $skill_name, $level, etc.) or refer to prior binds via "$<bind_name>".
        - "return": either "intermediate" or "final candidates".
        - "bind": (only for "intermediate" steps) the name you'll use to pass results forward (e.g. "skill_node", "courses").

        3. **Do not** output any other text outside the <think> block and the JSON.

        4. Escape all literal {{ and }} in the JSON by doubling them ({{/}}) if you plan to call Python's '.format()'. 

        5. Do not put any additional double quotes inside the cypher string—only use them to delimit the JSON value. 

        6. IMPORTANT: For steps combining multiple graph patterns, do NOT use "AND". 
        Use either a single chained pattern `(a)<-[:R1]-(b)-[:R2]->(c)` 
        or separate patterns separated by commas in MATCH.

        7. Do not invent non-existent relationships or labels—only use those listed above.

        8. Do not combine filters with AND inside a single MATCH pattern. Use separate MATCH + WHERE, or chained patterns:

            MATCH (sk:Skill {{name: $skill_name}})
            MATCH (sk)<-[:TEACHES]-(c:Course)
            WHERE c.rating > $min_rating
            RETURN c.url, c.name

        9. Special rule for levels:
            - Detect any of these keywords in the query (case-insensitive):
                Beginner, Introductory, Basics, Fundamentals...
                Intermediate, Mid-level... 
                Advanced, Expert, Hard... 
            - Map the matched keyword to Level.name exactly (“Beginner”, “Intermediate” or “Advanced”).
            - Remove that keyword (and any prepositions like “of”) from the skill phrase.
            - **Normalize the skill name** by converting it to Title Case (capitalize the first letter of each word).
            - Then generate one step to MATCH the Skill{{name: $skill_name}} and one step to MATCH courses with HAS_LEVEL → Level{{name: $level}}.

        10. When returning intermediate results, ALWAYS return primitive values (strings, numbers, lists) - never return full node objects.
        - Correct example: RETURN sk.name AS skill_name
        - Incorrect example: RETURN sk AS skill_node

        11. For MATCH patterns referencing nodes from previous steps, use property values for reference instead of node objects:
        - Instead of: UNWIND $skills AS sk MATCH (sk)<-[:TEACHES]-(c:Course)
        - Use: MATCH (c:Course)-[:TEACHES]->(:Skill {{name: $skill_name}})

        12. Ensure all parameters passed between steps are simple scalar values or lists of primitives - never graph entities (nodes/relationships/paths).

        EXAMPLE IF QUERY JUST ASK ABOUT SKILLS AND LEVELS:
        Example: "Find beginner courses about AWS SageMaker":

        <think>
        First locate the Skill node "AWS SageMaker", then collect all Courses teaching that skill, then filter those by Level = "Beginner".
        </think>

        
        ```json
        [
        {{
            "description": "Locate the AWS SageMaker skill node",
            "cypher": "MATCH (sk:Skill {{name: $skill_name}}) RETURN sk AS skill_node",
            "parameters": {{
            "skill_name": "AWS SageMaker"
            }},
            "return": "intermediate",
            "bind": "skill_node"
        }},
        {{
            "description": "Find all Beginner-level courses teaching that skill",
            "cypher": "MATCH (sk:Skill {{name: $skill_name}})<-[:TEACHES]-(c:Course)-[:HAS_LEVEL]->(l:Level {{name: $level}}) RETURN c.url AS course_url, c.name AS course_title",
            "parameters": {{
            "skill_name": "AWS SageMaker",
            "level": "Beginner"
            }},
            "return": "final candidates"
        }}
        ]
        ```
        EXAMPLE IF QUERY ASKS ABOUT RATING:
        Example: “Find machine learning basics courses with rating > 4.7”
        <think> 1. Recognize “machine learning basics” → skill = “machine learning”, level = “Beginner”. 
        2. Locate the Skill node “machine learning”. 
        3. Find Course nodes teaching that skill. 
        4. Filter those courses by `Level = "Beginner"` via HAS_LEVEL. 
        5. Further filter by `c.rating > 4.7`. 
        6. Return course URL and title. 
        </think>
        ```json

        [
        {{
            "description": "Locate the Machine Learning skill node",
            "cypher": "MATCH (sk:Skill {{name: $skill_name}}) RETURN collect(sk) AS skills",
            "parameters": {{"skill_name": "Machine Learning"}},
            "return": "intermediate",
            "bind": "skills"
        }},
        {{
            "description": "Collect all Course URLs teaching that skill",
            "cypher": "UNWIND $skills AS sk MATCH (sk)<-[:TEACHES]-(c:Course) RETURN collect(c.url) AS course_urls",
            "parameters": {{"skills":"$skills"}},
            "return": "intermediate",
            "bind": "course_urls"
        }},
        {{
            "description": "Filter those courses to Level = Beginner",
            "cypher": "UNWIND $course_urls AS url MATCH (c:Course {{url:url}})-[:HAS_LEVEL]->(l:Level {{name:$level}}) RETURN collect(c.url) AS beginner_urls",
            "parameters": {{"course_urls":"$course_urls","level":"Beginner"}},
            "return": "intermediate",
            "bind": "beginner_urls"
        }},
        {{
            "description": "Filter Beginner courses with rating > 4.7",
            "cypher": "UNWIND $beginner_urls AS url MATCH (c:Course {{url:url}}) WHERE toFloat(c.rating)>$min_rating RETURN c.url AS course_url, c.name AS course_title",
            "parameters": {{"beginner_urls":"$beginner_urls","min_rating":4.7}},
            "return": "final candidates"
        }}
        ]

        ```
        Query: {user_query}
        """
        
        # Generate response from LLM
        resp = ollama.generate(
            model=self.qwen_model,
            prompt=prompt.format(user_query=user_query)
        )

        # Log the entire raw response
        raw = resp["response"]
    
        # Pre-process response
        raw = raw.replace("‘", "'").replace("’", "'")  # Chuẩn hóa dấu quotes
        raw = re.sub(r'\\+', r'\\\\', raw)  # Xử lý backslashes
        raw = raw.replace('\n', ' ')  # Giảm complex formatting
        
        logging.debug("=== Raw Qwen response start ===")
        logging.debug(raw)
        logging.debug("=== Raw Qwen response end ===")

        # Extract JSON using improved method
        try:
            steps = self.extract_json_from_text(raw)
            return steps
        except Exception as e:
            logging.error(f"Error extracting JSON: {e}")
            logging.error(f"Raw response causing error: {raw}")
            raise RuntimeError(f"Failed to parse query steps from LLM response: {e}")

# Knowledge Base QA System
class KnowledgeBaseQA:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, qwen_model, embedding_model):
        self.neo4j_conn = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
        self.query_processor = QueryProcessor(qwen_model)
        self.embedding_model = embedding_model

    def precompute_embeddings(self, courses):
        descriptions = [course.get('description', '') for course in courses]
        self.embedding_model.fit(descriptions)
        for course in courses:
            embedding = self.embedding_model.get_embedding(course.get('description', ''))
            self.neo4j_conn.store_embedding(course.get('url', ''), embedding)

    @lru_cache(maxsize=100)
    # def process_query(self, user_query):
    #     try:
    #         # Analyze query with Qwen
    #         steps = self.query_processor.analyze_query(user_query)
            
    #         if not steps or not isinstance(steps, list):
    #             logging.error(f"Invalid steps format returned: {steps}")
    #             return []
                
    #         intermediate_results = []
    #         candidates = []

    #         # Execute each subquery
    #         for step_idx, step in enumerate(steps):
    #             logging.debug(f"Processing step {step_idx+1}: {step}")
                
    #             # Validate step format
    #             if not isinstance(step, dict) or "cypher" not in step:
    #                 logging.error(f"Invalid step format at index {step_idx}: {step}")
    #                 continue
                    
    #             cypher = step["cypher"]
    #             params = step.get("parameters", {})
    #             step_type = step.get("return", "").lower()
                
    #             logging.debug(f"Executing cypher: {cypher} with params: {params}")
                
    #             if step_type == "intermediate":
    #                 results = self.neo4j_conn.run_query(cypher, params)
    #                 for record in results:
    #                     # Extract various possible return values
    #                     for key in ["skill", "description", "name"]:
    #                         if key in record and record[key]:
    #                             intermediate_results.append(record[key])
    #                             break
                                
    #                 logging.debug(f"Intermediate results: {intermediate_results}")
                    
    #             elif step_type == "candidates":
    #                 # If we have intermediate results, use them in the query
    #                 if intermediate_results and params:
    #                     # Find a parameter that might accept a list of values
    #                     list_param = None
    #                     for param_name, param_value in params.items():
    #                         if isinstance(param_value, list):
    #                             list_param = param_name
    #                             break
                        
    #                     if list_param:
    #                         # Use intermediate results in the list parameter
    #                         params[list_param] = intermediate_results
    #                         results = self.neo4j_conn.run_query(cypher, params)
    #                         for record in results:
    #                             if "course_name" in record and record["course_name"]:
    #                                 candidates.append(record["course_name"])
    #                     else:
    #                         # Process each intermediate result individually
    #                         for item in intermediate_results:
    #                             for param_name in params:
    #                                 params_copy = params.copy()
    #                                 params_copy[param_name] = item
    #                                 results = self.neo4j_conn.run_query(cypher, params_copy)
    #                                 for record in results:
    #                                     if "course_name" in record and record["course_name"]:
    #                                         candidates.append(record["course_name"])
    #                 else:
    #                     # No intermediate results or params, just run the query directly
    #                     results = self.neo4j_conn.run_query(cypher, params)
    #                     for record in results:
    #                         if "course_name" in record and record["course_name"]:
    #                             candidates.append(record["course_name"])
                                
    #                 logging.debug(f"Candidates: {candidates}")

    #         # Rank candidates using embeddings
    #         if candidates:
    #             query_embedding = self.embedding_model.get_embedding(user_query)
    #             similarities = []
    #             unique_candidates = list(set(candidates))
    #             logging.debug(f"Unique candidates to rank: {unique_candidates}")
                
    #             for course_name in unique_candidates:
    #                 description = self.get_course_description(course_name)
    #                 if description:
    #                     course_embedding = self.embedding_model.get_embedding(description)
    #                     sim = np.dot(query_embedding, course_embedding) / (
    #                         np.linalg.norm(query_embedding) * np.linalg.norm(course_embedding) + 1e-8  # Add small epsilon to avoid division by zero
    #                     )
    #                     similarities.append((course_name, sim))
    #                 else:
    #                     # If no description, use a default low similarity
    #                     similarities.append((course_name, 0.0))

    #             similarities.sort(key=lambda x: x[1], reverse=True)
    #             return [x[0] for x in similarities[:5]]
    #         return []
            
    #     except Exception as e:
    #         logging.error(f"Error in process_query: {e}", exc_info=True)
    #         raise

    # 
    
    def process_query(self, user_query):
        """Process query and return formatted results for Streamlit UI"""
        try:
            steps = self.query_processor.analyze_query(user_query)
            if not isinstance(steps, list):
                return ["⚠️ Query analysis error"]

            bind_map = {}
            final_candidates = []

            for step_idx, step in enumerate(steps):
                # Validate step structure
                if not self._validate_step(step, step_idx):
                    return ["⚠️ Invalid query structure"]

                # Process parameters with value binding
                processed_params = {}
                for param_name, param_value in step.get("parameters", {}).items():
                    if isinstance(param_value, str) and param_value.startswith("$"):
                        bind_key = param_value[1:]
                        if bind_key not in bind_map:
                            raise ValueError(f"Missing parameter: {bind_key}")
                        processed_params[param_name] = bind_map[bind_key]
                    else:
                        processed_params[param_name] = param_value

                # Execute Cypher query
                records = self.neo4j_conn.run_query(step["cypher"], processed_params)
                
                if not records:
                    logging.warning(f"Step {step_idx+1} returned empty results")
                    continue

                # Handle intermediate results
                if step["return"].lower() == "intermediate":
                    bind_key = step["bind"]
                    if len(records) == 1 and len(records[0]) == 1:
                        bind_map[bind_key] = list(records[0].values())[0]
                    else:
                        bind_map[bind_key] = [item for r in records for item in r.values()]

                # Handle final results
                elif step["return"].lower() == "final candidates":
                    for record in records:
                        course_entry = self._format_course_entry(record)
                        if course_entry not in final_candidates:
                            final_candidates.append(course_entry)

            return [
                {
                    "title": record.get("title"),
                    "url": record.get("url"),
                    "rating": record.get("rating", 0.0),
                    "level": record.get("level", "Unknown"),
                    "skills": record.get("skills", [])
                }
                for record in final_candidates
            ][:10]

        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return []


    def _format_course_entry(self, record):
        """Format results for Streamlit UI"""
        return {
            "title": record.get("course_title", "Untitled Course"),
            "url": record.get("course_url", "#"),
            "rating": record.get("rating", 0.0),
            "skills": record.get("skills", []),
            "level": record.get("level", "Unknown")
        }

    def _validate_step(self, step, step_idx):
        """Validate query step structure"""
        required_keys = ["description", "cypher", "return"]
        if not all(k in step for k in required_keys):
            logging.error(f"Step {step_idx+1} missing keys: {step}")
            return False
        if step["return"] == "intermediate" and "bind" not in step:
            logging.error(f"Missing bind in step {step_idx+1}")
            return False
        return True


    def _process_parameters(self, step, intermediate):
        processed = {}
        for k, v in step.get("parameters", {}).items():
            # Nếu tham số là placeholder "$xxx", lấy binding từ intermediate
            if isinstance(v, str) and v.startswith('$'):
                key = v[1:]
                if key in intermediate:
                    processed[k] = intermediate[key]
                else:
                    logging.warning(f"Missing intermediate bind for {key}")
            else:
                processed[k] = v
        return processed


    def _execute_cypher(self, cypher, params, step_idx):
        try:
            return self.neo4j_conn.run_query(cypher, params)
        except Exception as e:
            logging.error(f"Query failed at step {step_idx+1}: {e}")
            logging.debug(f"Failed query: {cypher}")
            logging.debug(f"Parameters keys: {list(params.keys())}")
            return None

    def _handle_intermediate(self, step, records, intermediate):
        bind_key = step["bind"]
        if records and bind_key in records[0]:
            node = records[0][bind_key]
            
            # Handle different return types
            if isinstance(node, GraphNode):
                if 'Skill' in node.labels:
                    intermediate[bind_key] = node.get('name')
                elif 'Course' in node.labels:
                    intermediate[bind_key] = node.get('url')
            elif isinstance(node, dict):
                intermediate[bind_key] = node.get('name') or node.get('url')
            else:
                intermediate[bind_key] = node

    def _collect_candidates(self, records, candidates):
        for record in records:
            if 'course_title' in record and 'course_url' in record:
                candidates.append((
                    record['course_title'],
                    record['course_url']
                ))

    def _rank_results(self, candidates, query):
        if not candidates:
            return []

        unique_courses = list(set(candidates))
        
        # Batch processing for efficiency
        descriptions = {
            title: self.get_course_description(title) 
            for title, _ in unique_courses
        }
        
        query_embedding = self.embedding_model.get_embedding(query)
        course_embeddings = {
            title: self.embedding_model.get_embedding(desc) if desc 
            else np.zeros(self.embedding_model.model.output_shape[1])
            for title, desc in descriptions.items()
        }
        
        # Calculate similarities
        similarities = []
        for title, url in unique_courses:
            emb = course_embeddings[title]
            norm = np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8
            sim = np.dot(query_embedding, emb) / norm
            similarities.append((title, url, sim))
        
        # Return formatted results
        return [
            f"{title} ({url})" 
            for title, url, _ in sorted(
                similarities, 
                key=lambda x: x[2], 
                reverse=True
            )[:5]
        ]

    def get_course_description(self, course_name):
        query = "MATCH (c:Course {name: $name}) RETURN c.description AS description"
        result = self.neo4j_conn.run_query(query, {"name": course_name})
        if result and "description" in result[0]:
            return result[0]["description"]
        return ""

    def close(self):
        self.neo4j_conn.close()

# if __name__ == "__main__":
#     qa = KnowledgeBaseQA(
#         neo4j_uri="bolt://localhost:7687",
#         neo4j_user="neo4j",
#         neo4j_password="12345678",
#         #qwen_model="deepseek-r1:7b",
#         qwen_model= "qwen3:4b",
#         embedding_model=LSTMEmbeddingModel()
#     )
#     try:
#         while True:
#             # Show prompt and read user input
#             user_query = input("Enter your query (e.g., Find intermediate courses about AWS Lambda): ")
#             if not user_query.strip():
#                 print("Exiting.")
#                 break
#             logging.info(f"User query: {user_query}")
#             try:
#                 results = qa.process_query(user_query)
#                 if results:
#                     print("Top matching courses:")
#                     for idx, course in enumerate(results, 1):
#                         print(f"{idx}. {course}")
#                 else:
#                     print("No matching courses found.")
#             except Exception as e:
#                 logging.error(f"Error processing query: {e}", exc_info=True)
#                 print(f"An error occurred: {e}")
#     finally:
#         qa.close()

# ====================== Streamlit UI Integration ======================
def render_courses(courses):
    """Display course cards with rich formatting"""
    # if not courses:
    #     st.warning("No matching courses found")
    #     return
    
    # for course in courses:
    #     with st.container(border=True):
    #         cols = st.columns([4, 1])
    #         with cols[0]:
    #             st.markdown(f"### [{course['title']}]({course['url']})")
    #             st.caption(f"**Level:** {course['level']} | **Rating:** {course['rating']}/5.0")
                
    #             # if course['skills']:
    #             #     st.markdown("**Skills:** " + ", ".join(course['skills']))
                    
    #         with cols[1]:
    #             if st.button("ℹ️ Details", key=f"btn_{course['url']}"):
    #                 show_course_details(course['url'])
    #st.subheader("🔍 Raw Qwen JSON steps / results")
    st.json(courses)
    return
def show_course_details(url):
    """Display course details popover"""
    query = """
    MATCH (c:Course {url: $url})
    OPTIONAL MATCH (c)-[:TEACHES]->(s:Skill)
    OPTIONAL MATCH (c)-[:HAS_LEVEL]->(l:Level)
    RETURN c.name AS name, 
           c.description AS description,
           c.rating AS rating,
           collect(DISTINCT s.name) AS skills,
           l.name AS level
    """
    result = st.session_state.advisor.neo4j_conn.run_query(query, {"url": url})
    
    if result:
        details = result[0]
        with st.popover(f"📚 {details['name']}", use_container_width=True):
            st.markdown(f"**Description:** {details.get('description', 'N/A')}")
            st.divider()
            
            cols = st.columns(3)
            cols[0].metric("Rating", f"{details.get('rating', 0):.1f}/5.0")
            cols[1].metric("Level", details.get('level', 'Unknown'))
            cols[2].metric("Skills", len(details.get('skills', [])))
            
            if details.get('skills'):
                st.markdown("**Related Skills:**")
                st.write(", ".join(details['skills']))

def handle_query_submission(query: str):
    """Handle query submission and UI updates"""
    user_msg = {
        "role": "user",
        "type": "text",
        "content": query,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.messages.append(user_msg)
    
    try:
        with st.spinner("🔄 Processing..."):
            start_time = time.time()
            
            # Execute query
            results = st.session_state.advisor.process_query(query)
            
            # Format system response
            system_msg = {
                "role": "assistant",
                "type": "courses" if results else "text",
                "content": results if results else "No matching courses found",
                "metadata": {
                    "processing_time": f"{time.time()-start_time:.2f}s",
                    "query_type": "course_search"
                }
            }
            st.session_state.messages.append(system_msg)
            
            # Auto-refresh UI
            st.rerun()
            
    except Exception as e:
        handle_processing_error(e)

def handle_processing_error(error: Exception):
    """Display error messages in UI"""
    error_msg = {
        "role": "assistant",
        "type": "text",
        "content": f"⚠️ System error: {type(error).__name__}",
        "metadata": {
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }
    }
    st.session_state.messages.append(error_msg)
def render_sidebar_settings():
    """User settings panel"""
    with st.sidebar.expander("⚙️ SETTINGS", expanded=True):
        current_theme = st.selectbox(
            "Theme",
            ["Light", "Dark", "System"],
            index=2,
            help="Select display theme"
        )
        
        
        st.caption(f"Version: 1.0.0 | Mode: {current_theme}")

def display_chat_history():
    """Display chat message history"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=("👤" if msg["role"] == "user" else "🤖")):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "courses":
                render_courses(msg["content"])
            
            if "metadata" in msg:
                with st.expander("Technical details"):
                    st.json(msg["metadata"])

def show_quick_actions():
    """Sample query quick action buttons"""
    with st.container(border=True):
        st.markdown("**🚀 Quick Queries:**")
        cols = st.columns(3)
        sample_queries = [
            ("Python Basics", "Find beginner Python courses"),
            ("AWS Advanced", "Show advanced cloud computing courses with AWS"),
            ("Top Data Science", "Data science courses with rating > 4.5")
        ]
        
        for col, (title, query) in zip(cols, sample_queries):
            if col.button(title, help=query, use_container_width=True):
                handle_query_submission(query)

def process_chat_input():
    """Handle main chat input"""
    if prompt := st.chat_input("Ask about courses..."):
        handle_query_submission(prompt)

def main():
    st.set_page_config(
        page_title="EduAssistant", 
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS styling
    st.markdown("""
    <style>
    .course-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        transition: transform 0.2s;
        background: var(--background-color);
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .course-card:hover {
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "advisor" not in st.session_state:
        st.session_state.advisor = KnowledgeBaseQA(
            neo4j_uri="bolt://localhost:7687?address_resolver=ipv4",
            neo4j_user="neo4j",
            neo4j_password="12345678",
            qwen_model="deepseek-r1:7b",
            embedding_model=LSTMEmbeddingModel()
        )

    # Render UI components
    render_sidebar_settings()
    show_quick_actions()
    display_chat_history()
    process_chat_input()
    # Auto-scroll to bottom
    if st.session_state.messages:
        container = st.container()
        with container:
            js = """
            <script>
            window.addEventListener('DOMContentLoaded', function() {
                var messages = document.querySelector('.stChatMessage');
                messages.scrollTop = messages.scrollHeight;
            });
            </script>
            """
            st.components.v1.html(js, height=0)


if __name__ == "__main__":
    main()
                