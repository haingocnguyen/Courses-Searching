# import json
# import numpy as np
# import pandas as pd
# import networkx as nx
# from sentence_transformers import SentenceTransformer
# import faiss
# from typing import List, Dict, Tuple, Optional, Any, Union
# from datetime import datetime
# import ollama
# import re
# import traceback


# # ======================
# # 1. Knowledge Base Module
# # ======================
# class CourseKnowledgeBase:
#     def __init__(self, data_path: str):
#         self.data = pd.read_json(data_path)
#         self._preprocess()
#         self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
#         self._build_indices()
#         self._construct_knowledge_graph()
#         self.operation_log = []

#     def _preprocess(self):
#         """Robust data preprocessing with null handling"""
#         # Extract level with fallbacks
#         self.data['level'] = self.data.get('learning_path', {}).apply(
#             lambda x: x.get('suitable_for', ['Unknown'])[0] 
#             if isinstance(x, dict) and x.get('suitable_for')
#             else 'Unknown'
#         )

#         # Ensure teaches/prerequisites are always lists
#         def safe_get_list(field):
#             return self.data['knowledge_requirements'].apply(
#                 lambda x: x.get(field, []) 
#                 if isinstance(x, dict) 
#                 else []
#             ).apply(lambda y: y if isinstance(y, list) else [])

#         self.data['teaches'] = safe_get_list('teaches')
#         self.data['prerequisites'] = safe_get_list('prerequisites')

#         # Create search text safely
#         self.data['search_text'] = self.data.apply(
#             lambda row: ' '.join(filter(None, [
#                 str(row.get('title', '')),
#                 str(row.get('description', '')),
#                 ' '.join(map(str, row['teaches']))
#             ])), axis=1
#         )
        
#         # Ensure numeric fields are properly converted
#         self.data['duration_months'] = pd.to_numeric(
#             self.data['duration_months'],
#             errors='coerce'
#         ).fillna(0).astype(int)
        
#         # Add level as lowercase for easier matching
#         self.data['level_lower'] = self.data['level'].str.lower()
        
#         self._log_sample_data()

#     def _log_sample_data(self):
#         """Log sample data for inspection"""
#         sample = self.data.head(2).to_dict("records")
#         print("\n=== SAMPLE COURSE DATA ===")
#         for c in sample:
#             print(f"Course: {c['title']}")
#             print(f"Teaches: {c['teaches'][:3]}...")
#             print(f"Prereqs: {c['prerequisites'][:3]}...")
#             print(f"Duration: {c['duration_months']} months")
#             print(f"Level: {c['level']}\n")

#     def _build_indices(self):
#         """Create search indices with robust error handling"""
#         # Semantic index
#         text_embeddings = self.encoder.encode(self.data['search_text'].tolist())
#         self.semantic_index = faiss.IndexFlatIP(text_embeddings.shape[1])
#         self.semantic_index.add(text_embeddings.astype('float32'))
        
#         # Attribute indices
#         self.indices = {
#             'level': self._build_attribute_index('level_lower'),
#             'category': self._build_category_index(),
#             'duration': self._build_duration_index()  # NEW: Duration index
#         }
        
#         # Build skill index for faster lookup - NEW
#         self.skill_index = self._build_skill_index()

#     def _build_attribute_index(self, field: str) -> Dict:
#         """Safe attribute index builder"""
#         index = {}
#         if field in self.data.columns:
#             for val, group in self.data.groupby(field):
#                 key = str(val).lower()
#                 index[key] = set(group.index)
#         else:
#             print(f"Warning: Missing column '{field}' for indexing")
#         return index

#     def _build_category_index(self) -> Dict:
#         """Safe category index builder"""
#         index = {}
#         for idx, row in self.data.iterrows():
#             # Handle potential None values
#             category = str(row.get('category', 'Unknown')).lower()
#             sub_category = str(row.get('sub_category', 'Unknown')).lower()
            
#             # Ensure teaches is always iterable
#             teaches = row['teaches'] if isinstance(row['teaches'], list) else []
            
#             sources = [category, sub_category] + [str(t).lower() for t in teaches]
            
#             for val in sources:
#                 if val:  # Skip empty strings
#                     index.setdefault(val, set()).add(idx)
#         return index
    
#     def _build_duration_index(self) -> Dict:
#         """Build index for duration-based queries"""
#         index = {}
#         duration_ranges = [(0, 1), (0, 3), (0, 6), (0, 12), (0, 24)]
        
#         for start, end in duration_ranges:
#             index[f"{start}-{end}"] = set(
#                 self.data[
#                     (self.data['duration_months'] >= start) & 
#                     (self.data['duration_months'] <= end)
#                 ].index
#             )
#         return index
    
#     def _build_skill_index(self) -> Dict:
#         """Build index for skill lookups"""
#         skill_index = {}
        
#         # Process teaches field
#         for idx, row in self.data.iterrows():
#             for skill in row['teaches']:
#                 if not skill:
#                     continue
                    
#                 # Normalize skill names for better matching
#                 norm_skill = self._normalize_skill(skill)
#                 if norm_skill:
#                     skill_index.setdefault(norm_skill, set()).add(idx)
                    
#                     # Add individual words for partial matching
#                     words = re.findall(r'\w+', norm_skill)
#                     for word in words:
#                         if len(word) > 3:  # Only index meaningful words
#                             skill_index.setdefault(f"word:{word.lower()}", set()).add(idx)
        
#         return skill_index
    
#     def _normalize_skill(self, skill: str) -> str:
#         """Normalize skill names for consistent lookup"""
#         if not skill:
#             return ""
            
#         # Convert to string, lowercase, and remove extra whitespace
#         return re.sub(r'\s+', ' ', str(skill).lower().strip())

#     def _construct_knowledge_graph(self):
#         """Null-safe knowledge graph construction"""
#         self.graph = nx.DiGraph()
        
#         # Track added nodes to avoid duplicates
#         skill_nodes = set()
        
#         for idx, row in self.data.iterrows():
#             course_id = row['course_id']
#             course_data = row.to_dict()
            
#             # Make sure to convert numpy/pandas types to Python types
#             for k, v in course_data.items():
#                 if isinstance(v, (np.integer, np.floating)):
#                     course_data[k] = v.item()
#                 elif isinstance(v, np.ndarray):
#                     course_data[k] = v.tolist()
            
#             self.graph.add_node(course_id, type='course', **course_data)

#             # Add teaches relationships
#             for skill in row['teaches']:
#                 if skill and pd.notna(skill):
#                     # Normalize skill name
#                     normalized_skill = self._normalize_skill(skill)
#                     skill_node = f"skill:{normalized_skill}"
                    
#                     if skill_node not in skill_nodes:
#                         self.graph.add_node(skill_node, type='skill', name=normalized_skill)
#                         skill_nodes.add(skill_node)
                        
#                     self.graph.add_edge(course_id, skill_node, relation='teaches')

#             # Add prerequisites
#             for prereq in row['prerequisites']:
#                 if prereq and pd.notna(prereq):
#                     normalized_prereq = self._normalize_skill(prereq)
#                     prereq_node = f"skill:{normalized_prereq}"
                    
#                     if prereq_node not in skill_nodes:
#                         self.graph.add_node(prereq_node, type='skill', name=normalized_prereq)
#                         skill_nodes.add(prereq_node)
                        
#                     self.graph.add_edge(prereq_node, course_id, relation='prerequisite')
        
#         # Add skill similarity edges to make graph more connected
#         self._add_skill_similarity_edges()
        
#         print(f"Knowledge graph constructed with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
    
#     def _add_skill_similarity_edges(self):
#         """Add edges between similar skills to improve connectivity"""
#         # Get all skill nodes
#         skill_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'skill']
        
#         if len(skill_nodes) < 2:
#             return
            
#         # Get skill names
#         skill_names = [self.graph.nodes[n].get('name', '') for n in skill_nodes]
        
#         # Compute embeddings for skill names
#         try:
#             embeddings = self.encoder.encode(skill_names)
            
#             # Compute similarity matrix
#             similarity = np.dot(embeddings, embeddings.T)
            
#             # Add edges for high similarity skills
#             threshold = 0.7
#             for i in range(len(skill_nodes)):
#                 for j in range(i+1, len(skill_nodes)):
#                     if similarity[i, j] > threshold:
#                         self.graph.add_edge(
#                             skill_nodes[i], 
#                             skill_nodes[j], 
#                             relation='similar', 
#                             weight=float(similarity[i, j])
#                         )
#         except Exception as e:
#             print(f"Error computing skill similarities: {e}")

#     def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
#         """Safe semantic search with empty handling"""
#         if self.semantic_index is None or len(self.data) == 0:
#             return []
        
#         # Enhanced query with fallback
#         try:
#             query_embed = self.encoder.encode(query)
#             if query_embed.ndim == 1:
#                 query_embed = np.expand_dims(query_embed, 0)
            
#             # Check embedding dimensions match index
#             if query_embed.shape[1] != self.semantic_index.d:
#                 return []
            
#             scores, indices = self.semantic_index.search(query_embed.astype('float32'), top_k)
            
#             # Filter out very low relevance results
#             good_indices = [idx for score, idx in zip(scores[0], indices[0]) if score > 0.2]
            
#             if not good_indices:
#                 return []
                
#             results = self.data.iloc[good_indices].to_dict('records')
            
#             # Add score to results
#             for i, result in enumerate(results):
#                 result['relevance_score'] = float(scores[0][i])
            
#             return results
            
#         except Exception as e:
#             print(f"Semantic search error: {e}")
#             # Fallback to keyword search
#             return self._keyword_fallback_search(query, top_k)
    
#     def _keyword_fallback_search(self, query: str, top_k: int = 10) -> List[Dict]:
#         """Fallback to simple keyword matching when semantic search fails"""
#         try:
#             # Extract keywords
#             keywords = set(re.findall(r'\w+', query.lower()))
            
#             # Filter stopwords
#             stopwords = {'the', 'a', 'an', 'in', 'for', 'of', 'with', 'and', 'or', 'to'}
#             keywords = {k for k in keywords if k not in stopwords and len(k) > 2}
            
#             # Score courses by keyword matches
#             scores = []
#             for idx, row in self.data.iterrows():
#                 text = row['search_text'].lower()
#                 score = sum(1 for k in keywords if k in text)
#                 scores.append((score, idx))
            
#             # Get top matches
#             top_indices = [idx for _, idx in sorted(scores, reverse=True)[:top_k] if _ > 0]
            
#             if not top_indices:
#                 return []
                
#             return self.data.iloc[top_indices].to_dict('records')
#         except Exception as e:
#             print(f"Keyword search error: {e}")
#             return []
    
#     def skill_search(self, skill: str) -> List[Dict]:
#         """Enhanced skill search with partial matching and graph traversal"""
#         norm_skill = self._normalize_skill(skill)
#         result_indices = set()
        
#         # Try exact match first
#         if norm_skill in self.skill_index:
#             result_indices.update(self.skill_index[norm_skill])
        
#         # Try graph-based search
#         skill_node = f"skill:{norm_skill}"
#         if self.graph.has_node(skill_node):
#             # Get courses that teach this skill
#             for course_node in self.graph.predecessors(skill_node):
#                 if course_node in self.data['course_id'].values:
#                     idx = self.data[self.data['course_id'] == course_node].index[0]
#                     result_indices.add(idx)
            
#             # Get similar skills and their courses
#             for neighbor in nx.neighbors(self.graph, skill_node):
#                 if neighbor.startswith('skill:'):
#                     for course_node in self.graph.predecessors(neighbor):
#                         if course_node in self.data['course_id'].values:
#                             idx = self.data[self.data['course_id'] == course_node].index[0]
#                             result_indices.add(idx)
        
#         # If still no results, try partial word matching
#         if not result_indices:
#             words = re.findall(r'\w+', norm_skill)
#             for word in words:
#                 if len(word) > 3:  # Only use meaningful words
#                     word_key = f"word:{word.lower()}"
#                     if word_key in self.skill_index:
#                         result_indices.update(self.skill_index[word_key])
        
#         # Convert results to records
#         if result_indices:
#             return self.data.iloc[list(result_indices)].to_dict('records')
        
#         # Last resort: semantic similarity
#         return self.semantic_search(skill)


# # ======================
# # 2. Reasoning Components
# # ======================
# class QueryAnalyzer:
#     def __init__(self, llm_client):
#         self.llm = llm_client
#         self.retries = 3

#     def parse_query(self, query: str) -> Dict:
#         """Decompose query into structured operations"""
#         schema = {
#             "type": "object",
#             "properties": {
#                 "semantic_terms": {"type": "array", "items": {"type": "string"}},
#                 "filters": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "type": {"type": "string", "enum": ["attribute", "skill"]},
#                             "field": {"type": "string"},
#                             "value": {"type": "string"}
#                         },
#                         "required": ["type", "value"]
#                     }
#                 },
#                 "constraints": {
#                     "type": "object",
#                     "properties": {
#                         "duration": {"type": "number"},
#                         "level": {"type": "string"},
#                         "exclude_prerequisites": {"type": "array", "items": {"type": "string"}}
#                     }
#                 }
#             },
#             "required": ["semantic_terms", "filters"]
#         }

#         prompt = f"""Analyze this course search query and output JSON matching this schema:
#             {json.dumps(schema, indent=2)}
#             \nNote: Assume duration values without units are in months

#             Example 1:
#             Query: "Machine learning without math prerequisites"
#             Output: {{
#                 "semantic_terms": ["machine learning"],
#                 "filters": [
#                     {{"type": "skill", "value": "machine learning"}}
#                 ],
#                 "constraints": {{
#                     "exclude_prerequisites": ["math"]
#                 }}
#             }}

#             Example 2:
#             Query: "Web development under 4 months"
#             Output: {{
#                 "semantic_terms": ["web development"],
#                 "filters": [
#                     {{"type": "skill", "value": "web development"}},
#                     {{"type": "attribute", "field": "duration", "value": "4"}}
#                 ]
#             }}
            
#             Example 3:
#             Query: "Advanced Python courses with data science focus under 8 months"
#             Output: {{
#                 "semantic_terms": ["python", "data science"],
#                 "filters": [
#                     {{"type": "skill", "value": "python"}},
#                     {{"type": "skill", "value": "data science"}},
#                     {{"type": "attribute", "field": "duration", "value": "8"}}
#                 ],
#                 "constraints": {{
#                     "level": "advanced"
#                 }}
#             }}

#             Query: {query}
#             """
        
#         for attempt in range(self.retries):
#             try:
#                 response = ollama.generate(
#                     model="qwen2.5:1.5b",
#                     prompt=prompt,
#                     format="json",
#                     options={"temperature": 0.2}
#                 )
#                 result = json.loads(response['response'])
                
#                 # Ensure result has required keys
#                 if 'semantic_terms' not in result:
#                     result['semantic_terms'] = []
#                 if 'filters' not in result:
#                     result['filters'] = []
#                 if 'constraints' not in result:
#                     result['constraints'] = {}
                    
#                 return result
                
#             except Exception as e:
#                 print(f"Parse attempt {attempt+1} failed: {str(e)}")
                
#         # Fallback parsing if LLM fails
#         return self._fallback_parse(query)
    
#     def _fallback_parse(self, query: str) -> Dict:
#         """Simple rule-based fallback parser"""
#         result = {"semantic_terms": [], "filters": [], "constraints": {}}
        
#         # Extract duration constraints
#         duration_match = re.search(r'under\s+(\d+)\s*(month|months|mo)?', query)
#         if duration_match:
#             months = int(duration_match.group(1))
#             result["filters"].append({
#                 "type": "attribute",
#                 "field": "duration",
#                 "value": str(months)
#             })
#             result["constraints"]["duration"] = months
        
#         # Extract level
#         level_words = {
#             "beginner": "beginner",
#             "beginner friendly": "beginner",
#             "novice": "beginner",
#             "intermediate": "intermediate",
#             "advanced": "advanced",
#             "expert": "advanced"
#         }
        
#         for level_term, level_value in level_words.items():
#             if level_term in query.lower():
#                 result["constraints"]["level"] = level_value
#                 break
        
#         # Extract main topics as skills
#         common_words = {"course", "courses", "with", "without", "under", "over", 
#                       "about", "focus", "focused", "month", "months", "and", "the"}
        
#         words = query.lower().split()
#         potential_skills = []
        
#         i = 0
#         while i < len(words):
#             if words[i] not in common_words and len(words[i]) > 2:
#                 # Try to find multi-word skills
#                 for j in range(min(3, len(words) - i), 0, -1):
#                     skill = " ".join(words[i:i+j])
#                     if j > 1:  # Multi-word skills are more likely to be actual skills
#                         potential_skills.append(skill)
#                         i += j - 1
#                         break
#                     elif j == 1:  # Single words are added with lower confidence
#                         potential_skills.append(skill)
#             i += 1
        
#         # Add top skills as filters and semantic terms
#         for skill in potential_skills[:3]:  # Limit to top 3 skills
#             result["filters"].append({
#                 "type": "skill",
#                 "value": skill
#             })
#             result["semantic_terms"].append(skill)
        
#         return result


# class CourseValidator:
#     def __init__(self, knowledge_graph):
#         self.kg = knowledge_graph

#     def validate_course(self, course_id: str, query: str) -> Dict:
#         """Perform multi-aspect validation"""
#         if course_id not in self.kg.nodes:
#             return {"valid": False, "errors": ["Course not found"]}
            
#         course = self.kg.nodes[course_id]
#         results = {
#             "prerequisites": self._check_prerequisites(course),
#             "recency": self._check_recency(course),
#             "rating": self._check_rating(course),
#             "relevance": self._check_relevance(course, query)
#         }
#         results["valid"] = all(v for k, v in results.items())
#         return results

#     def _check_prerequisites(self, course: Dict) -> bool:
#         """Check if prerequisites exist in knowledge graph"""
#         if not course.get('prerequisites'):
#             return True  # No prerequisites is always valid
            
#         return all(
#             self.kg.has_node(f"skill:{str(p).lower().strip()}") 
#             for p in course.get('prerequisites', [])
#             if isinstance(p, str)
#         )

#     def _check_recency(self, course: Dict) -> bool:
#         """Check if course is relatively recent"""
#         current_year = datetime.now().year
#         course_year = course.get('year', current_year - 3)  # Default to 3 years old
#         return (current_year - course_year) <= 10  # More lenient check (10 years)

#     def _check_rating(self, course: Dict) -> bool:
#         """Check if course has acceptable rating"""
#         return course.get('rating', 0) >= 1.0  # More lenient check (1.0+)

#     def _check_relevance(self, course: Dict, query: str) -> bool:
#         """More lenient relevance check"""
#         query_terms = set(re.findall(r'\w+', query.lower()))
#         course_terms = set(re.findall(r'\w+', course.get('title', '').lower())) | \
#                       set(re.findall(r'\w+', course.get('description', '').lower())) | \
#                       set(str(t).lower() for t in course.get('teaches', []))
        
#         # Remove common words
#         stopwords = {'course', 'courses', 'the', 'and', 'with', 'for', 'in', 'on', 'of', 'to'}
#         query_terms = {t for t in query_terms if t not in stopwords and len(t) > 2}
        
#         # Even more lenient check - just one matching term
#         return len(query_terms & course_terms) >= 1


# # ======================
# # 3. Multi-Hop Search System
# # ======================
# class HiragSearchEngine:
#     def __init__(self, data_path: str):
#         self.kb = CourseKnowledgeBase(data_path)
#         self.validator = CourseValidator(self.kb.graph)
#         self.analyzer = QueryAnalyzer(ollama.Client())
        
#     def execute_search(self, query: str) -> Dict:
#         """Main search execution function with robust error handling"""
#         result = {
#             "query": query,
#             "analysis": {},
#             "hops": [],
#             "results": [],
#             "errors": [],
#             "logs": [],
#             "debug": {}
#         }
        
#         try:
#             # 1. Analyze query
#             analysis = self.analyzer.parse_query(query)
#             result["analysis"] = analysis
#             self._log_debug(result, "01_raw_analysis", analysis)
            
#             # 2. Log knowledge base state
#             self._log_debug(result, "02_kb_snapshot", {
#                 "total_courses": len(self.kb.data),
#                 "sample_skills": list(n for n, d in self.kb.graph.nodes(data=True) 
#                                      if d.get('type') == 'skill')[:5],
#                 "data_columns": list(self.kb.data.columns)
#             })

#             # 3. Execute search hops
#             all_hop_results = []
#             for idx, hop in enumerate(analysis.get("filters", [])):
#                 hop_result = self._execute_hop(hop, result)
#                 result["hops"].append(hop_result)
                
#                 # Track non-empty hop results
#                 if hop_result.get("results"):
#                     all_hop_results.extend(hop_result["results"])
                
#                 # Log hop details
#                 self._log_debug(result, f"03_hop_{idx}_details", {
#                     "hop_type": hop.get("type"),
#                     "hop_value": hop.get("value"),
#                     "results_count": len(hop_result.get("results", [])),
#                     "sample_results": [c.get("title", "Unknown") for c in hop_result.get("results", [])[:3]]
#                 })

#             # 4. Integrate results from all hops
#             candidates = self._integrate_results(all_hop_results, analysis, result)
            
#             # 5. Log candidate metrics
#             self._log_debug(result, "04_candidates", {
#                 "count": len(candidates),
#                 "sample_titles": [c.get("title", "Unknown") for c in candidates[:3]]
#             })

#             # 6. Final validation, ranking and limiting
#             final_results = self._validate_and_rank(candidates, analysis, query, result)
#             result["results"] = final_results
            
#             # 7. Log final results
#             self._log_debug(result, "05_final_results", {
#                 "count": len(final_results),
#                 "sample": [c.get("title", "Unknown") for c in final_results[:3]]
#             })
            
#             # 8. Add status summary
#             result["summary"] = {
#                 "total_hits": len(final_results),
#                 "status": "success" if final_results else "no_results"
#             }

#         except Exception as e:
#             result["errors"].append({
#                 "message": f"Search engine error: {str(e)}",
#                 "traceback": traceback.format_exc()
#             })
#             result["summary"] = {"status": "error", "message": str(e)}
            
#         return result

#     def _execute_hop(self, hop: Dict, result: Dict) -> Dict:
#         """Execute a search hop with robust error handling"""
#         hop_result = {
#             "type": hop.get("type", "unknown"),
#             "parameters": hop,
#             "results": [],
#             "metrics": {"results_found": 0},
#             "error": None
#         }
        
#         try:
#             # Execute different hop types
#             if hop["type"] == "attribute":
#                 results = self._execute_attribute_hop(hop)
#             elif hop["type"] == "skill":
#                 results = self._execute_skill_hop(hop)
#             else:
#                 results = []
                
#             hop_result["results"] = results
#             hop_result["metrics"]["results_found"] = len(results)
            
#             self._log(result, "HopExecuted", 
#                      f"{hop['type']} hop for '{hop.get('value')}' found {len(results)} results")
            
#         except Exception as e:
#             error_msg = f"Hop execution failed: {str(e)}"
#             hop_result["error"] = error_msg
#             self._log_error(result, error_msg, {"hop": hop})
            
#             # Try fallback for failed hops
#             try:
#                 if hop["type"] == "skill":
#                     # Fallback to semantic search
#                     hop_result["results"] = self.kb.semantic_search(hop.get("value", ""))
#                     hop_result["metrics"]["results_found"] = len(hop_result["results"])
#                     hop_result["metrics"]["used_fallback"] = True
#                     self._log(result, "HopFallback", 
#                              f"Used semantic fallback for {hop['type']} hop")
#             except Exception as fallback_error:
#                 self._log_error(result, f"Fallback also failed: {str(fallback_error)}")
            
#         return hop_result

#     def _execute_attribute_hop(self, hop: Dict) -> List[Dict]:
#         """Enhanced attribute handling with fuzzy matching"""
#         field = hop.get("field", "").lower()
#         value = hop.get("value", "").lower()
        
#         # Duration filter
#         if field == "duration":
#             try:
#                 months = int("".join(filter(str.isdigit, value)))
#                 results = self.kb.data[
#                     self.kb.data["duration_months"] <= months
#                 ].to_dict("records")
#                 return results
#             except Exception as e:
#                 print(f"Duration filter error: {e}")
#                 # Try using the duration index instead
#                 duration_key = next((k for k in self.kb.indices.get('duration', {}) 
#                                    if k.split('-')[1] == str(months)), None)
#                 if duration_key and duration_key in self.kb.indices.get('duration', {}):
#                     indices = list(self.kb.indices['duration'][duration_key])
#                     return self.kb.data.iloc[indices].to_dict("records")
#                 return []
        
#         # Level filter
#         if field == "level":
#             # Fuzzy level matching
#             level_terms = {
#                 "beginner": ["beginner", "novice", "starter", "introductory", "basic"],
#                 "intermediate": ["intermediate", "medium"],
#                 "advanced": ["advanced", "expert", "professional", "experienced"]
#             }
            
#             # Find matching level category
#             target_level = None
#             for level, terms in level_terms.items():
#                 if value in terms:
#                     target_level = level
#                     break
            
#             if target_level:
#                 # Try both exact and partial matching
#                 pattern = re.compile(fr'\b{re.escape(target_level)}', re.IGNORECASE)
#                 matches = self.kb.data[
#                     self.kb.data["level"].str.contains(pattern, na=False)
#                 ]
#                 return matches.to_dict("records")
        
#         return []

#     def _execute_skill_hop(self, hop: Dict) -> List[Dict]:
#         """Enhanced knowledge graph-based skill search"""
#         skill = hop.get("value", "").lower().strip()
        
#         # Use the enhanced skill search
#         results = self.kb.skill_search(skill)
        
#         # If no results, try to extract keywords and search again
#         if not results:
#             # Extract keywords from skill
#             keywords = re.findall(r'\w+', skill)
#             for keyword in keywords:
#                 if len(keyword) > 3:  # Only use meaningful keywords
#                     keyword_results = self.kb.skill_search(keyword)
#                     if keyword_results:
#                         results.extend(keyword_results)
        
#         # Deduplicate results
#         seen = set()
#         deduped = []
#         for course in results:
#             course_id = course.get("course_id")
#             if course_id and course_id not in seen:
#                 seen.add(course_id)
#                 deduped.append(course)
        
#         return deduped

#     def _integrate_results(self, hop_results: List[Dict], analysis: Dict, 
#                           result: Dict) -> List[Dict]: