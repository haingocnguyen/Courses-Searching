import json
import numpy as np
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import ollama
import re
from sentence_transformers import util
from typing import Any, Dict, List, Optional, Tuple, Union
import traceback


# ======================
# 1. Knowledge Base Module
# ======================
class CourseKnowledgeBase:
    def __init__(self, data_path: str):
        self.data = pd.read_json(data_path)
        self._preprocess()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_indices()
        self._construct_knowledge_graph()
        self.operation_log = []

    def _preprocess(self):
        # Extract level with better fallbacks
        LEVEL_MAP = {
            'advanced learners': 'advanced',
            'beginner friendly': 'beginner',
            'intermediate': 'intermediate',
            'adv': 'advanced',
            'beginner level': 'beginner'
        }

        def normalize_level(x):
            if isinstance(x, dict):
                levels = x.get('suitable_for', [])
                if levels:
                    raw_level = str(levels[0]).lower()
                    return LEVEL_MAP.get(raw_level, raw_level)
            return 'unknown'

        self.data['level'] = self.data['learning_path'].apply(normalize_level)
        
        # Fallback to category-based level inference
        self.data['level'] = self.data.apply(
            lambda row: row['level'] if row['level'] != 'unknown' 
            else 'advanced' if 'advanced' in row['category'].lower() 
            else row['level'],
            axis=1
        )
        # Ensure teaches/prerequisites are always lists
        def safe_get_list(field):
            return self.data['knowledge_requirements'].apply(
                lambda x: x.get(field, []) 
                if isinstance(x, dict) 
                else []
            ).apply(lambda y: y if isinstance(y, list) else [])

        self.data['teaches'] = safe_get_list('teaches')
        self.data['prerequisites'] = safe_get_list('prerequisites')

        # Create search text safely
        # Include more fields in search text
        self.data['search_text'] = self.data.apply(
            lambda row: ' '.join(filter(None, [
                str(row.get('title', '')),
                str(row.get('description', '')),
                ' '.join(map(str, row['teaches'])),
                str(row.get('category', '')),
                str(row.get('sub_category', ''))
            ])), axis=1
        )
        self.data['duration_months'] = (
            pd.to_numeric(self.data['duration_months'], errors='coerce')
            .fillna(0)  # Treat missing as 0
            .astype(int)
        )
        # Cap unrealistic durations
        self.data['duration_months'] = self.data['duration_months'].apply(
            lambda x: x if 1 <= x <= 36 else 0
        )
        self._log_sample_data()

    def _log_sample_data(self):
        """Log sample data for inspection"""
        sample = self.data.head(2).to_dict("records")
        print("\n=== SAMPLE COURSE DATA ===")
        for c in sample:
            print(f"Course: {c['title']}")
            print(f"Teaches: {c['teaches'][:3]}...")
            print(f"Prereqs: {c['prerequisites'][:3]}...")
            print(f"Duration: {c['duration_months']} months")
            print(f"Level: {c['level']}\n")
        print("\n=== END SAMPLE COURSE DATA ===")

    def _build_indices(self):
        """Create search indices with robust error handling"""
        # Semantic index
        text_embeddings = self.encoder.encode(self.data['search_text'].tolist())
        self.semantic_index = faiss.IndexFlatIP(text_embeddings.shape[1])
        self.semantic_index.add(text_embeddings.astype('float32'))
        
        # Attribute indices
        self.indices = {
            'level': self._build_attribute_index('level'),
            'category': self._build_category_index()
        }

    def _build_attribute_index(self, field: str) -> Dict:
        """Safe attribute index builder"""
        index = {}
        if field in self.data.columns:
            for val, group in self.data.groupby(field):
                key = str(val).lower()
                index[key] = set(group.index)
        else:
            print(f"Warning: Missing column '{field}' for indexing")
        return index

    def _build_category_index(self) -> Dict:
        """Safe category index builder"""
        index = {}
        for idx, row in self.data.iterrows():
            # Handle potential None values
            category = str(row.get('category', 'Unknown')).lower()
            sub_category = str(row.get('sub_category', 'Unknown')).lower()
            
            # Ensure teaches is always iterable
            teaches = row['teaches'] if isinstance(row['teaches'], list) else []
            
            sources = [category, sub_category] + [str(t).lower() for t in teaches]
            
            for val in sources:
                if val:  # Skip empty strings
                    index.setdefault(val, set()).add(idx)
        return index

    def _construct_knowledge_graph(self):
        """Null-safe knowledge graph construction"""
        self.graph = nx.DiGraph()
        
        for _, row in self.data.iterrows():
            course_id = row['course_id']
            self.graph.add_node(course_id, type='course', **row.to_dict())

            # Add teaches relationships
            for skill in row['teaches']:
                if skill and pd.notna(skill):
                    # Improved normalization: remove special chars, lowercase, and trim
                    normalized_skill = re.sub(r'[^\w\s]', '', str(skill).lower()).strip()
                    if normalized_skill:  # Skip empty strings
                        skill_node = f"skill:{normalized_skill}"
                        self.graph.add_edge(course_id, skill_node, relation='teaches')

            # Add prerequisites
            for prereq in row['prerequisites']:
                if prereq and pd.notna(prereq):
                    prereq_node = f"skill:{str(prereq).lower().strip()}"
                    self.graph.add_edge(prereq_node, course_id, relation='prerequisite')

    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Safe semantic search with empty handling"""
        if self.semantic_index is None or len(self.data) == 0:
            return []
        
        query_embed = self.encoder.encode(query)
        if query_embed.ndim == 1:
            query_embed = np.expand_dims(query_embed, 0)
        
        # Check embedding dimensions match index
        if query_embed.shape[1] != self.semantic_index.d:
            return []
        
        _, indices = self.semantic_index.search(query_embed.astype('float32'), top_k)
        return self.data.iloc[indices[0]].to_dict('records')

# ======================
# 2. Reasoning Components
# ======================
class QueryAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.retries = 3
        ALLOWED_LEVELS = ["beginner", "intermediate", "advanced"]

    def parse_query(self, query: str) -> Dict:
        """Decompose query into structured operations"""
        schema = {
            "type": "object",
            "properties": {
                "semantic_terms": {"type": "array", "items": {"type": "string"}},
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["attribute", "skill"]},
                            "field": {"type": "string"},
                            "value": {"type": "string"}
                        },
                        "required": ["type", "value"]
                    }
                },
                "constraints": {
                    "type": "object",
                    "properties": {
                        "duration": {"type": "number"},
                        "level": {"type": "string"},
                        "exclude_prerequisites": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["semantic_terms", "filters"]
        }

        prompt = f"""Analyze this course search query and output JSON matching this schema:
            {json.dumps(schema, indent=2)}

            Notes:
            - If the query mentions course difficulty ("beginner", "advanced", "expert", "intermediate"), set it under constraints.level.
            - Treat "professional", "expert", "senior" as equivalent to "advanced" difficulty and put "advanced" in constraints.level for query contains those words.
            - If the query mentions a maximum duration (e.g., "under 3 months"), put it under constraints.duration.
            - Filters are only for special attributes beyond standard fields.

            Assume duration values without units are in months.

            Examples:

            Example 1:
            Query: "Machine learning without math prerequisites"
            Output:
            {{
                "semantic_terms": ["machine learning"],
                "filters": [],
                "constraints": {{
                    "exclude_prerequisites": ["math"]
                }}
            }}

            Example 2:
            Query: "Web development under 4 months"
            Output:
            {{
                "semantic_terms": ["web development"],
                "filters": [],
                "constraints": {{
                    "duration": 4
                }}
            }}

            Example 3:
            Query: "Advanced Python courses"
            Output:
            {{
                "semantic_terms": ["Python"],
                "filters": [],
                "constraints": {{
                    "level": "advanced"
                }}
            }}

            Example 4:
            Query: "Deep learning beginner course with TensorFlow"
            Output:
            {{
                "semantic_terms": ["deep learning", "TensorFlow"],
                "filters": [],
                "constraints": {{
                    "level": "beginner"
                }}
            }}

            Example 5:
            Query: "Expert Python courses"
            Output:
            {{
                "semantic_terms": ["Python"],
                "filters": [],
                "constraints": {{
                    "level": "advanced"
                }}
            }}

            Now analyze this query:
            \"\"\"{query}\"\"\"
            Return only valid JSON.
            """
        
        for attempt in range(self.retries):
            try:
                response = ollama.generate(
                    model="qwen2.5:1.5b",
                    prompt=prompt,
                    format="json",
                    options={"temperature": 0.2}
                )
                return json.loads(response['response'])
            except Exception as e:
                print(f"Parse attempt {attempt+1} failed: {str(e)}")
        print(f"\n=== QUERY ANALYSIS DEBUG ===")
        print(f"Original query: {query}")
        print(f"LLM response: {response['response']}")
        return {"semantic_terms": [], "filters": [], "constraints": {}}

class CourseValidator:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def validate_course(self, course_id: str, query: str) -> Dict:
        if course_id not in self.kg.nodes:
            return {"valid": False, "errors": ["Course not found"]}
        
        course = self.kg.nodes[course_id]
        
        # Ensure default values for validation checks
        validation = {
            "prerequisites": self._check_prerequisites(course),
            "recency": datetime.now().year - course.get('year', 2000) <= 10,
            "rating": course.get('rating', 0) >= 1.0,
            "relevance": self._check_relevance(course, query)
        }
        
        # Generate detailed error messages
        errors = []
        if not validation["prerequisites"]:
            errors.append(f"Missing prerequisites: {course.get('prerequisites', [])}")
        if not validation["recency"]:
            errors.append(f"Last updated in {course.get('year', 'unknown')}")
        if not validation["rating"]:
            errors.append(f"Low rating ({course.get('rating', 0)})")
        if not validation["relevance"]:
            errors.append(f"Relevance score too low")
        
        return {
            "valid": any(validation.values()),  # Changed from all() to any()
            "errors": errors,
            "validation_details": validation
        }

    def _check_prerequisites(self, course: Dict) -> bool:
        return len(course.get('prerequisites', [])) > 0  # Just check existence

    def _check_recency(self, course: Dict) -> bool:
        return datetime.now().year - course.get('year', 0) <= 6

    def _check_rating(self, course: Dict) -> bool:
        return course.get('rating', 0) >= 1

    def _check_relevance(self, course: Dict, query: str) -> bool:
        """More lenient relevance check"""
        query_terms = set(re.findall(r'\w+', query.lower()))
        course_terms = set(re.findall(r'\w+', course.get('title', '').lower())) | \
                      set(re.findall(r'\w+', course.get('description', '').lower())) | \
                      set(str(t).lower() for t in course.get('teaches', []))
        return len(query_terms & course_terms) >= 1  # Reduced threshold

# ======================
# 3. Multi-Hop Search System
# ======================
class HiragSearchEngine:
    def __init__(self, data_path: str):
        self.kb = CourseKnowledgeBase(data_path)
        self.validator = CourseValidator(self.kb.graph)
        self.analyzer = QueryAnalyzer(ollama.Client())
        
    def execute_search(self, query: str) -> Dict:
        result = {
            "query": query,
            "analysis": {},
            "hops": [],
            "results": [],
            "errors": [],
            "logs": [],
            "debug": {}
        }
        
        try:
            # ========== NEW DEBUGGING ==========
            print(f"\n=== NEW SEARCH: {query} ===")
            print("Knowledge Graph Stats:")
            print(f"- Total nodes: {len(self.kb.graph.nodes)}")
            print(f"- Skill nodes: {len([n for n in self.kb.graph.nodes if n.startswith('skill:')])}")
            
            # 1. Enhanced query analysis with level detection
            analysis = self.analyzer.parse_query(query)
            result["analysis"] = analysis
            self._log_debug(result, "01_raw_analysis", analysis)
            print("\n=== QUERY ANALYSIS ===")
            print(f"Semantic Terms: {analysis['semantic_terms']}")
            print(f"Filters: {analysis['filters']}")
            print(f"Constraints: {analysis.get('constraints', {})}")
            # 2. Add course metadata snapshot
            self._log_debug(result, "02_kb_snapshot", {
                "total_courses": len(self.kb.data),
                "sample_skills": list(self.kb.graph.nodes)[:5],
                "data_columns": list(self.kb.data.columns),
                "duration_stats": {  # NEW: Duration metrics
                    "min": self.kb.data.duration_months.min(),
                    "max": self.kb.data.duration_months.max(),
                    "mean": self.kb.data.duration_months.mean()
                }
            })

            # 3. Execute hops with enhanced debugging
            for idx, hop in enumerate(analysis.get("filters", [])):
                print(f"\nExecuting {hop['type']} hop for {hop.get('value')}")  # NEW
                hop_result = self._execute_hop(hop, result)
                result["hops"].append(hop_result)
                
                # NEW: Log hop execution details
                self._log_debug(result, f"03_hop_{idx}_details", {
                    "hop_type": hop["type"],
                    "hop_value": hop.get("value"),
                    "raw_results_before_filter": [c["course_id"] for c in hop_result["results"]],
                    "graph_connections": self._get_graph_connections(hop),
                    # NEW: Add first 3 results preview
                    "sample_results": [
                        {k: v for k, v in c.items() if k in ['course_id', 'title', 'duration_months']}
                        for c in hop_result["results"][:3]
                    ]
                })
            
            # 4. Improved result integration
            candidates = self._integrate_results(result["hops"], analysis, result)
            print(f"\nPre-filter candidates: {len(candidates)}")  # NEW
            
            # NEW: Add semantic search fallback
            if not candidates:
                print("No candidates from hops, falling back to semantic search")
                candidates = self.kb.semantic_search(
                    " ".join(analysis["semantic_terms"]), 
                    top_k=20
                )
            # 5. Enhanced validation with debug
            validation_report = []
            print("\n=== VALIDATION BREAKDOWN ===")
            for c in candidates:
                valid = self.validator.validate_course(c["course_id"], query)
                # NEW: Add duration and level info
                validation_report.append({
                    "course_id": c["course_id"],
                    "title": c["title"],
                    "duration": c.get("duration_months"),
                    "level": c.get("level"),
                    "valid": valid["valid"],
                    "reasons": valid.get("errors", [])
                })
                # NEW: Print validation failures
                if not valid["valid"]:
                    print(f"Invalid course {c['course_id']}: {valid.get('errors')}")
            print("\n=== END VALIDATION BREAKDOWN ===")
            self._log_debug(result, "05_validation_report", validation_report)
            
            # 6. Final filtering and ranking
            final_results = self._validate_and_rank(
                candidates, 
                analysis, 
                query, 
                result
            )
            print("\n=== VALIDATION SUMMARY ===")
            print(f"Total candidates: {len(candidates)}")
            print(f"Valid courses: {len(final_results)}")
            # NEW: Add score-based ranking
            result["results"] = [{
                **c,
                "validation": self.validator.validate_course(c["course_id"], query),
                "score": self._calculate_quality_score(c)
            } for c in final_results[:10]]  # Return top 10

        except Exception as e:
            self._log_error(result, 
                f"ERROR: {type(e).__name__}\n"
                f"Message: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            # NEW: Print full error trace
            traceback.print_exc()
        
        # NEW: Final debug summary
        print("\n=== SEARCH SUMMARY ===")
        print(f"Hops executed: {len(result['hops'])})")
        print(f"Final results: {len(result['results'])}")
        print(f"Errors encountered: {len(result['errors'])}")
        
        return result
    def analyze_query(query: str) -> Dict:
        """Simple query analyzer: extract semantic terms, filters, and constraints."""
        level_keywords = {"beginner", "intermediate", "advanced", "expert"}

        # Normalize
        query = query.lower().strip()

        # Very simple token split (you can improve later)
        tokens = query.split()

        semantic_terms = []
        filters = []
        constraints = {}

        for token in tokens:
            if token in level_keywords:
                constraints["level"] = token
            else:
                semantic_terms.append(token)

        return {
            "semantic_terms": semantic_terms,
            "filters": [],  # you can add attribute filters later
            "constraints": constraints
        }


    def _execute_hop(self, hop: Dict, result: Dict) -> Dict:
        """Execute a search hop with multiple strategies"""
        hop_result = {
            "type": hop.get("type", "unknown"),
            "parameters": hop,
            "results": [],
            "error": None
        }
        
        try:
            if hop["type"] == "attribute":
                results = self._execute_attribute_hop(hop)
            elif hop["type"] == "skill":
                results = self._execute_skill_hop(hop)
            else:
                results = []
                
            hop_result["results"] = results
            self._log(result, "HopExecuted", 
                     f"{hop['type']} hop found {len(results)} results")
            
        except Exception as e:
            hop_result["error"] = str(e)
            self._log_error(result, f"Hop failed: {str(e)}", hop)
            
        return hop_result

    def _execute_attribute_hop(self, hop: Dict) -> List[Dict]:
        if hop["field"] == "duration":
            try:
                # Get numeric value from query
                months = int("".join(filter(str.isdigit, hop["value"])))
                
                # Debug: Show courses within duration range
                valid_courses = self.kb.data[
                    (self.kb.data.duration_months <= months) & 
                    (self.kb.data.duration_months > 0)
                ]
                
                print(f"Duration Filter Debug (<= {months} months):")
                print(valid_courses[['course_id', 'title', 'duration_months']])
                
                return valid_courses.to_dict("records")
                
            except Exception as e:
                print(f"Duration filter error: {str(e)}")
                return []

    def _meets_constraints(self, course: Dict, constraints: Dict) -> bool:
        """Enhanced constraint checking"""
        # Duration constraint
        if constraints.get("duration"):
            try:
                if course["duration_months"] > constraints["duration"]:
                    return False
            except (KeyError, TypeError):
                pass
        
        # Prerequisite exclusion
        excluded_terms = [t.lower() for t in constraints.get("exclude_prerequisites", [])]
        course_prereqs = [str(p).lower() for p in course.get("prerequisites", [])]
        if any(term in course_prereqs for term in excluded_terms):
            return False
            
        return True


    def _execute_skill_hop(self, hop: Dict) -> List[Dict]:
        """Enhanced skill hop debugging"""
        skill = hop.get("value", "").lower().strip()
        skill_node = f"skill:{skill}"
        print(f"DEBUG: Searching for skill '{skill}' in knowledge graph...")
        print(f"Existing skill nodes: {[n for n in self.kb.graph.nodes if n.startswith('skill:')][:5]}")

        # Debug: Check skill node existence
        node_exists = self.kb.graph.has_node(skill_node)
        print(f"\n=== SKILL HOP DEBUG ===")
        print(f"Skill: {skill} | Node exists: {node_exists}")

        if node_exists:
            # Safe to access predecessors
            courses = list(self.kb.graph.predecessors(skill_node))
            print(f"Connected courses: {courses}")
            return self.kb.data[self.kb.data['course_id'].isin(courses)].to_dict('records')
        else:
            # Node missing: directly fallback to semantic search
            fallback_results = self.kb.semantic_search(skill)
            print(f"Fallback results: {[r['course_id'] for r in fallback_results]}")
            return fallback_results


    def _integrate_results(self, context: List[Dict], analysis: Dict, 
                         result: Dict) -> List[Dict]:
        """Intelligent result integration"""
        # Collect all candidate results
        candidates = []
        for hop in context:
            if hop["results"]:
                candidates.extend(hop["results"])
        
        # Fallback to semantic search if no filters
        if not candidates:
            print("Using constraint-aware semantic fallback")
            boosted_query = " ".join([
                *analysis["semantic_terms"],
                analysis.get("constraints", {}).get("level", "")
            ])
            results = self.kb.semantic_search(boosted_query, top_k=25)
            return [c for c in results if self._meets_constraints(c, analysis.get("constraints", {}))]
                
        # Apply constraints
        constrained = [c for c in candidates 
                      if self._meets_constraints(c, analysis.get("constraints", {}))]
        
        # Deduplicate and rerank
        seen = set()
        deduped = [c for c in constrained 
                  if not (c["course_id"] in seen or seen.add(c["course_id"]))]
        
        return [c for c in candidates if self._meets_soft_constraints(c, analysis)]
    
    def _meets_soft_constraints(self, course: Dict, analysis: Dict) -> bool:
        """Relaxed constraint checking"""
        constraints = analysis.get("constraints", {})
        
        # Duration constraint
        if "duration" in constraints:
            max_duration = int(constraints["duration"])
            if course.get("duration_months", 0) > max_duration:
                return False
        
        # Level constraint (already verified in validation)
        return True
    
    def _matches_level(self, course: Dict, analysis: Dict) -> bool:
        # Normalization dictionary
        LEVEL_MAP = {
            'advanced learners': 'advanced',
            'beginner friendly': 'beginner',
            'intermediate': 'intermediate',
            'adv': 'advanced',
            'beginner level': 'beginner'
        }
        
        # Get constraint from query analysis
        constraint_level = analysis.get("constraints", {}).get("level", "").lower()
        
        # Get course level with fallback
        course_level = str(course.get("level", "")).lower().strip()
        
        # Normalize both values
        normalized_constraint = LEVEL_MAP.get(constraint_level, constraint_level)
        normalized_course = LEVEL_MAP.get(course_level, course_level)
        
        print(f"Level Check: {course['course_id']}")
        print(f"Course Level: {course_level} => {normalized_course}")
        print(f"Constraint Level: {constraint_level} => {normalized_constraint}")
        
        return normalized_course == normalized_constraint
            
    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Safe semantic reranking"""
        if not results:
            return []
            
        try:
            # Generate embeddings
            texts = [f"{c['title']} {c['description']}" for c in results]
            text_embeds = self.kb.encoder.encode(texts)
            query_embed = self.kb.encoder.encode(query)
            
            # Check embedding dimensions
            if text_embeds.shape[1] != query_embed.shape[0]:
                return results
                
            # Calculate scores
            scores = np.dot(text_embeds, query_embed)
            return [c for _, c in sorted(zip(scores, results), reverse=True)]
        except:
            return results

    def _validate_and_rank(self, candidates: List[Dict], analysis: Dict,
                          query: str, result: Dict) -> List[Dict]:
        """Updated validation with level matching"""
        validated = []
        for course in candidates:
            try:
                # Check all validation aspects
                valid = all([
                    self.validator.validate_course(course["course_id"], query)["valid"],
                    self._matches_level(course, analysis),
                    self._meets_prereq_constraints(course, analysis)
                ])
                
                if valid:
                    # Calculate quality score
                    score = self._calculate_quality_score(course)
                    validated.append((score, course))
                    
            except Exception as e:
                self._log_error(result, f"Validation error: {str(e)}", course)
        
        # Sort by score then duration
        return [c for _, c in sorted(validated, key=lambda x: (-x[0], x[1]["duration_months"]))][:10]

    def _calculate_quality_score(self, course: Dict) -> float:
        """Quality scoring algorithm"""
        score = 0.0
        # Recency (0-30 points)
        score += min(30, (datetime.now().year - course.get("year", 0)) * 3)
        # Rating (0-40 points)
        score += min(40, course.get("rating", 0) * 8)
        # Duration (0-30 points)
        score += max(0, 30 - course.get("duration_months", 0))
        return score

    def _meets_prereq_constraints(self, course: Dict, analysis: Dict) -> bool:
        """Check prerequisite constraints with improved matching"""
        excluded_terms = [str(p).lower() for p in analysis.get("constraints", {}).get("exclude_prerequisites", [])]
        course_prereqs = [str(p).lower() for p in course.get("prerequisites", [])]
        
        # Check if any excluded term exists in prerequisites
        return not any(excluded_term in " ".join(course_prereqs) 
                      for excluded_term in excluded_terms)


    def _log(self, result: Dict, category: str, message: str):
        """Add operational log entry"""
        result["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "message": message
        })

    def _log_error(self, result: Dict, message: str, context: Any = None):
        """Add error log entry"""
        result["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "context": context
        })
    def _log_debug(self, result: Dict, key: str, data: Any):
        """Store debug information"""
        result["debug"][key] = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
    def _get_graph_connections(self, hop: Dict) -> Dict:
        """Debug skill graph connections"""
        if hop["type"] != "skill":
            return {}
            
        skill_node = f"skill:{hop['value'].lower()}"
        return {
            "skill_node_exists": self.kb.graph.has_node(skill_node),
            "connected_courses": list(self.kb.graph.predecessors(skill_node))[:5],
            "related_skills": list(self.kb.graph.successors(skill_node))[:5]
        }
# ======================
# 4. Usage Example
# ======================
if __name__ == "__main__":
    engine = HiragSearchEngine("D:\\Thesis\\Courses-Searching\\src\\db\\processed_courses_detail.json")
    
    queries = [
        #"Advanced Python courses with data science focus under 8 months"
        #"Advanced web development courses covering both frontend and backend"
        #"Machine learning fundamentals with Python but no math prerequisites"
        #"Show me advanced machine learning courses that cover TensorFlow and PyTorch"
        #"Advanced short courses on data analysis using Excel, under 3 months."
        "Professional certification courses for cloud engineering with Kubernetes and AWS, rating above 4.0."
    ]
    
    for query in queries:
        print(f"\n{'='*60}\nQuery: {query}\n{'='*60}")
        start_time = datetime.now()
        result = engine.execute_search(query)
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\nAnalysis ({duration:.2f}s):")
        print(json.dumps(result["analysis"], indent=2))
        
        print("\nHops:")
        for hop in result["hops"]:
            print(f"- {hop['type']}: {hop.get('metrics', {}).get('results_found', 0)} results")
            
        print(f"\nResults ({len(result['results'])}):")
        for i, course in enumerate(result["results"][:3], 1):
            print(f"\n{i}. {course['title']} ({course['course_id']})")
            print(f"   Level: {course.get('level', 'N/A')}")
            print(f"   Duration: {course.get('duration_months', '?')} months")
            print(f"   Validation: {json.dumps(course['validation'], indent=2)}")
            
        if result["errors"]:
            print("\nErrors:")
            for error in result["errors"]:
                print(f"- {error['message']}")