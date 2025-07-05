import numpy as np
import logging
import streamlit as st
from typing import List, Dict
from models.llm import OllamaLLM
from config import LLM_MODEL

logger = logging.getLogger(__name__)

class EnhancedResultHandler:
    """Enhanced result handler for different entity types"""
    
    def __init__(self, llm_model=LLM_MODEL):
        self.llm_model = llm_model

    def detect_query_type(self, query: str, results: List[Dict]) -> str:
        """Enhanced detection for all entity types"""
        if not results:
            return "empty"
            
        first_result = results[0]
        query_lower = query.lower()
        
        # Priority 1: Check query intent keywords
        intent_mapping = {
            "instructor": ["instructor", "teacher", "professor", "taught by", "who teaches"],
            "organization": ["organization", "university", "college", "school", "institution"],
            "provider": ["provider", "platform", "offered by", "coursera", "edx", "udemy"],
            "review": ["review", "feedback", "comment", "rating", "student says"],
            "subject": ["subject", "topic", "area", "field", "domain"],
            "skill": ["skill", "ability", "competency", "learn skill"],
            "level": ["level", "difficulty", "beginner", "intermediate", "advanced"],
            "statistical": ["average", "statistics", "how many", "count", "total", "percentage"]
        }
        
        for query_type, keywords in intent_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                # Verify results match intent
                if query_type == "instructor" and any(key in first_result for key in ["instructor_name", "instructor_rating", "i.name"]):
                    return "instructor"
                elif query_type == "organization" and any(key in first_result for key in ["organization_name", "organization_description", "o.name"]):
                    return "organization"
                elif query_type == "provider" and any(key in first_result for key in ["provider_name", "provider_description", "p.name"]):
                    return "provider"
                elif query_type == "review" and any(key in first_result for key in ["review_comment", "review_rating", "r.comment"]):
                    return "review"
                elif query_type == "subject" and any(key in first_result for key in ["subject_name", "subject_description", "sub.name"]):
                    return "subject"
                elif query_type == "skill" and any(key in first_result for key in ["skill_name", "skill_description", "s.name"]):
                    return "skill"
                elif query_type == "level" and any(key in first_result for key in ["level_name", "level_description", "l.name"]):
                    return "level"
                elif query_type == "statistical" and any(key in first_result for key in ["avg_rating", "total_courses", "course_count"]):
                    return "statistical"
        
        # Priority 2: Check result structure
        if any(key in first_result for key in ["instructor_name", "instructor_rating", "i.name"]):
            return "instructor"
        elif any(key in first_result for key in ["organization_name", "organization_description", "o.name"]):
            return "organization"
        elif any(key in first_result for key in ["provider_name", "provider_description", "p.name"]):
            return "provider"
        elif any(key in first_result for key in ["review_comment", "review_rating", "r.comment"]):
            return "review"
        elif any(key in first_result for key in ["subject_name", "subject_description", "sub.name"]):
            return "subject"
        elif any(key in first_result for key in ["skill_name", "skill_description", "s.name"]):
            return "skill"
        elif any(key in first_result for key in ["level_name", "level_description", "l.name"]):
            return "level"
        elif any(key in first_result for key in ["avg_rating", "total_courses", "course_count"]):
            return "statistical"
        elif "url" in first_result:
            return "course"
        else:
            return "mixed"

    def analyze_results(self, results: List[Dict], query: str, query_type: str = None) -> str:
        """Generate analysis based on result type"""
        if not results:
            return "I couldn't find any results matching your query. Try rephrasing or asking about a different topic."
            
        if query_type is None:
            query_type = self.detect_query_type(query, results)
            
        return self._generate_analysis_by_type(results, query, query_type)
    
    def _generate_analysis_by_type(self, results: List[Dict], query: str, query_type: str) -> str:
        """Generate specific analysis based on query type"""
        
        if query_type == "instructor":
            return self._analyze_instructor_results(results, query)
        elif query_type == "organization":
            return self._analyze_organization_results(results, query)
        elif query_type == "provider":
            return self._analyze_provider_results(results, query)
        elif query_type == "review":
            return self._analyze_review_results(results, query)
        elif query_type == "subject":
            return self._analyze_subject_results(results, query)
        elif query_type == "statistical":
            return self._analyze_statistical_results(results, query)
        elif query_type == "course":
            return self._analyze_course_results(results, query)
        else:
            return self._analyze_mixed_results(results, query)
    
    def _analyze_instructor_results(self, results: List[Dict], query: str) -> str:
        """Generate analysis for instructor search results - focus on names and insights"""
        if not results:
            return "No instructors found matching your query."
        
        # Safe rating extraction with None handling
        def safe_rating(instructor):
            rating = instructor.get("instructor_rating")
            if rating is None or rating == "N/A":
                return 0.0
            try:
                return float(rating)
            except (ValueError, TypeError):
                return 0.0
        
        # Filter out "Unknown" instructors and get real names
        named_instructors = []
        for instructor in results:
            name = instructor.get("name", "").strip()
            if name and name != "Unknown" and name != "N/A":
                named_instructors.append(instructor)
        
        # Get notable instructors (with ratings or good names)
        notable_instructors = []
        for instructor in named_instructors:
            rating = safe_rating(instructor)
            if rating > 0:  # Has rating
                notable_instructors.append((instructor, rating))
        
        # Sort by rating
        notable_instructors.sort(key=lambda x: x[1], reverse=True)
        
        # Count specializations from all results
        specializations = {}
        for instructor in results:
            skills = instructor.get("skills", [])
            if isinstance(skills, list):
                for skill in skills:
                    if skill and skill != "N/A" and skill.strip():
                        specializations[skill] = specializations.get(skill, 0) + 1
        
        top_specializations = sorted(specializations.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate streamlined analysis
        analysis = f"## 👨‍🏫 Instructor Landscape\n\n"
        analysis += f"Found **{len(results)} instructors** in this field"
        
        if named_instructors:
            analysis += f", including **{len(named_instructors)} named instructors**.\n\n"
            
            # Show notable names only
            if notable_instructors:
                analysis += f"**Top Rated Instructors:**\n"
                for i, (instructor, rating) in enumerate(notable_instructors[:3], 1):
                    name = instructor.get("name", "Unknown")
                    org = instructor.get("organization", "")
                    
                    analysis += f"{i}. **{name}** ({rating:.1f}/5.0)"
                    if org and org != "N/A":
                        analysis += f" - {org}"
                    analysis += "\n"
                analysis += "\n"
            
            # List other notable names without details
            other_names = [inst.get("name", "") for inst in named_instructors 
                        if inst.get("name", "") not in [inst[0].get("name", "") for inst in notable_instructors[:3]]]
            other_names = [name for name in other_names if name and name != "Unknown"]
            
            if other_names:
                analysis += f"**Other Notable Instructors:** "
                analysis += ", ".join(other_names[:5])
                if len(other_names) > 5:
                    analysis += f" and {len(other_names) - 5} more"
                analysis += "\n\n"
        else:
            analysis += ".\n\n"
        
        # Specializations
        if top_specializations:
            analysis += f"**Teaching Specializations:**\n"
            for skill, count in top_specializations:
                analysis += f"- **{skill}**: {count} instructors\n"
        
        return analysis
    def _analyze_course_results(self, results: List[Dict], query: str) -> str:
        """Generate analysis for course search results - streamlined version"""
        if not results:
            return "No courses found matching your query."
        
        # Safe rating extraction
        def safe_course_rating(course):
            rating = course.get("rating")
            if rating is None or rating == "N/A":
                return 0.0
            try:
                return float(rating)
            except (ValueError, TypeError):
                return 0.0
        
        # Get courses with valid names and ratings
        valid_courses = []
        for course in results:
            name = course.get("name", "").strip()
            if name and name != "Unknown" and name != "N/A":
                rating = safe_course_rating(course)
                valid_courses.append((course, rating))
        
        # Sort by rating
        valid_courses.sort(key=lambda x: x[1], reverse=True)
        
        # Statistics from all results
        all_ratings = [safe_course_rating(c) for c in results]
        valid_ratings = [r for r in all_ratings if r > 0]
        avg_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else 0
        
        # Levels and providers
        levels = {}
        providers = {}
        
        for course in results:
            level = course.get("level", "")
            if level and level != "N/A" and level.strip():
                levels[level] = levels.get(level, 0) + 1
                
            provider = course.get("provider", "")
            if provider and provider != "N/A" and provider.strip():
                providers[provider] = providers.get(provider, 0) + 1
        
        # Generate streamlined analysis
        analysis = f"## 📚 Course Landscape\n\n"
        analysis += f"Found **{len(results)} courses** in this area"
        
        if valid_ratings:
            analysis += f" with an average rating of **{avg_rating:.1f}/5.0**.\n\n"
        else:
            analysis += ".\n\n"
        
        # Top courses (only show if they have good ratings)
        top_rated = [(course, rating) for course, rating in valid_courses if rating > 3.5][:5]
        if top_rated:
            analysis += f"**Highly Rated Courses:**\n"
            for i, (course, rating) in enumerate(top_rated, 1):
                name = course.get("name", "Unknown")
                provider = course.get("provider", "")
                
                analysis += f"{i}. **{name}** ({rating:.1f}/5.0)"
                if provider and provider != "N/A":
                    analysis += f" - {provider}"
                analysis += "\n"
            analysis += "\n"
        
        # Quick stats
        if levels:
            level_summary = ", ".join([f"{count} {level.lower()}" for level, count in 
                                    sorted(levels.items(), key=lambda x: x[1], reverse=True)[:3]])
            analysis += f"**Difficulty Mix:** {level_summary}\n\n"
        
        if providers:
            top_providers = sorted(providers.items(), key=lambda x: x[1], reverse=True)[:3]
            provider_summary = ", ".join([f"{provider} ({count})" for provider, count in top_providers])
            analysis += f"**Top Providers:** {provider_summary}\n"
        
        return analysis
    def _analyze_organization_results(self, results: List[Dict], query: str) -> str:
        """Analyze organization query results"""
        total = len(results)
        # Get course counts if available
        course_counts = [int(r.get("courses_offered", r.get("course_count", 0))) for r in results]
        total_courses = sum(course_counts) if course_counts else 0
        
        context = f"""
User asked: "{query}"
Found {total} organizations offering {total_courses} total courses.
Top organizations by course count: {', '.join([r.get('organization_name', r.get('organization', 'Unknown')) for r in results[:3]])}.
Provide insights about these educational organizations in 3-4 sentences.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_provider_results(self, results: List[Dict], query: str) -> str:
        """Analyze provider query results"""
        total = len(results)
        course_counts = [int(r.get("total_courses", 0)) for r in results]
        total_courses = sum(course_counts) if course_counts else 0
        
        context = f"""
User asked: "{query}"
Found {total} course providers with {total_courses} total courses.
Major providers: {', '.join([r.get('provider_name', 'Unknown') for r in results[:5]])}.
Provide insights about the course provider landscape in 3-4 sentences.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_review_results(self, results: List[Dict], query: str) -> str:
        """Analyze review query results"""
        total = len(results)
        ratings = [float(r.get("review_rating", 0)) for r in results if r.get("review_rating")]
        avg_rating = np.mean(ratings) if ratings else 0
        
        context = f"""
User asked: "{query}"
Found {total} course reviews, average rating {avg_rating:.2f}.
Reviews range from various courses. Provide insights about student feedback patterns in 3-4 sentences.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_subject_results(self, results: List[Dict], query: str) -> str:
        """Analyze subject query results"""
        total = len(results)
        subjects = [r.get('subject_name', 'Unknown') for r in results[:5]]
        
        context = f"""
User asked: "{query}"
Found {total} subjects: {', '.join(subjects)}.
Provide insights about these subject areas and their educational coverage in 3-4 sentences.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_statistical_results(self, results: List[Dict], query: str) -> str:
        """Analyze statistical query results"""
        context = f"""
User asked: "{query}"
Statistical analysis results: {len(results)} data points found.
Key metrics include ratings, course counts, and other quantitative measures.
Provide insights about these statistics in 3-4 sentences, highlighting key trends and patterns.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_mixed_results(self, results: List[Dict], query: str) -> str:
        """Analyze mixed query results"""
        total = len(results)
        context = f"""
User asked: "{query}"
Found {total} results with mixed data types including courses, instructors, organizations, and other educational entities.
Provide a comprehensive overview in 3-4 sentences about what was found.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])

def needs_clarification(query: str) -> bool:
    """Simple check - only need check intent is course_search needs clarification"""
    return True

def clear_all_caches():
    """Clear all caches at exit"""
    try:
        import streamlit as _st
        _st.cache_data.clear()
        _st.cache_resource.clear()
    except:
        pass
    
    import os
    import shutil
    for d in (
        os.path.expanduser("~/.streamlit/cache"),
        os.path.expanduser("~/.cache/huggingface"),
        os.getenv("TEMP", "")
    ):
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)