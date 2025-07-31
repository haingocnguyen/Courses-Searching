import numpy as np
import logging
from typing import List, Dict
from models.llm import OllamaLLM
from config import LLM_MODEL
import re

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
        """Enhanced instructor analysis with comprehensive overview"""
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
        
        # Filter and analyze instructors
        named_instructors = []
        all_ratings = []
        specializations = {}
        organizations = []
        course_counts = []
        
        for instructor in results:
            name = instructor.get("instructor_name", instructor.get("i.name", instructor.get("name", ""))).strip()
            if name and name != "Unknown" and name != "N/A":
                named_instructors.append(instructor)
                
            rating = safe_rating(instructor)
            if rating > 0:
                all_ratings.append(rating)
                
            # Collect specializations
            skills = instructor.get("skills", [])
            if isinstance(skills, list):
                for skill in skills:
                    if skill and skill != "N/A" and skill.strip():
                        specializations[skill] = specializations.get(skill, 0) + 1
            
            # Organizations
            org = instructor.get("organization", "")
            if org and org != "N/A":
                organizations.append(org)
                
            # Course counts
            courses = instructor.get("courses_taught", instructor.get("course_count", 0))
            try:
                if courses and str(courses).isdigit():
                    course_counts.append(int(courses))
            except:
                pass
        
        # Statistics
        avg_rating = sum(all_ratings) / len(all_ratings) if all_ratings else 0
        rating_range = f"{min(all_ratings):.1f}-{max(all_ratings):.1f}" if len(all_ratings) > 1 else f"{all_ratings[0]:.1f}" if all_ratings else "N/A"
        total_courses = sum(course_counts) if course_counts else 0
        avg_courses = total_courses / len(course_counts) if course_counts else 0
        
        from collections import Counter
        top_specializations = Counter(specializations).most_common(5)
        top_organizations = Counter(organizations).most_common(3)
        
        # Sort instructors by rating
        notable_instructors = []
        for instructor in named_instructors:
            rating = safe_rating(instructor)
            notable_instructors.append((instructor, rating))
        notable_instructors.sort(key=lambda x: x[1], reverse=True)
        
        # Build text parts separately for f-string
        specializations_text = '\n'.join([f"- {spec}: {count} instructors" for spec, count in top_specializations[:3]])
        institutions_text = '\n'.join([f"- {org}: {count} instructors" for org, count in top_organizations])
        #instructors_text = '\n'.join([f"- {inst.get('instructor_name', inst.get('name', 'Unknown'))}: {rating:.1f}/5.0" for inst, rating in notable_instructors[:5] if rating > 0])
        
        # Build LLM prompt
        context = f"""You are an educational analyst. Provide a comprehensive overview of instructors in this field:

USER QUERY: "{query}"

INSTRUCTOR LANDSCAPE:
- Total instructors found: {len(results)}
- Named instructors: {len(named_instructors)}
- Total courses taught: {total_courses}
- Average courses per instructor: {avg_courses:.1f}

TOP SPECIALIZATIONS:
{specializations_text}

TOP INSTITUTIONS:
{institutions_text}


ANALYSIS REQUIREMENTS:
1. Overview of instructor expertise landscape in this field
2. Specialization patterns and expertise areas
3. Institutional diversity and academic backing
4. Notable standout educators and their strengths
5. Guidance for learners choosing instructors

Use markdown formatting with headers, emojis, and bold emphasis. Aim for 4-5 sections.
Generate comprehensive instructor landscape analysis:"""

        try:
            llm = OllamaLLM(model=self.llm_model)
            analysis = llm.invoke([{"role": "user", "content": context}])
            analysis = re.sub(r'<think>.*?</think>', '', analysis, flags=re.DOTALL).strip()
            
            if len(analysis) < 200:
                return self._generate_instructor_fallback(results, query, {
                    'total': len(results),
                    'named': len(named_instructors),
                    'avg_rating': avg_rating,
                    'top_specializations': top_specializations[:3],
                    'notable_instructors': notable_instructors[:3]
                })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Instructor analysis failed: {e}")
            return self._generate_instructor_fallback(results, query, {
                'total': len(results),
                'named': len(named_instructors),
                'avg_rating': avg_rating,
                'top_specializations': top_specializations[:3],
                'notable_instructors': notable_instructors[:3]
            })

    def _generate_instructor_fallback(self, results, query, stats):
        """Enhanced instructor fallback"""
        # Build text parts separately
        specializations_text = '\n'.join([f"• **{spec}**: {count} instructors" for spec, count in stats['top_specializations']])
        instructors_text = '\n'.join([f"• **{inst.get('instructor_name', inst.get('name', 'Unknown'))}** ({rating:.1f}/5.0)" for inst, rating in stats['notable_instructors'] if rating > 0])
        
        analysis = f"""## 👨‍🏫 Instructor Expertise Landscape

I found **{stats['total']} instructors** in this field, including **{stats['named']} named educators** with established teaching profiles. The instructor quality shows an average rating of **{stats['avg_rating']:.1f}/5.0**, indicating {"excellent" if stats['avg_rating'] > 4.5 else "strong"} teaching standards.

### 🎯 **Specialization Areas**
The primary teaching specializations include:
{specializations_text}

### ⭐ **Notable Educators**
Among the top-rated instructors:
{instructors_text}

### 💡 **Selection Insights**
The instructor landscape shows {"strong diversity" if len(stats['top_specializations']) > 2 else "focused expertise"} in teaching approaches, with educators bringing both academic rigor and practical experience to their courses."""
        return analysis.strip()

    def _analyze_organization_results(self, results: List[Dict], query: str) -> str:
        """Enhanced organization analysis"""
        if not results:
            return "No organizations found matching your query."
        
        # Collect organization data
        course_counts = []
        org_types = {}
        total_courses = 0
        
        for org in results:
            courses = org.get("courses_offered", org.get("course_count", 0))
            try:
                if courses and str(courses).isdigit():
                    count = int(courses)
                    course_counts.append(count)
                    total_courses += count
            except:
                pass
            
            # Categorize organization types
            name = org.get("organization_name", org.get("organization", ""))
            if name:
                if any(word in name.lower() for word in ["university", "college"]):
                    org_types["Universities"] = org_types.get("Universities", 0) + 1
                elif any(word in name.lower() for word in ["institute", "school"]):
                    org_types["Institutes"] = org_types.get("Institutes", 0) + 1
                else:
                    org_types["Other Institutions"] = org_types.get("Other Institutions", 0) + 1
        
        avg_courses = sum(course_counts) / len(course_counts) if course_counts else 0
        
        # Sort by course count
        sorted_orgs = sorted(results, key=lambda x: int(x.get("courses_offered", x.get("course_count", 0)) or 0), reverse=True)
        
        # Build text parts separately
        org_types_text = '\n'.join([f"- {org_type}: {count}" for org_type, count in org_types.items()])
        providers_text = '\n'.join([f"- {org.get('organization_name', org.get('organization', 'Unknown'))}: {org.get('courses_offered', org.get('course_count', 0))} courses" for org in sorted_orgs[:5]])
        
        context = f"""You are an educational analyst. Analyze the institutional landscape:

USER QUERY: "{query}"

ORGANIZATION LANDSCAPE:
- Total organizations: {len(results)}
- Total courses offered: {total_courses}
- Average courses per organization: {avg_courses:.1f}

INSTITUTION TYPES:
{org_types_text}

TOP COURSE PROVIDERS:
{providers_text}

Provide comprehensive analysis of the institutional education landscape including:
1. Overview of organizational diversity
2. Course offering patterns and institutional strengths
3. Academic vs professional training balance
4. Geographic or institutional prestige factors
5. Guidance for choosing institutions

Use markdown with headers and emojis. Generate analysis:"""

        try:
            llm = OllamaLLM(model=self.llm_model)
            analysis = llm.invoke([{"role": "user", "content": context}])
            return re.sub(r'<think>.*?</think>', '', analysis, flags=re.DOTALL).strip()
        except Exception as e:
            logger.error(f"Organization analysis failed: {e}")
            
            # Build fallback text parts separately
            org_types_fallback = '\n'.join([f"• **{org_type}**: {count} institutions" for org_type, count in org_types.items()])
            providers_fallback = '\n'.join([f"• **{org.get('organization_name', org.get('organization', 'Unknown'))}**: {org.get('courses_offered', org.get('course_count', 0))} courses" for org in sorted_orgs[:3]])
            
            return f"""## 🏛️ Institutional Landscape

Found **{len(results)} organizations** offering **{total_courses} total courses** in this field. The institutional diversity includes {len(org_types)} different types of educational providers.

### 📊 **Provider Types**
{org_types_fallback}

### 🔝 **Leading Providers**
{providers_fallback}

The institutional landscape shows {"strong diversity" if len(org_types) > 2 else "focused expertise"} in educational approaches."""

    def _analyze_provider_results(self, results: List[Dict], query: str) -> str:
        """Enhanced provider analysis"""
        if not results:
            return "No providers found matching your query."
        
        total_courses = sum([int(r.get("total_courses", r.get("course_count", 0)) or 0) for r in results])
        course_counts = [int(r.get("total_courses", r.get("course_count", 0)) or 0) for r in results]
        avg_courses = sum(course_counts) / len(course_counts) if course_counts else 0
        
        # Sort providers by course count
        sorted_providers = sorted(results, key=lambda x: int(x.get("total_courses", x.get("course_count", 0)) or 0), reverse=True)
        
        # Build text parts separately
        providers_text = '\n'.join([f"- {p.get('provider_name', 'Unknown')}: {p.get('total_courses', p.get('course_count', 0))} courses" for p in sorted_providers[:5]])
        
        context = f"""Analyze the course provider ecosystem:

USER QUERY: "{query}"

PROVIDER LANDSCAPE:
- Total providers: {len(results)}
- Total courses available: {total_courses}
- Average courses per provider: {avg_courses:.1f}

MAJOR PROVIDERS:
{providers_text}

Provide analysis covering:
1. Platform diversity and market coverage
2. Specialization patterns among providers
3. Quality and accessibility considerations
4. Pricing and business model insights
5. Recommendations for platform selection

Use markdown formatting. Generate comprehensive provider analysis:"""

        try:
            llm = OllamaLLM(model=self.llm_model)
            analysis = llm.invoke([{"role": "user", "content": context}])
            return re.sub(r'<think>.*?</think>', '', analysis, flags=re.DOTALL).strip()
        except Exception as e:
            # Build fallback text separately
            providers_fallback = '\n'.join([f"• **{p.get('provider_name', 'Unknown')}**: {p.get('total_courses', p.get('course_count', 0))} courses" for p in sorted_providers[:3]])
            
            return f"""## 🌐 Course Provider Landscape

Found **{len(results)} providers** offering **{total_courses} total courses**. The platform ecosystem shows {"strong diversity" if len(results) > 3 else "focused offerings"} in educational delivery.

### 🔝 **Leading Platforms**
{providers_fallback}

### 📊 **Market Coverage**
The provider landscape demonstrates comprehensive coverage with an average of **{avg_courses:.1f} courses per platform**."""

    def _analyze_review_results(self, results: List[Dict], query: str) -> str:
        """Enhanced review analysis"""
        if not results:
            return "No reviews found matching your query."
        
        # Extract ratings and sentiment
        ratings = []
        positive_reviews = 0
        total_reviews = len(results)
        
        for review in results:
            rating = review.get("review_rating", review.get("r.rating", 0))
            try:
                if rating:
                    rating_val = float(rating)
                    ratings.append(rating_val)
                    if rating_val >= 4.0:
                        positive_reviews += 1
            except:
                pass
        
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        positive_percent = (positive_reviews / total_reviews) * 100 if total_reviews > 0 else 0
        
        # Build text parts separately
        reviews_text = '\n'.join([f"- \"{r.get('review_comment', r.get('r.comment', 'No comment'))[:100]}...\"" for r in results[:3] if r.get('review_comment', r.get('r.comment'))])
        
        context = f"""Analyze student feedback patterns:

USER QUERY: "{query}"

REVIEW LANDSCAPE:
- Total reviews analyzed: {total_reviews}
- Average rating: {avg_rating:.1f}/5.0
- Positive reviews (4+ stars): {positive_percent:.1f}%

SAMPLE REVIEW COMMENTS:
{reviews_text}

Provide analysis covering:
1. Overall satisfaction trends
2. Common praise points and concerns
3. Quality patterns across courses
4. Learning outcome feedback
5. Recommendations based on student feedback

Generate comprehensive review analysis:"""
        
        try:
            llm = OllamaLLM(model=self.llm_model)
            analysis = llm.invoke([{"role": "user", "content": context}])
            return re.sub(r'<think>.*?</think>', '', analysis, flags=re.DOTALL).strip()
        except Exception as e:
            return f"""## ⭐ Student Feedback Landscape

Analyzed **{total_reviews} reviews** with an average rating of **{avg_rating:.1f}/5.0**. Student satisfaction shows **{positive_percent:.1f}%** positive feedback (4+ stars).

### 📊 **Satisfaction Trends**
The review data indicates {"excellent" if avg_rating > 4.5 else "strong" if avg_rating > 4.0 else "moderate"} learner satisfaction across courses in this area.

### 💡 **Key Insights**
Student feedback suggests {"consistent quality" if positive_percent > 80 else "variable experiences"} in course delivery and learning outcomes."""

    def _analyze_subject_results(self, results: List[Dict], query: str) -> str:
        """Enhanced subject analysis"""
        if not results:
            return "No subjects found matching your query."
        
        # Collect subject data
        course_counts = []
        total_courses = 0
        
        for subject in results:
            courses = subject.get("courses_count", subject.get("course_count", 0))
            try:
                if courses:
                    count = int(courses)
                    course_counts.append(count)
                    total_courses += count
            except:
                pass
        
        avg_courses = sum(course_counts) / len(course_counts) if course_counts else 0
        sorted_subjects = sorted(results, key=lambda x: int(x.get("courses_count", x.get("course_count", 0)) or 0), reverse=True)
        
        # Build text parts separately
        subjects_text = '\n'.join([f"- {s.get('subject_name', s.get('sub.name', 'Unknown'))}: {s.get('courses_count', s.get('course_count', 0))} courses" for s in sorted_subjects[:5]])
        
        context = f"""Analyze the subject area landscape:

USER QUERY: "{query}"

SUBJECT LANDSCAPE:
- Total subjects: {len(results)}
- Total courses across subjects: {total_courses}
- Average courses per subject: {avg_courses:.1f}

TOP SUBJECTS BY COURSE COUNT:
{subjects_text}

Provide analysis covering:
1. Subject area coverage and breadth
2. Popular vs niche specializations
3. Learning pathway connections
4. Career relevance of different subjects
5. Recommendations for subject exploration

Generate comprehensive subject landscape analysis:"""

        try:
            llm = OllamaLLM(model=self.llm_model)
            analysis = llm.invoke([{"role": "user", "content": context}])
            return re.sub(r'<think>.*?</think>', '', analysis, flags=re.DOTALL).strip()
        except Exception as e:
            # Build fallback text separately
            subjects_fallback = '\n'.join([f"• **{s.get('subject_name', s.get('sub.name', 'Unknown'))}**: {s.get('courses_count', s.get('course_count', 0))} courses" for s in sorted_subjects[:3]])
            
            return f"""## 📚 Subject Area Landscape

Found **{len(results)} subjects** with **{total_courses} total courses** available. The subject diversity shows {"comprehensive coverage" if len(results) > 5 else "focused specialization"}.

### 🔝 **Popular Subject Areas**
{subjects_fallback}

### 📊 **Coverage Analysis**
The subject landscape offers an average of **{avg_courses:.1f} courses per subject area**."""

    def _analyze_skill_results(self, results: List[Dict], query: str) -> str:
        """Enhanced skill analysis"""
        if not results:
            return "No skills found matching your query."
        
        # Collect skill data
        course_counts = []
        total_courses = 0
        
        for skill in results:
            courses = skill.get("courses_teaching", skill.get("course_count", 0))
            try:
                if courses:
                    count = int(courses)
                    course_counts.append(count)
                    total_courses += count
            except:
                pass
        
        sorted_skills = sorted(results, key=lambda x: int(x.get("courses_teaching", x.get("course_count", 0)) or 0), reverse=True)
        
        # Build text parts separately
        skills_text = '\n'.join([f"- {s.get('skill_name', s.get('s.name', 'Unknown'))}: {s.get('courses_teaching', s.get('course_count', 0))} courses" for s in sorted_skills[:5]])
        
        context = f"""Analyze the skill development landscape:

USER QUERY: "{query}"

SKILL LANDSCAPE:
- Total skills available: {len(results)}
- Total courses teaching these skills: {total_courses}
- Skills range from fundamental to advanced

TOP SKILLS BY COURSE AVAILABILITY:
{skills_text}

Provide analysis covering:
1. Skill demand and market relevance
2. Learning progression from basic to advanced
3. Career pathway connections
4. Industry alignment and job market value
5. Recommendations for skill development strategy

Generate comprehensive skill landscape analysis:"""

        try:
            llm = OllamaLLM(model=self.llm_model)
            analysis = llm.invoke([{"role": "user", "content": context}])
            return re.sub(r'<think>.*?</think>', '', analysis, flags=re.DOTALL).strip()
        except Exception as e:
            # Build fallback text separately
            skills_fallback = '\n'.join([f"• **{s.get('skill_name', s.get('s.name', 'Unknown'))}**: {s.get('courses_teaching', s.get('course_count', 0))} courses" for s in sorted_skills[:3]])
            
            return f"""## 🎯 Skill Development Landscape

Found **{len(results)} skills** with **{total_courses} courses** available for skill development. The skill ecosystem shows {"comprehensive coverage" if len(results) > 5 else "focused specialization"}.

### 🔝 **High-Demand Skills**
{skills_fallback}

### 💼 **Market Relevance**
The skill landscape reflects {"strong industry alignment" if total_courses > 50 else "targeted specialization"} with current market demands."""

    def _analyze_level_results(self, results: List[Dict], query: str) -> str:
        """Enhanced level analysis"""
        if not results:
            return "No difficulty levels found matching your query."
        
        course_counts = []
        total_courses = 0
        
        for level in results:
            courses = level.get("courses_count", level.get("course_count", 0))
            try:
                if courses:
                    count = int(courses)
                    course_counts.append(count)
                    total_courses += count
            except:
                pass
        
        sorted_levels = sorted(results, key=lambda x: int(x.get("courses_count", x.get("course_count", 0)) or 0), reverse=True)
        
        # Build text parts separately
        levels_text = '\n'.join([f"• **{level.get('level_name', level.get('l.name', 'Unknown'))}**: {level.get('courses_count', level.get('course_count', 0))} courses" for level in sorted_levels])
        
        return f"""## 📊 Learning Level Distribution

Found **{len(results)} difficulty levels** with **{total_courses} total courses** distributed across skill levels.

### 🎯 **Level Breakdown**
{levels_text}

### 📈 **Learning Progression**
The level distribution shows {"comprehensive pathway coverage" if len(results) >= 3 else "focused level targeting"}, enabling learners to progress systematically through skill development stages.

### 💡 **Selection Guidance**
The variety in difficulty levels ensures learners can find appropriate entry points and advancement opportunities based on their current expertise."""

    def _analyze_statistical_results(self, results: List[Dict], query: str) -> str:
        """Enhanced statistical analysis"""
        if not results:
            return "No statistical data found matching your query."
        
        # Extract key metrics
        metrics = {}
        for result in results:
            for key, value in result.items():
                if key not in ['similarity', 'source', 'skills', 'subjects', 'query_type']:
                    if isinstance(value, (int, float)):
                        metrics[key] = metrics.get(key, [])
                        metrics[key].append(value)
        
        # Build text parts separately
        metrics_text = '\n'.join([f"• **{key.replace('_', ' ').title()}**: {len(values)} data points (avg: {sum(values)/len(values):.1f})" for key, values in metrics.items() if values])
        
        return f"""## 📈 Statistical Analysis

Found **{len(results)} data points** with comprehensive metrics across the educational landscape.

### 📊 **Key Metrics**
{metrics_text}

### 🔍 **Data Insights**
The statistical analysis reveals {"diverse patterns" if len(metrics) > 3 else "focused trends"} in educational data, providing quantitative insights into course quality, availability, and learner engagement.

### 💡 **Analytical Value**
These statistics enable data-driven decision making for course selection and educational planning based on measurable outcomes and performance indicators."""
    
    def _analyze_course_results(self, results: List[Dict], query: str) -> str:
        """Enhanced course analysis with comprehensive overview"""
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
        
        # Enhanced statistics collection
        all_ratings = [safe_course_rating(c) for c in results]
        valid_ratings = [r for r in all_ratings if r > 0]
        avg_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else 0
        rating_range = f"{min(valid_ratings):.1f}-{max(valid_ratings):.1f}" if len(valid_ratings) > 1 else f"{valid_ratings[0]:.1f}" if valid_ratings else "N/A"
        
        # Collect comprehensive data
        levels = {}
        providers = {}
        skills_all = []
        durations = []
        organizations = []
        
        for course in results:
            # Levels
            level = course.get("level", "")
            if level and level != "N/A" and level.strip():
                levels[level] = levels.get(level, 0) + 1
            
            # Providers    
            provider = course.get("provider", "")
            if provider and provider != "N/A" and provider.strip():
                providers[provider] = providers.get(provider, 0) + 1
            
            # Skills
            course_skills = course.get("skills", [])
            if isinstance(course_skills, list):
                skills_all.extend(course_skills)
            
            # Duration analysis
            duration = course.get("duration", "")
            if duration and isinstance(duration, (int, float)):
                durations.append(duration)
            elif duration and "week" in str(duration).lower():
                try:
                    weeks = int(''.join(filter(str.isdigit, str(duration))))
                    durations.append(weeks)
                except:
                    pass
            
            # Organizations
            org = course.get("organization", "")
            if org and org != "N/A" and org.strip():
                organizations.append(org)
        
        # Advanced analytics
        from collections import Counter
        skill_counts = Counter(skills_all)
        top_skills = [skill for skill, count in skill_counts.most_common(5)]
        dominant_level = Counter(levels).most_common(1)[0][0] if levels else "Mixed"
        unique_providers = len(set(providers)) if providers else 0
        unique_orgs = len(set(organizations)) if organizations else 0
        avg_duration = sum(durations) / len(durations) if durations else None
        
        # Get top courses for highlighting
        top_rated = [(course, rating) for course, rating in valid_courses if rating > 3.5][:5]
        
        # Build text parts separately for f-string
        skills_text = '\n'.join([f"- {spec}" for spec in top_skills[:3]])
        courses_text = '\n'.join([f"- {course.get('name', 'Unknown')} ({rating:.1f}/5.0)" for course, rating in top_rated])
        level_breakdown = ", ".join([f"{level}: {count} courses" for level, count in sorted(levels.items(), key=lambda x: x[1], reverse=True)[:3]])
        provider_breakdown = ", ".join([f"{provider} ({count} courses)" for provider, count in sorted(providers.items(), key=lambda x: x[1], reverse=True)[:3]])
        
        # Build comprehensive LLM prompt
        context = f"""You are an educational course analyst. Provide a comprehensive overview of the course landscape based on this data:

USER QUERY: "{query}"

DATASET OVERVIEW:
- Total courses found: {len(results)}
- Average rating: {avg_rating:.1f}/5.0 (range: {rating_range})
- Course duration: {"averaging " + str(round(avg_duration, 1)) + " weeks" if avg_duration else "varying durations"}
- Level distribution: {dominant_level} level predominant ({len(levels)} different levels available)
- Provider diversity: {unique_providers} unique providers
- Institution diversity: {unique_orgs} unique organizations

TOP SKILLS COVERED:
{skills_text}

TOP-RATED COURSES:
{courses_text}

LEVEL BREAKDOWN:
{level_breakdown}

TOP PROVIDERS:
{provider_breakdown}

ANALYSIS REQUIREMENTS:
1. Start with an engaging overview paragraph about the course landscape for this topic
2. Analyze the quality distribution and what it means for learners
3. Discuss the skill focus areas and learning progression paths
4. Compare different course approaches, levels, or specializations
5. Highlight any notable patterns in providers or institutional offerings
6. End with practical selection insights for learners

FORMATTING:
- Use markdown headers and structure (## for main sections, **bold** for emphasis)
- Include relevant emojis for sections
- Keep it informative but engaging
- Aim for 4-6 paragraphs with clear sections
- Bold key statistics and important insights
- Focus on analytical insights, not just listing data

Generate a comprehensive course landscape analysis:"""

        # Get LLM analysis
        try:
            llm = OllamaLLM(model=self.llm_model)
            analysis = llm.invoke([{"role": "user", "content": context}])
            # Clean artifacts and thinking tags
            analysis = re.sub(r'<think>.*?</think>', '', analysis, flags=re.DOTALL)
            analysis = analysis.strip()
            
            # If analysis is too short or failed, use enhanced fallback
            if len(analysis) < 200 or "couldn't" in analysis.lower() or "I don't" in analysis:
                analysis = self._generate_enhanced_course_fallback(results, query, {
                    'total': len(results),
                    'avg_rating': avg_rating,
                    'rating_range': rating_range,
                    'top_skills': top_skills[:3],
                    'top_courses': top_rated[:3],
                    'dominant_level': dominant_level,
                    'top_providers': sorted(providers.items(), key=lambda x: x[1], reverse=True)[:3]
                })
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLM course analysis failed: {e}")
            return self._generate_enhanced_course_fallback(results, query, {
                'total': len(results),
                'avg_rating': avg_rating,
                'top_skills': top_skills[:3],
                'top_courses': top_rated[:3],
                'dominant_level': dominant_level
            })

    def _generate_enhanced_course_fallback(self, results: List[Dict], query: str, stats: dict) -> str:
        """Generate rich fallback analysis when LLM fails"""
        
        # Extract top courses for highlighting
        top_courses = stats.get('top_courses', [])
        course_highlights = []
        for course, rating in top_courses:
            name = course.get('name', 'Unknown')
            course_highlights.append(f"**{name}** ({rating:.1f}/5.0)")
        
        # Extract provider info
        top_providers = stats.get('top_providers', [])
        provider_info = []
        for provider, count in top_providers:
            provider_info.append(f"**{provider}** ({count} courses)")
        
        # Build enhanced structured analysis
        subject_area = "programming" if any(word in query.lower() for word in ["python", "java", "programming", "code"]) else "educational"
        
        # Build text parts separately
        course_highlights_text = '\n'.join([f"• {highlight}" for highlight in course_highlights])
        
        analysis = f"""## 🎓 {subject_area.title()} Course Landscape

I found **{stats['total']} courses** in this area with strong quality indicators. The landscape shows an average rating of **{stats['avg_rating']:.1f}/5.0**, indicating {"excellent" if stats['avg_rating'] > 4.5 else "high" if stats['avg_rating'] > 4.0 else "good"} learner satisfaction across the board.

### 📊 **Quality & Standards**
The course ecosystem demonstrates {"excellent" if stats['avg_rating'] > 4.5 else "solid"} standards with ratings {stats.get('rating_range', 'in the high range')}. This consistency suggests learners have access to well-designed educational content regardless of their chosen path.

### 🎯 **Core Learning Focus**
The most emphasized skills across these courses include **{', '.join(stats['top_skills'])}**, reflecting the practical, hands-on approach most courses take. This alignment indicates strong focus on real-world application and job-relevant skills.

### ⭐ **Top-Rated Offerings**
Among the highest-rated courses:
{course_highlights_text}

### 🏫 **Provider Landscape**
{f"Leading course providers include {', '.join(provider_info[:2])}, showing diversity in educational approaches." if provider_info else "Multiple providers offer courses in this area, ensuring variety in teaching styles and approaches."}

### 💡 **Selection Insights**
The course landscape shows strong consistency in quality, with most offerings targeting **{stats['dominant_level'].lower()}** level learners. This {"diversity" if "mixed" in stats['dominant_level'].lower() else "focus"} means learners can find appropriate entry points matching their current skill level and learning goals.

The comprehensive coverage of fundamental concepts across these courses suggests that any choice from the top-rated options would provide solid foundation-building opportunities."""
        
        return analysis.strip()

    def _analyze_mixed_results(self, results: List[Dict], query: str) -> str:
        """Enhanced mixed results analysis"""
        if not results:
            return "No results found matching your query."
        
        # Categorize different types of entities in results
        entity_types = {
            'courses': [],
            'instructors': [],
            'organizations': [],
            'providers': [],
            'reviews': [],
            'subjects': [],
            'skills': [],
            'levels': [],
            'statistical': []
        }
        
        for result in results:
            # Determine entity type based on keys present
            if "url" in result or "course_name" in result:
                entity_types['courses'].append(result)
            elif any(key in result for key in ["instructor_name", "instructor_rating", "i.name"]):
                entity_types['instructors'].append(result)
            elif any(key in result for key in ["organization_name", "organization_description", "o.name"]):
                entity_types['organizations'].append(result)
            elif any(key in result for key in ["provider_name", "provider_description", "p.name"]):
                entity_types['providers'].append(result)
            elif any(key in result for key in ["review_comment", "review_rating", "r.comment"]):
                entity_types['reviews'].append(result)
            elif any(key in result for key in ["subject_name", "subject_description", "sub.name"]):
                entity_types['subjects'].append(result)
            elif any(key in result for key in ["skill_name", "skill_description", "s.name"]):
                entity_types['skills'].append(result)
            elif any(key in result for key in ["level_name", "level_description", "l.name"]):
                entity_types['levels'].append(result)
            elif any(key in result for key in ["avg_rating", "total_courses", "course_count"]):
                entity_types['statistical'].append(result)
        
        # Filter out empty categories
        non_empty_types = {k: v for k, v in entity_types.items() if v}
        
        # Build text parts separately
        entity_breakdown_text = '\n'.join([f"- {entity_type.title()}: {len(entities)} items" for entity_type, entities in non_empty_types.items()])
        
        # Build comprehensive mixed analysis
        context = f"""You are an educational analyst. Analyze this mixed dataset of educational entities:

USER QUERY: "{query}"

MIXED DATASET OVERVIEW:
- Total results: {len(results)}
- Entity types found: {len(non_empty_types)}

ENTITY BREAKDOWN:
{entity_breakdown_text}

ANALYSIS REQUIREMENTS:
1. Overview of the diverse educational landscape found
2. How different entity types complement each other
3. Patterns and connections between different educational components
4. Quality and diversity insights across entity types
5. Comprehensive guidance for navigating this mixed educational ecosystem

FORMATTING:
- Use markdown headers with emojis
- Bold key insights and statistics
- Structure analysis by entity relationships
- Provide actionable insights for learners

Generate comprehensive mixed entity analysis:"""

        try:
            llm = OllamaLLM(model=self.llm_model)
            analysis = llm.invoke([{"role": "user", "content": context}])
            analysis = re.sub(r'<think>.*?</think>', '', analysis, flags=re.DOTALL).strip()
            
            if len(analysis) < 200:
                return self._generate_mixed_fallback(results, query, non_empty_types)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Mixed analysis failed: {e}")
            return self._generate_mixed_fallback(results, query, non_empty_types)

    def _generate_mixed_fallback(self, results: List[Dict], query: str, entity_types: dict) -> str:
        """Generate fallback for mixed results"""
        
        # ✅ FIX: Build text parts separately
        entity_distribution_text = '\n'.join([f"• **{entity_type.title()}**: {len(entities)} items" for entity_type, entities in entity_types.items()])
        
        analysis = f"""## 🌐 Comprehensive Educational Landscape

Found **{len(results)} educational entities** spanning **{len(entity_types)} different categories**, providing a comprehensive view of the educational ecosystem.

### 📊 **Entity Distribution**
{entity_distribution_text}

### 🔗 **Ecosystem Connections**
This diverse dataset reveals the interconnected nature of educational resources, spanning from individual courses and instructors to institutional providers and student feedback systems.

### 💡 **Navigation Insights**
The variety of entity types suggests a mature educational landscape with multiple pathways for learning and skill development. {"This diversity provides learners with comprehensive options for course selection, instructor evaluation, and institutional choice." if len(entity_types) > 3 else "The focused entity types indicate specialized educational offerings in this area."}

### 🎯 **Strategic Recommendations**
Consider exploring the connections between these different educational components to make informed decisions about learning pathways and educational investments."""
        
        return analysis.strip()

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