from neo4j import GraphDatabase
import json

class Neo4jCourseGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def setup_constraints(self):
        """Tạo các ràng buộc duy nhất để tránh trùng lặp dữ liệu."""
        with self.driver.session() as session:
            # Cập nhật write_transaction thành execute_write để tránh cảnh báo
            session.execute_write(self._create_constraints)

    @staticmethod
    def _create_constraints(tx):
        """Tạo ràng buộc duy nhất cho các nút."""
        tx.run("CREATE CONSTRAINT course_url IF NOT EXISTS FOR (c:Course) REQUIRE c.url IS UNIQUE")
        tx.run("CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE")
        tx.run("CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE")
        tx.run("CREATE CONSTRAINT instructor_name IF NOT EXISTS FOR (i:Instructor) REQUIRE i.name IS UNIQUE")
        tx.run("CREATE CONSTRAINT level_name IF NOT EXISTS FOR (l:Level) REQUIRE l.name IS UNIQUE")

    def load_courses_from_json(self, json_file):
        """Tải dữ liệu khóa học từ tệp JSON và tạo đồ thị trong Neo4j."""
        with open(json_file, 'r', encoding='utf-8') as f:
            courses = json.load(f)
        
        with self.driver.session() as session:
            for course in courses:
                session.execute_write(self._create_course_node, course)
                session.execute_write(self._create_skill_nodes_and_relationships, course)
                session.execute_write(self._create_organization_node_and_relationship, course)
                session.execute_write(self._create_instructor_node_and_relationship, course)
                session.execute_write(self._create_level_node_and_relationship, course)

    @staticmethod
    def _create_course_node(tx, course):
        """Tạo nút Course trong Neo4j."""
        query = (
            "MERGE (c:Course {url: $url}) "
            "SET c.name = $name, c.duration = $duration, "
            "c.rating = $rating, c.description = $description"
        )
        tx.run(query, 
               url=course.get('url', ''),
               name=course.get('course_name', ''),
               duration=course.get('Duration', 0),
               rating=course.get('rating', 'No rating'),
               description=course.get('description', ''))

    @staticmethod
    def _create_skill_nodes_and_relationships(tx, course):
        """Tạo nút Skill và mối quan hệ TEACHES."""
        skills = course.get('skills', [])
        course_url = course.get('url', '')
        for skill in skills:
            query = (
                "MERGE (s:Skill {name: $skill}) "
                "MERGE (c:Course {url: $course_url}) "
                "MERGE (c)-[:TEACHES]->(s)"
            )
            tx.run(query, skill=skill, course_url=course_url)

    @staticmethod
    def _create_organization_node_and_relationship(tx, course):
        """Tạo nút Organization và mối quan hệ OFFERED_BY."""
        organizations = course.get('organization', [])
        course_url = course.get('url', '')
        for org in organizations:
            query = (
                "MERGE (o:Organization {name: $org}) "
                "MERGE (c:Course {url: $course_url}) "
                "MERGE (c)-[:OFFERED_BY]->(o)"
            )
            tx.run(query, org=org, course_url=course_url)

    @staticmethod
    def _create_instructor_node_and_relationship(tx, course):
        """Tạo nút Instructor và mối quan hệ TAUGHT_BY."""
        instructor = course.get('instructor', '')
        course_url = course.get('url', '')
        if instructor:
            query = (
                "MERGE (i:Instructor {name: $instructor}) "
                "MERGE (c:Course {url: $course_url}) "
                "MERGE (c)-[:TAUGHT_BY]->(i)"
            )
            tx.run(query, instructor=instructor, course_url=course_url)

    @staticmethod
    def _create_level_node_and_relationship(tx, course):
        """Tạo nút Level và mối quan hệ HAS_LEVEL."""
        level = course.get('level', 'Intermediate')
        course_url = course.get('url', '')
        query = (
            "MERGE (l:Level {name: $level}) "
            "MERGE (c:Course {url: $course_url}) "
            "MERGE (c)-[:HAS_LEVEL]->(l)"
        )
        tx.run(query, level=level, course_url=course_url)

if __name__ == "__main__":
    neo4j_graph = Neo4jCourseGraph("neo4j://localhost:7687", "neo4j", "12345678")
    
    neo4j_graph.setup_constraints()
    
    neo4j_graph.load_courses_from_json("D:\\Thesis\\Courses-Searching\\course_searching_main\\data\\combined_dataset.json")
    
    neo4j_graph.close()
