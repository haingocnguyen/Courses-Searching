import json
import logging
import time
from typing import List, Any, Dict
from neo4j import GraphDatabase
from neo4j.graph import Node

class CypherEvaluator:
    def __init__(self, test_data_path: str, neo4j_config: dict):
        self.test_data = self.load_test_data(test_data_path)
        self.driver = GraphDatabase.driver(**neo4j_config)
        self.results = []

    def load_test_data(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def execute_query(self, query: str) -> List[Any]:
        """Execute query and return sorted list of first values"""
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return self.process_result(result)
        except Exception as e:
            logging.error(f"Query failed: {str(e)}")
            return []

    def process_result(self, result) -> List[Any]:
        """Process Neo4j result to comparable format"""
        processed = []
        for record in result:
            value = record[0]  # Lấy giá trị đầu tiên
            processed.append(self._convert_value(value))
        return sorted(processed, key=str)

    def _convert_value(self, value: Any) -> Any:
        """Convert complex types to comparable format"""
        if isinstance(value, Node):
            return {**dict(value.items()), "id": value.id}
        return value

    def compare_results(self, generated: List[Any], expected: List[Any]) -> bool:
        """Compare two lists of results"""
        return generated == expected

    def evaluate(self):
        for case in self.test_data:
            case_result = {
                "query_id": case["query_id"],
                "status": "pending",
                "expected": [],
                "generated": [],
                "match": False
            }

            try:
                # Get expected results
                case_result["expected"] = self.execute_query(case["cypher_query"])

                # Generate and execute query
                generated_query = self.generate_query(case["natural_language_query"])
                case_result["generated"] = self.execute_query(generated_query)
                
                # Compare results
                case_result["match"] = self.compare_results(
                    case_result["generated"], 
                    case_result["expected"]
                )
                case_result["status"] = "success"

            except Exception as e:
                case_result["status"] = f"error: {str(e)}"

            self.results.append(case_result)

    def save_results(self, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    evaluator = CypherEvaluator(
        test_data_path="D:\\Thesis\\Courses-Searching\\course_searching_main\\data\\test40_queries.json",
        neo4j_config={
            "uri": "bolt://localhost:7687",
            "auth": ("neo4j", "12345678")
        }
    )
    evaluator.evaluate()
    evaluator.save_results("D:\\Thesis\\Courses-Searching\\course_searching_main\\data\\enhanced_test_queries.json")

