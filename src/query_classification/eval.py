import json
import logging
import time
from typing import List, Dict, Any
from neo4j import GraphDatabase
from graph_rag_1cypher import QueryProcessor  # Import module xử lý query của bạn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class CypherEvaluator:
    def __init__(self, test_data_path: str, neo4j_config: dict):
        self.test_data = self.load_test_data(test_data_path)
        self.processor = QueryProcessor()
        self.driver = GraphDatabase.driver(**neo4j_config)
        self.results = []

    def load_test_data(self, path: str) -> List[Dict]:
        """Load dữ liệu test từ file JSON"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                required_fields = ["query_id", "natural_language_query", "cypher_query"]
                for item in data:
                    if not all(field in item for field in required_fields):
                        raise ValueError("Invalid test data format")
                return data
        except Exception as e:
            logger.error(f"Lỗi đọc file test data: {str(e)}")
            return []

    def execute_cypher(self, query: str) -> List[Any]:
        """Thực thi Cypher query và trả về kết quả đã chuẩn hóa"""
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return self.normalize_result(result)
        except Exception as e:
            logger.error(f"Lỗi thực thi query: {str(e)}")
            return None

    def normalize_result(self, result) -> List[Any]:
        """Chuẩn hóa kết quả từ Neo4j thành dạng có thể so sánh được"""
        processed = []
        for record in result:
            # Chuyển đổi Node thành dictionary
            if hasattr(record[0], 'items'):  # Kiểm tra nếu là Node
                processed.append(dict(record[0].items()))
            else:
                processed.append(record[0])
        return sorted(processed, key=lambda x: str(x))

    def generate_query(self, natural_query: str) -> Dict:
        """Sinh Cypher query từ câu hỏi tự nhiên"""
        try:
            plan = self.processor.generate_query_plan(natural_query)
            return {
                "query": plan.get("final_query", ""),
                "status": "generation_success" if plan.get("final_query") else "generation_failed"
            }
        except Exception as e:
            logger.error(f"Lỗi sinh query: {str(e)}")
            return {"query": "", "status": f"generation_error: {str(e)}"}

    def run_evaluation(self):
        """Chạy toàn bộ quá trình đánh giá"""
        logger.info("Bắt đầu đánh giá...")
        
        for case in self.test_data:
            case_result = {
                "query_id": case["query_id"],
                "natural_language_query": case["natural_language_query"],
                "cypher_query": case["cypher_query"],
                "expected_results": [],
                "generated_query": "",
                "generated_results": [],
                "status": "pending",
                "match": False,
                "error": None
            }

            try:
                logger.info(f"Xử lý case {case['query_id']}...")

                # Bước 1: Thực thi query mẫu để lấy expected_results
                expected = self.execute_cypher(case["cypher_query"])
                if expected is None:
                    case_result["status"] = "expected_execution_failed"
                    self.results.append(case_result)
                    continue
                case_result["expected_results"] = expected

                # Bước 2: Sinh query từ câu hỏi tự nhiên
                gen_result = self.generate_query(case["natural_language_query"])
                case_result["generated_query"] = gen_result["query"]
                case_result["status"] = gen_result["status"]
                
                if gen_result["status"] != "generation_success":
                    self.results.append(case_result)
                    continue

                # Bước 3: Thực thi query được sinh
                generated = self.execute_cypher(gen_result["query"])
                if generated is None:
                    case_result["status"] = "generated_execution_failed"
                    self.results.append(case_result)
                    continue
                case_result["generated_results"] = generated

                # Bước 4: So sánh kết quả
                case_result["match"] = (generated == expected)
                case_result["status"] = "full_match" if case_result["match"] else "partial_match"

            except Exception as e:
                case_result["status"] = f"unexpected_error: {str(e)}"
                case_result["error"] = str(e)
                logger.error(f"Lỗi không xác định với case {case['query_id']}: {str(e)}")

            finally:
                self.results.append(case_result)
                self.log_case_status(case_result)

    def log_case_status(self, case: Dict):
        """Ghi log chi tiết trạng thái xử lý"""
        logger.info(f"""
            [Case {case['query_id']}] {case['status']}
            Expected results: {len(case['expected_results'])}
            Generated results: {len(case['generated_results'])}
            Match: {case['match']}
            Generated query: {case['generated_query'][:100]}...
            Error: {case['error'] or 'None'}
        """)

    def save_results(self, output_path: str):
        """Lưu kết quả đánh giá"""
        output = []
        for res in self.results:
            output.append({
                "query_id": res["query_id"],
                "natural_language_query": res["natural_language_query"],
                "cypher_query": res["cypher_query"],
                "expected_results": res["expected_results"],
                "status": res["status"],
                "match": res["match"]
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"Đã lưu kết quả vào {output_path}")

if __name__ == "__main__":
    evaluator = CypherEvaluator(
        test_data_path="D:\\Thesis\\Courses-Searching\\course_searching_main\\data\\test40_queries.json",
        neo4j_config={
            "uri": "bolt://localhost:7687",
            "auth": ("neo4j", "12345678")
        }
    )
    
    evaluator.run_evaluation()
    evaluator.save_results("D:\\Thesis\\Courses-Searching\\course_searching_main\\data\\evaluation_results.json")
    
    # Thống kê
    success = sum(1 for r in evaluator.results if r["status"] == "full_match")
    total = len(evaluator.results)
    print(f"\nKết quả: {success}/{total} truy vấn chính xác ({success/total:.1%})")