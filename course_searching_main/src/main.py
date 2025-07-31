from models.learning_assistant import LearningAssistant
import time
from datetime import datetime 
import json

def main():
    # Initialize with caching
    start_time = time.time()
    assistant = LearningAssistant(
        data_path="D:\\Thesis\\Courses-Searching\\course_searching_main\\data\\processed_courses_detail.json",
        use_caching=True
    )
    print(f"Initialization took {time.time() - start_time:.2f} seconds\n")
    
    while True:
        query = input("\nEnter your learning query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
            
        process_start = time.time()
        result = assistant.process_query(query)
        process_time = time.time() - process_start
        
        print(f"\nProcessing time: {process_time:.2f}s")
        print_results(result)

def print_results(result):
    print("\n" + "="*60)
    print(f"Results for: {result['query']}")
    print("="*60)
    
    # Print analysis
    print("\nAnalysis:")
    print(json.dumps(result['analysis'], indent=2))
    
    # Print summary
    summary = result['results']['summary']
    print("\nSummary:")
    print(f"- {summary.get('summary', 'No summary available')}")
    if 'key_insights' in summary and len(summary['key_insights']) > 0:
        print("\nKey Insights:")
        for idx, insight in enumerate(summary['key_insights'][:3], 1):
            print(f"{idx}. {insight}")
    
    # Print courses
    if len(result['results']['courses']) > 0:
        print("\nTop Courses:")
        for idx, course in enumerate(result['results']['courses'][:3], 1):
            print(f"{idx}. {course.get('title', 'Untitled Course')}")
            print(f"   Score: {course.get('validation', {}).get('relevance_score', 0):.2f}")
            print(f"   URL: {course.get('url', 'No URL available')}\n")
    else:
        print("\nNo relevant courses found.")

if __name__ == "__main__":
    main()