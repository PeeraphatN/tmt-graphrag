import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import GraphRAGPipeline

def test_pipeline():
    print("Initializing Pipeline...")
    try:
        pipeline = GraphRAGPipeline()
        print("Pipeline Initialized.")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    question = "ยา paracetamol แก้ อะไร"
    print(f"\nTesting Query: '{question}'")
    
    try:
        answer = pipeline.run(question)
        print("\nFinal Answer:")
        print(answer)
    except Exception as e:
        print(f"\nFailed to run pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
