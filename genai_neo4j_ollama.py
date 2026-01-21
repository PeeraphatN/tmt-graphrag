import atexit
from src.knowledge_graph import close_driver
from src.pipeline import GraphRAGPipeline

# Register cleanup
atexit.register(close_driver)

def main():
    try:
        # Initialize Pipeline (Validator + Connect DB)
        pipeline = GraphRAGPipeline()
    except Exception as e:
        print(f"Startup Error: {e}")
        return

    print("พิมพ์คำถาม หรือ 'exit' เพื่อออก")

    try:
        while True:
            q = input("\nถาม: ").strip()
            if q.lower() in ("exit", "quit"):
                pipeline.print_cache_stats()
                break
            
            # Delegate all processing to the pipeline
            pipeline.run(q)

    except (KeyboardInterrupt, EOFError):
        # Show stats on exit
        pipeline.print_cache_stats()
        print("\nออกจากโปรแกรม")

if __name__ == "__main__":
    main()