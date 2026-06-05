import sys
import os
import traceback

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("Starting test_runner.py...")
print("Working directory:", os.getcwd())
print("sys.path:", sys.path)

if __name__ == "__main__":
    try:
        print("Importing run_pipeline...")
        import run_pipeline
        print("Running end-to-end demo...")
        run_pipeline.run_end_to_end_demo()
        print("Finished demo successfully!")
    except Exception as e:
        print("=== CRITICAL EXCEPTION CAUGHT ===")
        print("Exception type:", type(e))
        print("Exception message:", str(e))
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
    except BaseException as e:
        print("=== CRITICAL BASEEXCEPTION CAUGHT ===")
        print("Exception type:", type(e))
        print("Exception message:", str(e))
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
