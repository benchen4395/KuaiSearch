#!/usr/bin/env python3
"""
Unified Data Processing Pipeline
Execution Order:
1. process.py - Process raw data to generate queries and qrels
2. build_dpr.py - Build DPR training data
3. construct_i2q.py - Construct docTquery training data
4. title_embedding_prepare.py - Prepare product title embeddings
5. faiss_kmeans_quantize.py - FAISS K-means quantization
6. process4gr.py - Build query-to-code mappings
"""

import subprocess
import sys
import os
from datetime import datetime


def run_step(step_name: str, script_path: str) -> bool:
    """
    Execute a single processing step with detailed logging.
    
    Args:
        step_name: Descriptive name of the step (for logging)
        script_path: Path to the Python script to execute
        
    Returns:
        True if step completed successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Step: {step_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script: {script_path}")
    print(f"{'='*60}\n")
    
    try:
        # Execute script and stream output in real-time
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True,
            capture_output=False
        )
        
        print(f"\n✅ {step_name} completed successfully\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {step_name} failed")
        print(f"Error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {step_name} encountered an exception: {str(e)}")
        return False


def main():
    """Main pipeline execution function"""
    print("\n" + "="*60)
    print("Starting Data Processing Pipeline")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Define processing steps (in execution order)
    steps = [
        ("1. Process raw data (process.py)", "recall/data/process.py"),
        ("2. Build DPR training data (build_dpr.py)", "recall/data/build_dpr.py"),
        ("3. Construct docTquery data (construct_i2q.py)", "recall/data/construct_i2q.py"),
        ("4. Prepare product title embeddings (title_embedding_prepare.py)", "recall/data/title_embedding_prepare.py"),
        ("5. FAISS K-means quantization (faiss_kmeans_quantize.py)", "recall/data/faiss_kmeans_quantize.py"),
        ("6. Build query-code mappings (process4gr.py)", "recall/data/process4gr.py"),
    ]
    
    # Pre-flight check: Verify all scripts exist
    missing_scripts = []
    for step_name, script_path in steps:
        if not os.path.exists(script_path):
            missing_scripts.append(script_path)
    
    if missing_scripts:
        print("\n❌ Error: The following script files are missing:")
        for script in missing_scripts:
            print(f"  - {script}")
        sys.exit(1)
    
    # Execute steps in sequence (halt on first failure)
    success_count = 0
    failed_step = None
    
    for step_name, script_path in steps:
        if run_step(step_name, script_path):
            success_count += 1
        else:
            failed_step = step_name
            break
    
    # Final summary report
    print("\n" + "="*60)
    print("Pipeline Execution Complete")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful steps: {success_count}/{len(steps)}")
    
    if failed_step:
        print(f"❌ Failed at: {failed_step}")
        print("="*60 + "\n")
        sys.exit(1)
    else:
        print("✅ All steps completed successfully!")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()