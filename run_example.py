"""
Simple Deep Learning Example Runner
เรียกใช้งาน deep learning example
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deep_learning_example import main

def run_example():
    """เรียกใช้งานตัวอย่าง deep learning"""
    print("🎯 Starting Deep Learning Examples...")
    print("This example will create:")
    print("  1. Sample CSV datasets")
    print("  2. Train neural networks for regression and classification")
    print("  3. Save trained models and visualizations")
    print("-" * 60)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all required packages are installed")

if __name__ == "__main__":
    run_example()
