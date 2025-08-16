#!/usr/bin/env python3
"""
Simple test script for the literature analysis system
"""

import asyncio
import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
    except ImportError as e:
        print(f"❌ matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ seaborn")
    except ImportError as e:
        print(f"❌ seaborn: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn")
    except ImportError as e:
        print(f"❌ scikit-learn: {e}")
        return False
    
    try:
        import networkx as nx
        print("✅ networkx")
    except ImportError as e:
        print(f"❌ networkx: {e}")
        return False
    
    try:
        from docx import Document
        print("✅ python-docx")
    except ImportError as e:
        print(f"❌ python-docx: {e}")
        return False
    
    try:
        import anthropic
        print("✅ anthropic")
    except ImportError as e:
        print(f"❌ anthropic: {e}")
        return False
    
    return True

async def test_analysis():
    """Test the analysis system with a small sample."""
    print("\n🔬 Testing analysis system...")
    
    # Test data file
    csv_file = "/Users/v/solomonresearch/blockchain sna litrev/RIS files/SNA Blockchain - Filtered all.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ Input file not found: {csv_file}")
        return False
    
    # Import the analysis system
    try:
        from literature_analysis import LiteratureAnalyzer
        print("✅ Analysis system imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import analysis system: {e}")
        return False
    
    # Create analyzer
    try:
        analyzer = LiteratureAnalyzer(csv_file, batch_size=2)  # Very small batch for testing
        print("✅ Analyzer created successfully")
    except Exception as e:
        print(f"❌ Failed to create analyzer: {e}")
        return False
    
    # Test data loading
    try:
        if analyzer.load_and_validate_data():
            print("✅ Data loaded successfully")
            print(f"   • Total papers: {len(analyzer.df)}")
            print(f"   • Columns: {list(analyzer.df.columns)}")
        else:
            print("❌ Failed to load data")
            return False
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False
    
    print("✅ All tests passed! System is ready to run.")
    return True

async def main():
    """Main test function."""
    print("🧪 Literature Analysis System Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing packages.")
        return
    
    # Test analysis system
    if await test_analysis():
        print("\n🎉 All tests passed! You can now run the full analysis.")
        print("\n🚀 To run the full analysis:")
        print("python literature_analysis.py")
    else:
        print("\n❌ Analysis test failed. Check error messages above.")

if __name__ == "__main__":
    asyncio.run(main())