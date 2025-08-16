#!/usr/bin/env python3
"""
Run only the descriptive analysis (before batch processing)
This gives you a quick overview of the entire dataset.
"""

import asyncio
import sys
import os

async def run_descriptive_analysis():
    """Run only the descriptive analysis part."""
    print("📊 Running Descriptive Analysis Only")
    print("=" * 50)
    
    # Import the analysis system
    from literature_analysis import LiteratureAnalyzer
    
    # Configuration
    csv_file_path = "/Users/v/solomonresearch/blockchain sna litrev/RIS files/SNA Blockchain - Filtered all.csv"
    batch_size = 20  # Won't be used for descriptive analysis
    
    # Create analyzer
    analyzer = LiteratureAnalyzer(csv_file_path, batch_size)
    
    # Load data
    if not analyzer.load_and_validate_data():
        print("❌ Failed to load data")
        return False
    
    # Run only descriptive analysis
    analyzer.perform_descriptive_analysis()
    
    print("\n🎉 Descriptive Analysis Complete!")
    print("\n📄 Output file: literature_analysis_descriptive.docx")
    print("\n💡 To run the full batch analysis, use: python literature_analysis.py")
    
    return True

if __name__ == "__main__":
    asyncio.run(run_descriptive_analysis())