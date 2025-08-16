#!/usr/bin/env python3
"""
Literature Analysis Runner Script
================================

This script helps you set up and run the literature analysis system.
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def check_and_install_requirements():
    """Check and install required packages."""
    print("🔧 Checking and installing required packages...")
    
    # Install additional requirements
    requirements_file = "requirements_simple.txt"
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("✅ Additional packages installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Some packages failed to install: {e}")
    
    # Download NLTK data
    try:
        import nltk
        print("📥 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️ NLTK download warning: {e}")

def check_input_file():
    """Check if the input CSV file exists."""
    input_file = "/Users/v/solomonresearch/blockchain sna litrev/RIS files/SNA Blockchain - Filtered all.csv"
    if os.path.exists(input_file):
        print(f"✅ Input file found: {input_file}")
        return True
    else:
        print(f"❌ Input file not found: {input_file}")
        print("Please ensure the CSV file exists at the specified location.")
        return False

def check_api_key():
    """Check if Claude API key is configured."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('CLAUDE_API_KEY')
    if api_key:
        print("✅ Claude API key configured")
        return True
    else:
        print("❌ Claude API key not found in .env file")
        print("Please add CLAUDE_API_KEY to your .env file")
        return False

async def run_analysis():
    """Run the literature analysis."""
    print("🚀 Starting Literature Analysis System...")
    
    # Import and run the analyzer
    from literature_analysis import LiteratureAnalyzer
    
    # Configuration
    csv_file_path = "/Users/v/solomonresearch/blockchain sna litrev/RIS files/SNA Blockchain - Filtered all.csv"
    batch_size = 20
    
    # Create and run analyzer
    analyzer = LiteratureAnalyzer(csv_file_path, batch_size)
    success = await analyzer.run_analysis()
    
    return success

def main():
    """Main setup and execution function."""
    print("📚 Literature Review Analysis System")
    print("=" * 50)
    
    # Step 1: Check and install requirements
    check_and_install_requirements()
    
    # Step 2: Check input file
    if not check_input_file():
        print("\n❌ Setup failed: Input file not found")
        return
    
    # Step 3: Check API configuration
    if not check_api_key():
        print("\n❌ Setup failed: API key not configured")
        return
    
    print("\n✅ All checks passed! Ready to run analysis.")
    print("\n🔄 Starting batch processing...")
    
    # Step 4: Run analysis
    try:
        success = asyncio.run(run_analysis())
        
        if success:
            print("\n🎉 Analysis completed successfully!")
            print("\n📊 Output files generated:")
            print("• literature_analysis_report.docx (incremental updates)")
            print("• literature_analysis_report_final.docx (comprehensive report)")
            print("\n📍 Files saved to: /Users/v/solomonresearch/blockchain sna litrev/RIS files/")
        else:
            print("\n❌ Analysis failed. Check error messages above.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")

if __name__ == "__main__":
    main()