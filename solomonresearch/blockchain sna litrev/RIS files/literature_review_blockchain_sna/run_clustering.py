#!/usr/bin/env python3
"""
Literature Review Clustering System Launcher
Blockchain & Social Network Analysis Project

This script launches the comprehensive literature review clustering system
that processes 1095 papers using Claude, Gemini, and DeepSeek APIs.

Usage:
    python run_clustering.py [--test] [--batch-size N] [--start-batch N]

Options:
    --test          Run with first 30 papers only for testing
    --batch-size N  Set custom batch size (default: 10)
    --start-batch N Resume from specific batch number
    --config FILE   Use custom config file
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

# Add scripts directory to Python path
sys.path.append(str(Path(__file__).parent / "scripts"))

def main():
    parser = argparse.ArgumentParser(description="Literature Review Clustering System")
    parser.add_argument("--test", action="store_true", 
                       help="Run with first 30 papers only for testing")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Number of papers per batch (default: 10)")
    parser.add_argument("--start-batch", type=int, default=1,
                       help="Resume from specific batch number")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Config file path")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test API connections without processing papers")
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    print(f"""
🚀 Starting Literature Review Clustering System
{'='*60}
Configuration:
  • Test Mode: {'Yes (30 papers)' if args.test else 'No (full 1095 papers)'}
  • Batch Size: {args.batch_size} papers
  • Start Batch: {args.start_batch}
  • Config File: {args.config}
  • Dry Run: {'Yes' if args.dry_run else 'No'}
{'='*60}
    """)
    
    if args.dry_run:
        print("🧪 Running API connection test...")
        # Test API connections
        asyncio.run(test_apis())
        return
    
    if args.test:
        # Create test dataset
        create_test_dataset()
        input_file = "data/papers_input_test.csv"
        output_file = "data/papers_classified_test.csv"
    else:
        input_file = "data/papers_prepared.csv"
        output_file = "data/papers_classified.csv"
    
    # Import and run the clustering system
    try:
        from clustering_main import LiteratureReviewClustering
        
        # Initialize clustering system
        clusterer = LiteratureReviewClustering(
            input_csv=input_file,
            output_csv=output_file
        )
        
        # Override batch size if specified
        clusterer.batch_size = args.batch_size
        
        # Run the clustering
        asyncio.run(clusterer.run_clustering())
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user")
        print(f"Progress has been saved. You can resume with:")
        print(f"python run_clustering.py --start-batch {args.start_batch + 1}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print(f"Check logs/processing_log.txt for details")
        return 1
    
    return 0

def create_test_dataset():
    """Create a test dataset with first 30 papers"""
    import pandas as pd
    
    print("📝 Creating test dataset (30 papers)...")
    
    # Read full dataset  
    df = pd.read_csv("data/papers_prepared.csv")
    
    # Take first 30 papers
    test_df = df.head(30)
    
    # Save test dataset
    test_df.to_csv("data/papers_input_test.csv", index=False)
    
    print(f"✅ Test dataset created: {len(test_df)} papers")

async def test_apis():
    """Test Claude API connection"""
    from dotenv import load_dotenv
    import anthropic
    import json
    
    load_dotenv()
    
    try:
        client = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        
        test_prompt = '''You are categorizing academic papers for a systematic literature review on Blockchain and Social Network Analysis.

Paper Details:
Title: Test Paper for API Connection
Abstract: This is a test paper to verify API connectivity. It combines blockchain and social network analysis concepts for testing purposes.
Year: 2024
Source: Test

Return ONLY valid JSON:
{
    "primary_category": "Blockchain-SNA Integration",
    "secondary_category": "",
    "research_type": "theoretical",
    "data_type": "analytical",
    "gaps": ["", ""],
    "relevance_score": 8,
    "confidence": 0.9
}'''
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0.1,
            messages=[{"role": "user", "content": test_prompt}]
        )
        
        # Show raw response for debugging
        raw_response = response.content[0].text
        print(f"\nRaw API Response:")
        print(f"'{raw_response}'")
        
        # Clean JSON response (remove markdown code blocks if present)
        json_text = raw_response
        if json_text.startswith('```json'):
            json_text = json_text[7:]  # Remove ```json
        if json_text.endswith('```'):
            json_text = json_text[:-3]  # Remove ```
        json_text = json_text.strip()
        
        # Try to parse the response
        try:
            result = json.loads(json_text)
            print(f"\n📊 API Test Results:")
            print(f"  Claude: ✅ Connected")
            print(f"  Test Classification: {result.get('primary_category', 'Unknown')}")
            print(f"  Confidence: {result.get('confidence', 0)}")
            print(f"\n📈 Overall Success Rate: 100.0%")
        except json.JSONDecodeError as je:
            print(f"\n📊 API Test Results:")
            print(f"  Claude: ❌ JSON Parse Failed")
            print(f"    Raw response: '{raw_response[:100]}...'")
            print(f"    Cleaned JSON: '{json_text[:100]}...'")
            print(f"    JSON Error: {str(je)}")
            print(f"\n📈 Overall Success Rate: 0.0%")
        
    except Exception as e:
        print(f"\n📊 API Test Results:")
        print(f"  Claude: ❌ Failed")
        print(f"    Error: {str(e)}")
        print(f"\n📈 Overall Success Rate: 0.0%")

if __name__ == "__main__":
    exit(main())