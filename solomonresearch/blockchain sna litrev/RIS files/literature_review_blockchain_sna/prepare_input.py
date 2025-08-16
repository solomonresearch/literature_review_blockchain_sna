#!/usr/bin/env python3
"""
Prepare the input CSV file by mapping Zotero export columns to expected format
"""

import pandas as pd
import sys

def prepare_csv_file(input_file, output_file):
    """Prepare the CSV file with expected column names"""
    print(f"Reading {input_file}...")
    
    # Read the original CSV
    df = pd.read_csv(input_file)
    
    print(f"Original dataset: {len(df)} papers")
    print(f"Columns: {list(df.columns)}")
    
    # Create the expected format
    prepared_df = pd.DataFrame()
    
    # Map columns to expected names
    prepared_df['title'] = df.get('Title', '')
    prepared_df['abstract'] = df.get('Abstract Note', '')  
    prepared_df['authors'] = df.get('Author', '')
    prepared_df['year'] = df.get('Publication Year', '')
    prepared_df['doi'] = df.get('DOI', '')
    prepared_df['source_db'] = df.get('Library Catalog', 'Zotero')
    prepared_df['journal'] = df.get('Publication Title', '')
    prepared_df['url'] = df.get('Url', '')
    
    # Clean up the data
    prepared_df['title'] = prepared_df['title'].fillna('').astype(str)
    prepared_df['abstract'] = prepared_df['abstract'].fillna('').astype(str)
    prepared_df['authors'] = prepared_df['authors'].fillna('').astype(str)
    prepared_df['year'] = prepared_df['year'].fillna('').astype(str)
    
    # Filter out papers without abstracts
    before_filter = len(prepared_df)
    prepared_df = prepared_df[prepared_df['abstract'].str.len() > 50]  # At least 50 characters
    after_filter = len(prepared_df)
    
    print(f"Filtered out {before_filter - after_filter} papers without substantial abstracts")
    print(f"Final dataset: {after_filter} papers")
    
    # Save prepared dataset
    prepared_df.to_csv(output_file, index=False)
    print(f"Saved prepared dataset to {output_file}")
    
    # Show sample
    print("\nSample of prepared data:")
    for i, row in prepared_df.head(3).iterrows():
        print(f"\nPaper {i+1}:")
        print(f"Title: {row['title'][:80]}...")
        print(f"Abstract: {row['abstract'][:100]}...")
        print(f"Year: {row['year']}")
        print(f"Authors: {row['authors'][:60]}...")

if __name__ == "__main__":
    input_file = "../1 Master SNA blockchain.csv"
    output_file = "data/papers_prepared.csv"
    
    prepare_csv_file(input_file, output_file)