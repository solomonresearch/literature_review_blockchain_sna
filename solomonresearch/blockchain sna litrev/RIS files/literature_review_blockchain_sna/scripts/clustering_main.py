import pandas as pd
import numpy as np
import os
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
from colorama import init, Fore, Style
from dotenv import load_dotenv
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize colorama for colored terminal output
init()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/processing_log.txt'),
        logging.StreamHandler()
    ]
)

# Complete category taxonomy
CATEGORIES = {
    "Blockchain-SNA Integration": {
        "description": "Papers combining blockchain technology with social network analysis methods",
        "subfields": ["decentralized social networks", "trust networks on blockchain", "consensus in social systems", "tokenomics and network effects"],
        "priority": 1
    },
    
    "Financial Networks & DeFi": {
        "description": "Decentralized finance, financial networks, and economic systems on blockchain",
        "subfields": ["DeFi protocols", "cryptocurrency networks", "financial contagion", "payment networks"],
        "priority": 2
    },
    
    "Economic Models & Game Theory": {
        "description": "Economic modeling, game theory, and incentive mechanisms in distributed systems",
        "subfields": ["mechanism design", "auction theory", "behavioral economics", "market microstructure"],
        "priority": 2
    },
    
    "Blockchain Infrastructure": {
        "description": "Core blockchain technology, protocols, consensus mechanisms, and scalability",
        "subfields": ["consensus algorithms", "layer-2 solutions", "interoperability", "smart contracts"],
        "priority": 3
    },
    
    "Blockchain Applications": {
        "description": "Applied blockchain solutions across various domains",
        "subfields": ["supply chain", "healthcare", "identity management", "voting systems"],
        "priority": 3
    },
    
    "Network Science & Methods": {
        "description": "Theoretical and methodological contributions to network analysis",
        "subfields": ["graph algorithms", "community detection", "link prediction", "network dynamics"],
        "priority": 3
    },
    
    "Social Media & Online Networks": {
        "description": "Analysis of social media platforms and online social networks",
        "subfields": ["influence propagation", "viral marketing", "echo chambers", "user behavior"],
        "priority": 3
    },
    
    "Security & Privacy": {
        "description": "Cybersecurity, privacy-preserving technologies, and cryptographic methods",
        "subfields": ["privacy-preserving SNA", "homomorphic encryption", "attack detection"],
        "priority": 4
    },
    
    "Governance & Regulation": {
        "description": "Governance models, regulatory frameworks, and compliance",
        "subfields": ["DAO governance", "regulatory compliance", "smart contract law", "digital identity"],
        "priority": 4
    },
    
    "AI & Machine Learning": {
        "description": "AI and ML applications in blockchain or network analysis",
        "subfields": ["graph neural networks", "federated learning", "anomaly detection", "predictive modeling"],
        "priority": 4
    },
    
    "IoT & Edge Computing": {
        "description": "Internet of Things, edge computing, and distributed sensor networks",
        "subfields": ["IoT security", "edge consensus", "sensor networks", "smart cities"],
        "priority": 5
    },
    
    "Healthcare & Biomedical": {
        "description": "Healthcare applications, medical records, and biomedical network analysis",
        "subfields": ["electronic health records", "drug discovery networks", "epidemic modeling"],
        "priority": 5
    },
    
    "Supply Chain & Logistics": {
        "description": "Supply chain management, logistics networks, and traceability",
        "subfields": ["provenance tracking", "logistics optimization", "supplier networks"],
        "priority": 5
    },
    
    "Organizational Networks": {
        "description": "Organizational behavior, knowledge management, and enterprise networks",
        "subfields": ["knowledge networks", "collaboration patterns", "innovation networks"],
        "priority": 5
    },
    
    "Business Models & Strategy": {
        "description": "Business models, strategic management, and platform economics",
        "subfields": ["platform economy", "network effects", "digital transformation"],
        "priority": 5
    },
    
    "Literature Reviews": {
        "description": "Systematic reviews, meta-analyses, surveys, and bibliometric studies",
        "subfields": ["systematic review", "meta-analysis", "scoping review", "bibliometric analysis"],
        "priority": 6
    },
    
    "Emerging Technologies": {
        "description": "Emerging tech like quantum, metaverse, and Web3",
        "subfields": ["quantum blockchain", "metaverse", "Web3", "digital twins"],
        "priority": 7
    },
    
    "Related Work": {
        "description": "Tangentially related papers with limited direct relevance",
        "subfields": ["distributed systems", "P2P", "cloud computing"],
        "priority": 8
    },
    
    "Out of Scope": {
        "description": "Papers without meaningful connection to research objectives",
        "subfields": ["unrelated", "false positive"],
        "priority": 9
    }
}

class LiteratureReviewClustering:
    def __init__(self, input_csv: str, output_csv: str):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = None
        self.batch_size = 10
        self.categories = CATEGORIES
        
        # Initialize API client (Claude only)
        self.claude_client = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful_categorizations': 0,
            'failed_api_calls': 0,
            'papers_needing_review': 0,
            'category_distribution': {}
        }

    def create_api_prompt(self, paper: Dict) -> str:
        """Create standardized prompt for all APIs"""
        return f'''You are categorizing academic papers for a systematic literature review on Blockchain and Social Network Analysis.

Paper Details:
Title: {paper.get('title', '')}
Abstract: {paper.get('abstract', '')}
Year: {paper.get('year', '')}
Source: {paper.get('source_db', '')}

Analyze and categorize into ONE of these categories:
1. "Blockchain-SNA Integration" - Combines blockchain and social network analysis
2. "Financial Networks & DeFi" - Decentralized finance and financial networks
3. "Economic Models & Game Theory" - Economic modeling and incentives
4. "Blockchain Infrastructure" - Core blockchain technology
5. "Blockchain Applications" - Applied blockchain solutions
6. "Network Science & Methods" - Network analysis methods
7. "Social Media & Online Networks" - Social media analysis
8. "Security & Privacy" - Security and privacy focus
9. "Governance & Regulation" - Governance and regulatory frameworks
10. "AI & Machine Learning" - AI/ML in blockchain or networks
11. "IoT & Edge Computing" - IoT and distributed sensors
12. "Healthcare & Biomedical" - Healthcare applications
13. "Supply Chain & Logistics" - Supply chain networks
14. "Organizational Networks" - Organizational networks
15. "Business Models & Strategy" - Business strategy
16. "Literature Reviews" - Reviews and meta-analyses
17. "Emerging Technologies" - Quantum, metaverse, Web3
18. "Related Work" - Tangentially related
19. "Out of Scope" - No connection

Return ONLY valid JSON (no markdown, no extra text):
{{
    "primary_category": "[category name exactly as listed above]",
    "secondary_category": "[second category or empty string]",
    "research_type": "[empirical|theoretical|methodological|review|applied|experimental|conceptual|]",
    "data_type": "[real-world|synthetic|simulation|analytical|mixed|none|]",
    "gaps": ["[first gap if explicitly mentioned or empty]", "[second gap or empty]"],
    "relevance_score": [1-10 integer],
    "confidence": [0.0-1.0 float]
}}

RULES:
- Return empty string "" for missing information
- Gaps must be explicitly stated in abstract
- Choose category based on PRIMARY contribution
- No hallucination - only what's in the abstract'''

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def call_claude_api(self, prompt: str) -> Dict:
        """Call Claude API with retry logic"""
        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            # Clean JSON response (remove markdown code blocks if present)
            raw_text = response.content[0].text
            json_text = raw_text
            if json_text.startswith('```json'):
                json_text = json_text[7:]  # Remove ```json
            if json_text.endswith('```'):
                json_text = json_text[:-3]  # Remove ```
            json_text = json_text.strip()
            
            result = json.loads(json_text)
            return {"success": True, **result}
        except Exception as e:
            logging.error(f"Claude API error: {str(e)}")
            return {"success": False}


    async def process_paper_with_claude_api(self, paper: Dict) -> Dict:
        """Process single paper with Claude API only"""
        prompt = self.create_api_prompt(paper)
        
        # Call Claude API
        claude_result = await self.call_claude_api(prompt)
        
        return {
            'claude': claude_result
        }

    def calculate_result_from_claude(self, api_results: Dict) -> Dict:
        """Calculate result from Claude API response"""
        claude_result = api_results.get('claude', {})
        
        # Check if Claude API was successful
        if not claude_result.get('success'):
            return {
                'Final_Category': 'INSUFFICIENT_DATA',
                'Agreement_Level': 'API_FAILED',
                'Review_Status': 'API_ERROR',
                'Overall_Relevance': 'Unknown',
                'Agreement_Count': '0/1'
            }
        
        # Extract Claude's classification
        category = claude_result.get('primary_category', '')
        relevance_score = claude_result.get('relevance_score', 0)
        confidence = claude_result.get('confidence', 0)
        
        # Validate category
        if category not in self.categories:
            category = 'INVALID_CATEGORY'
        
        # Calculate relevance level
        if relevance_score >= 8:
            overall_relevance = 'High'
        elif relevance_score >= 6:
            overall_relevance = 'Medium'
        elif relevance_score >= 3:
            overall_relevance = 'Low'
        else:
            overall_relevance = 'Very Low'
        
        # Determine review status based on confidence
        if confidence >= 0.8 and relevance_score > 0:
            review_status = 'Clear'
            agreement_level = 'High_Confidence'
        elif confidence >= 0.6:
            review_status = 'Clear'
            agreement_level = 'Medium_Confidence'
        else:
            review_status = 'Needs Review'
            agreement_level = 'Low_Confidence'
        
        return {
            'Final_Category': category,
            'Agreement_Level': agreement_level,
            'Review_Status': review_status,
            'Overall_Relevance': overall_relevance,
            'Agreement_Count': '1/1',
            'Claude_Confidence': confidence
        }

    def display_batch_progress(self, batch_num: int, batch_results: List[Dict], total_batches: int):
        """Display detailed progress after each batch"""
        print(f"\n{'='*80}")
        print(f"{Fore.CYAN}📚 BATCH {batch_num}/{total_batches} COMPLETE{Style.RESET_ALL}")
        print(f"{'='*80}")
        
        # Category distribution
        categories = {}
        for r in batch_results:
            cat = r.get('Final_Category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n{Fore.YELLOW}📊 Category Distribution (This Batch):{Style.RESET_ALL}")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            bar = '█' * count + '░' * (10 - count)
            color = Fore.GREEN if cat == 'Blockchain-SNA Integration' else Fore.WHITE
            print(f"  {color}{cat:<35} [{bar}] {count} papers{Style.RESET_ALL}")
        
        # Quality statistics
        high_confidence = sum(1 for r in batch_results if r.get('Agreement_Level') == 'High_Confidence')
        need_review = sum(1 for r in batch_results if r.get('Review_Status') == 'Needs Review')
        
        print(f"\n{Fore.YELLOW}✅ Quality Metrics:{Style.RESET_ALL}")
        print(f"  • High Confidence: {high_confidence}/{len(batch_results)} papers")
        print(f"  • Need Manual Review: {need_review} papers")
        
        # Show papers needing review
        if need_review > 0:
            print(f"\n{Fore.RED}⚠ Papers Needing Review:{Style.RESET_ALL}")
            for r in batch_results:
                if r.get('Review_Status') == 'Needs Review':
                    title = r.get('title', 'Unknown')[:60]
                    print(f"  • {title}...")
        
        print(f"\n{Fore.GREEN}💾 CSV saved: {self.output_csv}{Style.RESET_ALL}")
        print(f"{'='*80}")

    async def process_batch(self, batch_df: pd.DataFrame, batch_num: int) -> List[Dict]:
        """Process a batch of 10 papers"""
        batch_results = []
        
        print(f"\n{Fore.BLUE}Starting Batch {batch_num} - {len(batch_df)} papers{Style.RESET_ALL}")
        
        for idx, row in batch_df.iterrows():
            paper = row.to_dict()
            print(f"  [{idx % 10 + 1}/{len(batch_df)}] Processing: {paper.get('title', 'Unknown')[:50]}...", end='')
            
            # Get results from Claude API
            api_results = await self.process_paper_with_claude_api(paper)
            
            # Prepare result row
            result = paper.copy()
            
            # Add results from Claude API
            claude_data = api_results['claude']
            if claude_data.get('success'):
                result['Claude_Category'] = claude_data.get('primary_category', '')
                result['Claude_Secondary'] = claude_data.get('secondary_category', '')
                result['Claude_Type'] = claude_data.get('research_type', '')
                result['Claude_Data'] = claude_data.get('data_type', '')
                gaps = claude_data.get('gaps', ['', ''])
                result['Claude_Gap1'] = gaps[0] if len(gaps) > 0 else ''
                result['Claude_Gap2'] = gaps[1] if len(gaps) > 1 else ''
                result['Claude_Score'] = claude_data.get('relevance_score', '')
                result['Claude_Confidence'] = claude_data.get('confidence', '')
            else:
                # Empty strings for failed API call
                for suffix in ['Category', 'Secondary', 'Type', 'Data', 'Gap1', 'Gap2', 'Score', 'Confidence']:
                    result[f'Claude_{suffix}'] = ''
            
            # Calculate result from Claude
            classification_result = self.calculate_result_from_claude(api_results)
            result.update(classification_result)
            
            # Add metadata
            result['Batch_Number'] = batch_num
            result['Processing_Timestamp'] = datetime.now().isoformat()
            
            batch_results.append(result)
            
            # Update stats
            self.stats['total_processed'] += 1
            if classification_result['Review_Status'] == 'Clear':
                self.stats['successful_categorizations'] += 1
            elif classification_result['Review_Status'] == 'Needs Review':
                self.stats['papers_needing_review'] += 1
            
            print(f" ✓ {classification_result['Final_Category'][:20]}")
        
        return batch_results

    async def run_clustering(self):
        """Main clustering pipeline"""
        print(f"""
{Fore.CYAN}╔════════════════════════════════════════════════════════════╗
║     SYSTEMATIC LITERATURE REVIEW CLASSIFICATION SYSTEM      ║
╠════════════════════════════════════════════════════════════╣
║  Research Focus: Blockchain & Social Network Analysis       ║
║  Total Papers: 1094                                        ║
║  Batch Size: 10 papers                                     ║
║  API: Claude Sonnet 4 (Single API Mode)                    ║
╚════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
        """)
        
        # Load data
        print(f"\n{Fore.YELLOW}Loading input CSV...{Style.RESET_ALL}")
        self.df = pd.read_csv(self.input_csv)
        total_papers = len(self.df)
        total_batches = (total_papers + self.batch_size - 1) // self.batch_size
        
        print(f"✓ Loaded {total_papers} papers")
        print(f"✓ Will process in {total_batches} batches")
        
        # Initialize output dataframe
        all_results = []
        
        # Process batches
        for batch_num in range(1, total_batches + 1):
            start_idx = (batch_num - 1) * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_papers)
            batch_df = self.df.iloc[start_idx:end_idx]
            
            # Process batch
            batch_results = await self.process_batch(batch_df, batch_num)
            all_results.extend(batch_results)
            
            # Save after each batch
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(self.output_csv, index=False)
            
            # Display progress
            self.display_batch_progress(batch_num, batch_results, total_batches)
            
            # Backup every 5 batches
            if batch_num % 5 == 0:
                backup_file = f"data/backup/backup_batch_{batch_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                results_df.to_csv(backup_file, index=False)
                print(f"  📁 Backup saved: {backup_file}")
            
            # Overall progress
            progress_pct = (batch_num / total_batches) * 100
            print(f"\n{Fore.GREEN}OVERALL PROGRESS: {batch_num}/{total_batches} batches ({progress_pct:.1f}%){Style.RESET_ALL}")
            
            # Rate limiting between batches
            if batch_num < total_batches:
                print(f"⏳ Waiting 2 seconds before next batch...")
                await asyncio.sleep(2)
        
        # Final statistics
        self.display_final_statistics(pd.DataFrame(all_results))
        
        print(f"\n{Fore.GREEN}✅ CLUSTERING COMPLETE!{Style.RESET_ALL}")
        print(f"Output saved to: {self.output_csv}")

    def display_final_statistics(self, results_df: pd.DataFrame):
        """Display comprehensive final statistics"""
        print(f"""
{Fore.CYAN}╔════════════════════════════════════════════════════════════╗
║                  FINAL STATISTICS                           ║
╚════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
        """)
        
        total = len(results_df)
        
        # Category distribution
        category_counts = results_df['Final_Category'].value_counts()
        
        print(f"\n{Fore.YELLOW}📊 FINAL CATEGORY DISTRIBUTION:{Style.RESET_ALL}")
        for cat, count in category_counts.items():
            pct = (count / total) * 100
            bar_length = int(pct / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {cat:<35} [{bar}] {count:4} ({pct:.1f}%)")
        
        # Key metrics
        blockchain_sna = len(results_df[results_df['Final_Category'] == 'Blockchain-SNA Integration'])
        high_relevance = len(results_df[results_df['Overall_Relevance'] == 'High'])
        high_confidence = len(results_df[results_df['Agreement_Level'] == 'High_Confidence'])
        need_review = len(results_df[results_df['Review_Status'] == 'Needs Review'])
        
        print(f"\n{Fore.YELLOW}🎯 KEY METRICS:{Style.RESET_ALL}")
        print(f"  • Blockchain-SNA Integration Papers: {blockchain_sna} ({blockchain_sna/total*100:.1f}%)")
        print(f"  • High Relevance Papers: {high_relevance} ({high_relevance/total*100:.1f}%)")
        print(f"  • High Confidence Classifications: {high_confidence} ({high_confidence/total*100:.1f}%)")
        print(f"  • Papers Needing Manual Review: {need_review} ({need_review/total*100:.1f}%)")
        
        # Save statistics to JSON
        stats = {
            'total_papers': total,
            'blockchain_sna_integration': int(blockchain_sna),
            'categories_used': len(category_counts),
            'high_relevance_papers': int(high_relevance),
            'papers_needing_review': int(need_review),
            'category_distribution': category_counts.to_dict(),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        with open('../reports/statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{Fore.GREEN}📄 Statistics saved to reports/statistics.json{Style.RESET_ALL}")

# MAIN EXECUTION
async def main():
    # Create necessary directories
    os.makedirs('../data/backup', exist_ok=True)
    os.makedirs('../logs/api_responses', exist_ok=True)
    os.makedirs('../reports', exist_ok=True)
    
    # Initialize and run clustering
    clusterer = LiteratureReviewClustering(
        input_csv='../data/papers_prepared.csv',
        output_csv='../data/papers_classified.csv'
    )
    
    await clusterer.run_clustering()

if __name__ == "__main__":
    asyncio.run(main())