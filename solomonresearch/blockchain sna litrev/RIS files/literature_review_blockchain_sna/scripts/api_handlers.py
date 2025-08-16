import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

import anthropic
import google.generativeai as genai
from openai import OpenAI


class APIHandler:
    """Base class for API handlers with common functionality"""
    
    def __init__(self, api_name: str):
        self.api_name = api_name
        self.success_count = 0
        self.failure_count = 0
        self.response_cache = {}
        
    def log_response(self, prompt_hash: str, response: Dict, paper_id: str):
        """Log API response for audit purposes"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'api': self.api_name,
            'timestamp': timestamp,
            'paper_id': paper_id,
            'prompt_hash': prompt_hash,
            'response': response,
            'success': response.get('success', False)
        }
        
        log_file = f"../logs/api_responses/{self.api_name}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def get_stats(self) -> Dict:
        """Return API usage statistics"""
        total_calls = self.success_count + self.failure_count
        success_rate = self.success_count / total_calls if total_calls > 0 else 0
        
        return {
            'api_name': self.api_name,
            'total_calls': total_calls,
            'successful_calls': self.success_count,
            'failed_calls': self.failure_count,
            'success_rate': success_rate
        }


class ClaudeHandler(APIHandler):
    """Handler for Claude API interactions"""
    
    def __init__(self, api_key: str):
        super().__init__("claude")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-opus-20240229"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def classify_paper(self, prompt: str, paper_id: str = "") -> Dict:
        """Classify a paper using Claude API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            response_text = response.content[0].text.strip()
            
            # Handle potential markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
                
            result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['primary_category', 'relevance_score', 'confidence']
            for field in required_fields:
                if field not in result:
                    result[field] = ""
            
            # Ensure gaps is a list
            if 'gaps' not in result or not isinstance(result['gaps'], list):
                result['gaps'] = ["", ""]
            elif len(result['gaps']) < 2:
                result['gaps'].extend([""] * (2 - len(result['gaps'])))
            
            final_result = {"success": True, **result}
            self.success_count += 1
            
        except json.JSONDecodeError as e:
            logging.error(f"Claude JSON parsing error for paper {paper_id}: {str(e)}")
            logging.error(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
            final_result = {"success": False, "error": "JSON parsing failed"}
            self.failure_count += 1
            
        except Exception as e:
            logging.error(f"Claude API error for paper {paper_id}: {str(e)}")
            final_result = {"success": False, "error": str(e)}
            self.failure_count += 1
        
        # Log response for audit
        self.log_response(hash(prompt), final_result, paper_id)
        return final_result


class GeminiHandler(APIHandler):
    """Handler for Gemini API interactions"""
    
    def __init__(self, api_key: str):
        super().__init__("gemini")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def classify_paper(self, prompt: str, paper_id: str = "") -> Dict:
        """Classify a paper using Gemini API"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500
                )
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Handle potential markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
                
            result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['primary_category', 'relevance_score', 'confidence']
            for field in required_fields:
                if field not in result:
                    result[field] = ""
            
            # Ensure gaps is a list
            if 'gaps' not in result or not isinstance(result['gaps'], list):
                result['gaps'] = ["", ""]
            elif len(result['gaps']) < 2:
                result['gaps'].extend([""] * (2 - len(result['gaps'])))
                
            final_result = {"success": True, **result}
            self.success_count += 1
            
        except json.JSONDecodeError as e:
            logging.error(f"Gemini JSON parsing error for paper {paper_id}: {str(e)}")
            logging.error(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
            final_result = {"success": False, "error": "JSON parsing failed"}
            self.failure_count += 1
            
        except Exception as e:
            logging.error(f"Gemini API error for paper {paper_id}: {str(e)}")
            final_result = {"success": False, "error": str(e)}
            self.failure_count += 1
        
        # Log response for audit
        self.log_response(hash(prompt), final_result, paper_id)
        return final_result


class DeepSeekHandler(APIHandler):
    """Handler for DeepSeek API interactions"""
    
    def __init__(self, api_key: str):
        super().__init__("deepseek")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def classify_paper(self, prompt: str, paper_id: str = "") -> Dict:
        """Classify a paper using DeepSeek API"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Handle potential markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
                
            result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['primary_category', 'relevance_score', 'confidence']
            for field in required_fields:
                if field not in result:
                    result[field] = ""
            
            # Ensure gaps is a list
            if 'gaps' not in result or not isinstance(result['gaps'], list):
                result['gaps'] = ["", ""]
            elif len(result['gaps']) < 2:
                result['gaps'].extend([""] * (2 - len(result['gaps'])))
                
            final_result = {"success": True, **result}
            self.success_count += 1
            
        except json.JSONDecodeError as e:
            logging.error(f"DeepSeek JSON parsing error for paper {paper_id}: {str(e)}")
            logging.error(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
            final_result = {"success": False, "error": "JSON parsing failed"}
            self.failure_count += 1
            
        except Exception as e:
            logging.error(f"DeepSeek API error for paper {paper_id}: {str(e)}")
            final_result = {"success": False, "error": str(e)}
            self.failure_count += 1
        
        # Log response for audit
        self.log_response(hash(prompt), final_result, paper_id)
        return final_result


class MultiAPIManager:
    """Manager for coordinating multiple API handlers"""
    
    def __init__(self, claude_key: str, gemini_key: str, deepseek_key: str):
        self.handlers = {
            'claude': ClaudeHandler(claude_key),
            'gemini': GeminiHandler(gemini_key),
            'deepseek': DeepSeekHandler(deepseek_key)
        }
        
    async def classify_paper_all_apis(self, prompt: str, paper_id: str = "") -> Dict:
        """Process a paper with all three APIs concurrently"""
        tasks = {}
        
        # Create tasks for all APIs
        for api_name, handler in self.handlers.items():
            tasks[api_name] = handler.classify_paper(prompt, paper_id)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True
        )
        
        # Map results back to API names
        api_results = {}
        for i, (api_name, result) in enumerate(zip(tasks.keys(), results)):
            if isinstance(result, Exception):
                logging.error(f"Exception in {api_name} for paper {paper_id}: {result}")
                api_results[api_name] = {"success": False, "error": str(result)}
            else:
                api_results[api_name] = result
                
        return api_results
    
    def get_all_stats(self) -> Dict:
        """Get statistics from all API handlers"""
        stats = {}
        total_calls = 0
        total_successes = 0
        
        for api_name, handler in self.handlers.items():
            api_stats = handler.get_stats()
            stats[api_name] = api_stats
            total_calls += api_stats['total_calls']
            total_successes += api_stats['successful_calls']
        
        stats['overall'] = {
            'total_calls': total_calls,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / total_calls if total_calls > 0 else 0
        }
        
        return stats
    
    async def test_all_apis(self, test_prompt: str) -> Dict:
        """Test all APIs with a sample prompt"""
        print("Testing API connections...")
        results = await self.classify_paper_all_apis(test_prompt, "test_paper")
        
        for api_name, result in results.items():
            status = "✅ Connected" if result.get('success') else "❌ Failed"
            print(f"  {api_name.capitalize()}: {status}")
            
        return results


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test_apis():
        manager = MultiAPIManager(
            claude_key=os.getenv('CLAUDE_API_KEY'),
            gemini_key=os.getenv('GEMINI_API_KEY'),
            deepseek_key=os.getenv('DEEPSEEK_API_KEY')
        )
        
        test_prompt = '''You are categorizing academic papers for a systematic literature review on Blockchain and Social Network Analysis.

Paper Details:
Title: Blockchain Networks and Social Analysis: A Test Paper
Abstract: This paper explores the intersection of blockchain technology and social network analysis, proposing new methods for analyzing decentralized networks.
Year: 2023
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
        
        results = await manager.test_all_apis(test_prompt)
        stats = manager.get_all_stats()
        
        print(f"\nAPI Statistics: {stats}")
        
    asyncio.run(test_apis())