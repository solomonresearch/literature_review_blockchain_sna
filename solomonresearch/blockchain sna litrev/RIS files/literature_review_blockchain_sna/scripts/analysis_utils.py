import pandas as pd
import numpy as np
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from jinja2 import Template


class ConsensusCalculator:
    """Calculate consensus and agreement metrics from multiple API responses"""
    
    def __init__(self, categories: Dict):
        self.categories = categories
        
    def calculate_consensus(self, api_results: Dict) -> Dict:
        """Calculate consensus from three API results"""
        categories = []
        relevance_scores = []
        confidence_scores = []
        research_types = []
        data_types = []
        
        # Extract valid responses
        for api_name in ['claude', 'gemini', 'deepseek']:
            if api_results[api_name].get('success'):
                cat = api_results[api_name].get('primary_category', '')
                if cat in self.categories:
                    categories.append(cat)
                
                score = api_results[api_name].get('relevance_score')
                if isinstance(score, (int, float)) and 1 <= score <= 10:
                    relevance_scores.append(score)
                
                conf = api_results[api_name].get('confidence')
                if isinstance(conf, (int, float)) and 0 <= conf <= 1:
                    confidence_scores.append(conf)
                    
                r_type = api_results[api_name].get('research_type', '')
                if r_type:
                    research_types.append(r_type)
                    
                d_type = api_results[api_name].get('data_type', '')
                if d_type:
                    data_types.append(d_type)
        
        # Handle no valid responses
        if not categories:
            return {
                'Final_Category': 'INSUFFICIENT_DATA',
                'Agreement_Level': 'No Data',
                'Review_Status': 'Insufficient Data',
                'Overall_Relevance': 'Unknown',
                'Agreement_Count': '0/3',
                'Consensus_Confidence': 0.0,
                'Final_Research_Type': '',
                'Final_Data_Type': ''
            }
        
        # Calculate category consensus
        category_counts = Counter(categories)
        most_common_category = category_counts.most_common(1)[0]
        
        # Calculate agreement level
        num_responses = len(categories)
        if num_responses == 3:
            if most_common_category[1] == 3:
                agreement = 'High'
                status = 'Clear'
            elif most_common_category[1] == 2:
                agreement = 'Medium'
                status = 'Clear'
            else:
                agreement = 'Low'
                status = 'Needs Review'
        elif num_responses == 2:
            agreement = 'Medium' if most_common_category[1] == 2 else 'Low'
            status = 'Clear' if most_common_category[1] == 2 else 'Needs Review'
        else:
            agreement = 'Single Source'
            status = 'Needs Review'
        
        # Calculate average relevance
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        if avg_relevance >= 8:
            overall_relevance = 'High'
        elif avg_relevance >= 6:
            overall_relevance = 'Medium'
        elif avg_relevance >= 3:
            overall_relevance = 'Low'
        else:
            overall_relevance = 'Very Low'
        
        # Calculate consensus confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Determine most common research and data types
        final_research_type = Counter(research_types).most_common(1)[0][0] if research_types else ''
        final_data_type = Counter(data_types).most_common(1)[0][0] if data_types else ''
        
        return {
            'Final_Category': most_common_category[0],
            'Agreement_Level': agreement,
            'Review_Status': status,
            'Overall_Relevance': overall_relevance,
            'Agreement_Count': f"{most_common_category[1]}/3",
            'Consensus_Confidence': round(avg_confidence, 3),
            'Final_Research_Type': final_research_type,
            'Final_Data_Type': final_data_type,
            'Average_Relevance_Score': round(avg_relevance, 2)
        }

    def calculate_inter_rater_agreement(self, results_df: pd.DataFrame) -> Dict:
        """Calculate inter-rater agreement statistics"""
        apis = ['Claude', 'Gemini', 'Deepseek']
        agreements = {}
        
        for api1 in apis:
            for api2 in apis:
                if api1 != api2:
                    col1 = f'{api1}_Category'
                    col2 = f'{api2}_Category'
                    
                    if col1 in results_df.columns and col2 in results_df.columns:
                        # Filter out empty responses
                        valid_pairs = results_df[(results_df[col1] != '') & (results_df[col2] != '')]
                        
                        if len(valid_pairs) > 0:
                            agreement_count = sum(valid_pairs[col1] == valid_pairs[col2])
                            agreement_rate = agreement_count / len(valid_pairs)
                            agreements[f'{api1}-{api2}'] = {
                                'agreement_rate': round(agreement_rate, 3),
                                'agreements': agreement_count,
                                'total_comparisons': len(valid_pairs)
                            }
        
        return agreements


class StatisticsGenerator:
    """Generate comprehensive statistics and reports"""
    
    def __init__(self, categories: Dict):
        self.categories = categories
        
    def generate_category_statistics(self, results_df: pd.DataFrame) -> Dict:
        """Generate detailed category statistics"""
        stats = {}
        
        # Overall distribution
        category_counts = results_df['Final_Category'].value_counts()
        total_papers = len(results_df)
        
        stats['total_papers'] = total_papers
        stats['categories_used'] = len(category_counts)
        stats['category_distribution'] = {}
        
        for category, count in category_counts.items():
            percentage = (count / total_papers) * 100
            priority = self.categories.get(category, {}).get('priority', 10)
            
            stats['category_distribution'][category] = {
                'count': int(count),
                'percentage': round(percentage, 2),
                'priority': priority
            }
        
        # High-priority categories
        high_priority_cats = [cat for cat, info in self.categories.items() if info.get('priority', 10) <= 3]
        high_priority_count = sum(category_counts.get(cat, 0) for cat in high_priority_cats)
        stats['high_priority_papers'] = {
            'count': int(high_priority_count),
            'percentage': round((high_priority_count / total_papers) * 100, 2),
            'categories': high_priority_cats
        }
        
        return stats

    def generate_quality_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Generate quality and reliability metrics"""
        metrics = {}
        
        # Agreement levels
        agreement_counts = results_df['Agreement_Level'].value_counts()
        total = len(results_df)
        
        metrics['agreement_distribution'] = {}
        for level, count in agreement_counts.items():
            metrics['agreement_distribution'][level] = {
                'count': int(count),
                'percentage': round((count / total) * 100, 2)
            }
        
        # Review status
        review_counts = results_df['Review_Status'].value_counts()
        metrics['review_status'] = {}
        for status, count in review_counts.items():
            metrics['review_status'][status] = {
                'count': int(count),
                'percentage': round((count / total) * 100, 2)
            }
        
        # Relevance distribution
        relevance_counts = results_df['Overall_Relevance'].value_counts()
        metrics['relevance_distribution'] = {}
        for level, count in relevance_counts.items():
            metrics['relevance_distribution'][level] = {
                'count': int(count),
                'percentage': round((count / total) * 100, 2)
            }
        
        # Confidence scores
        if 'Consensus_Confidence' in results_df.columns:
            conf_scores = results_df['Consensus_Confidence'][results_df['Consensus_Confidence'] > 0]
            if len(conf_scores) > 0:
                metrics['confidence_stats'] = {
                    'mean': round(conf_scores.mean(), 3),
                    'median': round(conf_scores.median(), 3),
                    'std': round(conf_scores.std(), 3),
                    'min': round(conf_scores.min(), 3),
                    'max': round(conf_scores.max(), 3)
                }
        
        return metrics

    def generate_research_type_analysis(self, results_df: pd.DataFrame) -> Dict:
        """Analyze research types and data types"""
        analysis = {}
        
        # Research types
        research_types = []
        for api in ['Claude', 'Gemini', 'Deepseek']:
            col = f'{api}_Type'
            if col in results_df.columns:
                valid_types = results_df[col][results_df[col] != '']
                research_types.extend(valid_types.tolist())
        
        if research_types:
            type_counts = Counter(research_types)
            analysis['research_types'] = {
                rtype: count for rtype, count in type_counts.most_common()
            }
        
        # Data types
        data_types = []
        for api in ['Claude', 'Gemini', 'Deepseek']:
            col = f'{api}_Data'
            if col in results_df.columns:
                valid_types = results_df[col][results_df[col] != '']
                data_types.extend(valid_types.tolist())
        
        if data_types:
            type_counts = Counter(data_types)
            analysis['data_types'] = {
                dtype: count for dtype, count in type_counts.most_common()
            }
        
        return analysis


class ReportGenerator:
    """Generate HTML and text reports"""
    
    def __init__(self, categories: Dict):
        self.categories = categories
        
    def generate_html_report(self, results_df: pd.DataFrame, output_path: str):
        """Generate comprehensive HTML report"""
        stats_gen = StatisticsGenerator(self.categories)
        consensus_calc = ConsensusCalculator(self.categories)
        
        # Generate all statistics
        category_stats = stats_gen.generate_category_statistics(results_df)
        quality_metrics = stats_gen.generate_quality_metrics(results_df)
        research_analysis = stats_gen.generate_research_type_analysis(results_df)
        agreement_stats = consensus_calc.calculate_inter_rater_agreement(results_df)
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Literature Review Classification Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; 
                 border-radius: 5px; min-width: 150px; text-align: center; }
        .category-item { margin: 10px 0; padding: 10px; background: #f1f3f4; border-radius: 5px; }
        .high-priority { background: #e8f5e8; border-left: 4px solid #28a745; }
        .medium-priority { background: #fff3cd; border-left: 4px solid #ffc107; }
        .low-priority { background: #f8d7da; border-left: 4px solid #dc3545; }
        .progress-bar { background: #e9ecef; border-radius: 10px; height: 20px; margin: 5px 0; }
        .progress-fill { height: 100%; border-radius: 10px; transition: width 0.5s ease; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 Literature Review Classification Report</h1>
        <h2>Blockchain & Social Network Analysis</h2>
        <p class="timestamp">Generated: {{ timestamp }}</p>
    </div>

    <div class="section">
        <h3>📊 Overview Statistics</h3>
        <div class="metric">
            <h4>{{ category_stats.total_papers }}</h4>
            <p>Total Papers Processed</p>
        </div>
        <div class="metric">
            <h4>{{ category_stats.categories_used }}</h4>
            <p>Categories Used</p>
        </div>
        <div class="metric">
            <h4>{{ category_stats.high_priority_papers.count }}</h4>
            <p>High Priority Papers</p>
        </div>
        <div class="metric">
            <h4>{{ quality_metrics.review_status.get('Needs Review', {}).get('count', 0) }}</h4>
            <p>Need Manual Review</p>
        </div>
    </div>

    <div class="section">
        <h3>🎯 Category Distribution</h3>
        {% for category, stats in category_stats.category_distribution.items() %}
        <div class="category-item {% if stats.priority <= 2 %}high-priority{% elif stats.priority <= 4 %}medium-priority{% else %}low-priority{% endif %}">
            <strong>{{ category }}</strong>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ stats.percentage }}%; 
                     background: {% if stats.priority <= 2 %}#28a745{% elif stats.priority <= 4 %}#ffc107{% else %}#dc3545{% endif %};"></div>
            </div>
            <span>{{ stats.count }} papers ({{ stats.percentage }}%)</span>
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h3>✅ Quality Metrics</h3>
        <h4>Agreement Levels</h4>
        <table>
            <tr><th>Agreement Level</th><th>Count</th><th>Percentage</th></tr>
            {% for level, data in quality_metrics.agreement_distribution.items() %}
            <tr><td>{{ level }}</td><td>{{ data.count }}</td><td>{{ data.percentage }}%</td></tr>
            {% endfor %}
        </table>

        <h4>Review Status</h4>
        <table>
            <tr><th>Status</th><th>Count</th><th>Percentage</th></tr>
            {% for status, data in quality_metrics.review_status.items() %}
            <tr><td>{{ status }}</td><td>{{ data.count }}</td><td>{{ data.percentage }}%</td></tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h3>🔬 Research Analysis</h3>
        {% if research_analysis.research_types %}
        <h4>Research Types</h4>
        <table>
            <tr><th>Type</th><th>Frequency</th></tr>
            {% for rtype, count in research_analysis.research_types.items() %}
            <tr><td>{{ rtype }}</td><td>{{ count }}</td></tr>
            {% endfor %}
        </table>
        {% endif %}

        {% if research_analysis.data_types %}
        <h4>Data Types</h4>
        <table>
            <tr><th>Type</th><th>Frequency</th></tr>
            {% for dtype, count in research_analysis.data_types.items() %}
            <tr><td>{{ dtype }}</td><td>{{ count }}</td></tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>

    <div class="section">
        <h3>🤝 Inter-Rater Agreement</h3>
        <table>
            <tr><th>API Comparison</th><th>Agreement Rate</th><th>Agreements</th><th>Total Comparisons</th></tr>
            {% for comparison, data in agreement_stats.items() %}
            <tr>
                <td>{{ comparison }}</td>
                <td>{{ (data.agreement_rate * 100) | round(1) }}%</td>
                <td>{{ data.agreements }}</td>
                <td>{{ data.total_comparisons }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h3>📈 Key Findings</h3>
        <ul>
            <li><strong>Primary Focus:</strong> {{ most_common_category }} ({{ most_common_count }} papers, {{ most_common_percent }}%)</li>
            <li><strong>High Agreement:</strong> {{ high_agreement_count }} papers achieved 3/3 API consensus</li>
            <li><strong>Review Required:</strong> {{ needs_review_count }} papers need manual review</li>
            <li><strong>Coverage:</strong> {{ category_stats.categories_used }} of 19 categories were used</li>
        </ul>
    </div>

</body>
</html>
        """
        
        # Calculate additional variables for template
        most_common_cat = max(category_stats['category_distribution'].items(), 
                             key=lambda x: x[1]['count'])
        most_common_category = most_common_cat[0]
        most_common_count = most_common_cat[1]['count']
        most_common_percent = most_common_cat[1]['percentage']
        
        high_agreement_count = quality_metrics['agreement_distribution'].get('High', {}).get('count', 0)
        needs_review_count = quality_metrics['review_status'].get('Needs Review', {}).get('count', 0)
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            category_stats=category_stats,
            quality_metrics=quality_metrics,
            research_analysis=research_analysis,
            agreement_stats=agreement_stats,
            most_common_category=most_common_category,
            most_common_count=most_common_count,
            most_common_percent=most_common_percent,
            high_agreement_count=high_agreement_count,
            needs_review_count=needs_review_count
        )
        
        # Save HTML report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML report generated: {output_path}")


class DataValidator:
    """Validate and clean classification results"""
    
    def __init__(self, categories: Dict):
        self.categories = categories
        self.valid_research_types = {
            'empirical', 'theoretical', 'methodological', 'review', 
            'applied', 'experimental', 'conceptual'
        }
        self.valid_data_types = {
            'real-world', 'synthetic', 'simulation', 'analytical', 'mixed', 'none'
        }
    
    def validate_results(self, results_df: pd.DataFrame) -> Dict:
        """Validate classification results and return validation report"""
        validation_report = {
            'total_papers': len(results_df),
            'validation_errors': [],
            'warnings': [],
            'data_quality_score': 0
        }
        
        errors = []
        warnings = []
        
        # Check for required columns
        required_cols = ['Final_Category', 'Agreement_Level', 'Review_Status']
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Validate categories
        invalid_categories = []
        for idx, row in results_df.iterrows():
            final_cat = row.get('Final_Category', '')
            if final_cat and final_cat not in self.categories and final_cat != 'INSUFFICIENT_DATA':
                invalid_categories.append((idx, final_cat))
        
        if invalid_categories:
            errors.append(f"Invalid categories found in {len(invalid_categories)} rows")
        
        # Check for empty critical fields
        empty_categories = results_df['Final_Category'].isnull().sum()
        if empty_categories > 0:
            warnings.append(f"{empty_categories} papers have empty Final_Category")
        
        # Validate research types
        for api in ['Claude', 'Gemini', 'Deepseek']:
            type_col = f'{api}_Type'
            if type_col in results_df.columns:
                invalid_types = results_df[
                    (results_df[type_col] != '') & 
                    (~results_df[type_col].isin(self.valid_research_types))
                ]
                if len(invalid_types) > 0:
                    warnings.append(f"Invalid research types in {type_col}: {len(invalid_types)} instances")
        
        # Calculate data quality score
        total_possible_responses = len(results_df) * 3  # 3 APIs per paper
        successful_responses = 0
        
        for api in ['Claude', 'Gemini', 'Deepseek']:
            cat_col = f'{api}_Category'
            if cat_col in results_df.columns:
                successful_responses += (results_df[cat_col] != '').sum()
        
        response_rate = successful_responses / total_possible_responses if total_possible_responses > 0 else 0
        
        # Score based on various factors
        high_agreement_rate = (results_df['Agreement_Level'] == 'High').sum() / len(results_df)
        clear_status_rate = (results_df['Review_Status'] == 'Clear').sum() / len(results_df)
        
        quality_score = (response_rate * 0.4 + high_agreement_rate * 0.3 + clear_status_rate * 0.3) * 100
        
        validation_report['validation_errors'] = errors
        validation_report['warnings'] = warnings
        validation_report['data_quality_score'] = round(quality_score, 2)
        validation_report['response_rate'] = round(response_rate * 100, 2)
        validation_report['high_agreement_rate'] = round(high_agreement_rate * 100, 2)
        validation_report['clear_status_rate'] = round(clear_status_rate * 100, 2)
        
        return validation_report

    def clean_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize results data"""
        cleaned_df = results_df.copy()
        
        # Standardize category names
        category_mapping = {cat.lower(): cat for cat in self.categories.keys()}
        
        for col in cleaned_df.columns:
            if 'Category' in col:
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: category_mapping.get(x.lower(), x) if isinstance(x, str) else x
                )
        
        # Clean research types
        research_type_mapping = {rt.lower(): rt for rt in self.valid_research_types}
        
        for api in ['Claude', 'Gemini', 'Deepseek']:
            type_col = f'{api}_Type'
            if type_col in cleaned_df.columns:
                cleaned_df[type_col] = cleaned_df[type_col].apply(
                    lambda x: research_type_mapping.get(x.lower(), x) if isinstance(x, str) else x
                )
        
        # Ensure numeric columns are properly typed
        numeric_cols = [col for col in cleaned_df.columns if 'Score' in col or 'Confidence' in col]
        for col in numeric_cols:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna('')
        
        return cleaned_df


# Example usage
if __name__ == "__main__":
    # This would be used in the main clustering script
    pass