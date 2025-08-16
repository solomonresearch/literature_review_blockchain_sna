# Literature Review Analysis System - Instructions

## 🎯 Overview

This advanced literature analysis system performs comprehensive analysis of academic papers using multiple methodologies including statistical analysis, NLP, topic modeling, network analysis, and AI-assisted insights.

## 📋 System Requirements

- Python 3.8+
- Virtual environment (recommended)
- Claude API key
- Input CSV file with literature data

## 🚀 Quick Start Instructions

### 1. Setup Environment

```bash
# Navigate to project directory
cd "/Users/v/solomonresearch/blockchain sna litrev/RIS files/literature_review_blockchain_sna"

# Activate virtual environment
source venv/bin/activate

# Install additional analysis requirements
pip install -r requirements_analysis.txt
```

### 2. Verify Configuration

Ensure your `.env` file contains:
```
CLAUDE_API_KEY=your_claude_api_key_here
```

### 3. Verify Input File

Ensure the CSV file exists at:
```
/Users/v/solomonresearch/blockchain sna litrev/RIS files/SNA Blockchain - Filtered all.csv
```

Required CSV columns:
- title
- abstract  
- authors
- year
- doi
- journal
- url
- Claude_Category
- Claude_Secondary
- Claude_Type
- Claude_Data
- Claude_Gap1
- Claude_Gap2

### 4. Run Analysis

**Option A: Automated Setup and Run**
```bash
python run_analysis.py
```

**Option B: Direct Execution**
```bash
python literature_analysis.py
```

## 📊 Analysis Methodologies

### Statistical Analysis
- Descriptive statistics and distributions
- Temporal evolution patterns
- Category and methodology distributions
- Cross-tabulation analysis

### Text Mining & NLP
- **TF-IDF Vectorization**: Keyword extraction and importance scoring
- **Sentiment Analysis**: Research gap sentiment using TextBlob
- **Named Entity Recognition**: Technical term identification
- **N-gram Analysis**: Multi-word phrase extraction

### Thematic Analysis
- **Topic Modeling**: Latent Dirichlet Allocation (LDA) using Gensim
- **Clustering**: K-means and hierarchical clustering of research gaps
- **Semantic Similarity**: Word2Vec-based similarity analysis
- **Co-occurrence Networks**: Term and concept relationship mapping

### Network Analysis
- **Category Relationship Networks**: Using NetworkX
- **Cross-domain Integration Patterns**: Interdisciplinary collaboration analysis
- **Centrality Measures**: Identifying key research areas
- **Community Detection**: Research cluster identification

### AI-Assisted Analysis
- **Claude API Integration**: Advanced thematic pattern recognition
- **Batch Processing**: Incremental analysis with AI insights
- **Research Gap Classification**: Automated gap categorization
- **Trend Prediction**: Emerging research direction identification

## 📄 Output Files

### Incremental Report
- **File**: `literature_analysis_report.docx`
- **Updates**: After each batch (20 papers)
- **Content**: Real-time analysis progress

### Final Comprehensive Report
- **File**: `literature_analysis_report_final.docx`
- **Content**: Complete analysis with:
  - Executive summary
  - Statistical distributions with tables
  - Temporal evolution analysis
  - Research gap thematic clustering
  - Cross-domain integration patterns
  - AI-assisted insights and recommendations
  - Referenced papers with DOIs

## 🔄 Batch Processing Details

- **Batch Size**: 20 papers per batch
- **Progress Updates**: Real-time console output
- **Document Updates**: Incremental DOCX updates
- **API Rate Limiting**: 3-second delays between batches
- **Error Handling**: Graceful failure recovery

## 📈 Expected Processing Time

- **Small Dataset** (100 papers): ~15-20 minutes
- **Medium Dataset** (500 papers): ~1-2 hours  
- **Large Dataset** (1000+ papers): ~2-4 hours

*Note: Time depends on Claude API response times and text complexity*

## 🛠 Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Install missing packages
pip install -r requirements_analysis.txt
```

**2. NLTK Data Missing**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

**3. Claude API Errors**
- Verify API key in `.env` file
- Check API quota and rate limits
- Ensure stable internet connection

**4. Input File Issues**
- Verify CSV file path and format
- Check required columns exist
- Ensure proper encoding (UTF-8)

### Debug Mode

For detailed debugging, modify the script:
```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Libraries and Dependencies

### Core Analysis
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization
- **plotly**: Interactive visualizations

### Text Processing
- **nltk**: Natural language processing toolkit
- **spacy**: Advanced NLP (optional)
- **textblob**: Sentiment analysis
- **gensim**: Topic modeling and word embeddings
- **scikit-learn**: Machine learning and clustering

### Network Analysis
- **networkx**: Graph theory and network analysis

### Document Generation
- **python-docx**: Professional document creation

### API Integration
- **anthropic**: Claude API client
- **python-dotenv**: Environment variable management

## 🎯 Academic Publication Ready

The generated report includes:

1. **Methodological Rigor**: Detailed methodology documentation
2. **Statistical Validation**: Proper statistical tests and confidence intervals
3. **Reproducibility**: Complete parameter documentation
4. **Academic Formatting**: Professional tables and references
5. **DOI References**: Citable paper references throughout
6. **Transparent Analysis**: Clear analytical decision documentation

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all requirements are installed
3. Check console output for specific error messages
4. Ensure input data format matches requirements

## ⚡ Performance Tips

1. **Use SSD storage** for faster file I/O
2. **Ensure stable internet** for Claude API calls
3. **Close unnecessary applications** to free memory
4. **Monitor batch progress** via console output
5. **Check intermediate files** for early results

---

**Ready to analyze your literature dataset!** 🚀