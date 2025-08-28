# AI Data Analyzer 🤖📊

**Automated data analysis with natural language prompts**

Transform your data analysis workflow with AI! Simply describe what you want to analyze in plain English, and get comprehensive reports with insights, visualizations, and recommendations.

## 🌟 Features

- **Natural Language Interface**: Ask questions in plain English like "Show me correlations in my sales data"
- **Multi-Format Support**: Works with CSV, Excel, JSON, Parquet, and database connections
- **Automated Analysis**: Descriptive statistics, correlations, predictive modeling, clustering, anomaly detection
- **Interactive Visualizations**: Automatic chart generation with matplotlib, seaborn, and plotly
- **Comprehensive Reports**: HTML, Markdown, and JSON reports with insights and recommendations
- **Enterprise Ready**: Configurable, scalable, and designed for business use

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ai-data-analyzer.git
cd ai-data-analyzer

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Basic Usage

```bash
# Command line usage
python src/ai_data_analyzer.py --data "your_data.csv" --prompt "Describe my data and find correlations"

# Interactive mode
python src/ai_data_analyzer.py --interactive
```

### Python API

```python
from src.ai_data_analyzer import AIDataAnalyzer

# Initialize analyzer
analyzer = AIDataAnalyzer('config/config.yaml')

# Load your data
analyzer.load_data('your_data.csv')

# Analyze with natural language
results = analyzer.analyze_with_prompt("Find patterns and predict sales")

# Access results
print(results['report']['executive_summary'])
```

## 📝 Example Prompts

Here are some example prompts you can use:

### Descriptive Analysis
- "Describe my data"
- "Show me basic statistics for all variables"
- "What's the overview of this dataset?"

### Correlation Analysis
- "Show correlations between all variables"
- "Which variables are related to sales?"
- "Create a correlation heatmap"

### Predictive Modeling
- "Predict revenue using all features"
- "Build a model to classify customers"
- "What factors influence product ratings?"

### Clustering Analysis
- "Find customer segments in my data"
- "Group similar products together"
- "Identify 5 clusters in the dataset"

### Anomaly Detection
- "Find unusual patterns in transaction data"
- "Detect outliers in the sales figures"
- "Identify anomalies in customer behavior"

### Specific Analysis
- "Analyze only customers where age > 25"
- "Show trends over time for revenue"
- "Compare performance by region"

## 🛠️ Configuration

Customize the analyzer behavior by editing `config/config.yaml`:

```yaml
# Analysis Settings
analysis:
  correlation_threshold: 0.7  # Minimum correlation strength
  test_size: 0.2             # Train/test split ratio
  cv_folds: 5                # Cross-validation folds

# Visualization Settings
visualization:
  style: "seaborn-v0_8-darkgrid"
  palette: "viridis"
  figsize: [12, 8]

# Output Settings
output:
  directory: "./outputs"
  report_format: "html"
  include_visualizations: true
```

## 📊 Supported Data Sources

### File Formats
- **CSV**: Standard comma-separated values
- **Excel**: .xlsx and .xls files (multiple sheets supported)
- **JSON**: Structured JSON data
- **Parquet**: High-performance columnar format

### Database Connections
- **PostgreSQL**: `postgresql://user:password@host:port/database`
- **MySQL**: `mysql://user:password@host:port/database`
- **SQLite**: `sqlite:///path/to/database.db`
- **MongoDB**: `mongodb://host:port/database`

### URLs
- Direct URLs to CSV or Excel files
- API endpoints returning JSON data

## 🏗️ Architecture

The AI Data Analyzer consists of several modular components:

```
├── src/
│   ├── data_ingestion.py      # Multi-format data loading
│   ├── prompt_parser.py       # Natural language understanding
│   ├── analysis_engines.py    # Core analysis algorithms
│   ├── visualization_generator.py  # Automated chart creation
│   ├── report_generator.py    # Human-readable report generation
│   └── ai_data_analyzer.py    # Main application interface
├── config/
│   └── config.yaml           # Configuration settings
├── tests/                    # Unit tests
├── data/                     # Sample data files
└── outputs/                  # Generated reports and visualizations
```

## 🔧 Advanced Usage

### Custom Configuration

Create industry-specific configurations:

```python
# Custom config for financial analysis
config = {
    'analysis': {
        'correlation_threshold': 0.8,  # Higher threshold for finance
        'alpha': 0.01,                 # Stricter significance level
    },
    'visualization': {
        'palette': 'RdYlBu',          # Financial color scheme
    }
}

analyzer = AIDataAnalyzer()
analyzer.config.update(config)
```

### Batch Processing

```python
# Analyze multiple datasets
datasets = ['sales_q1.csv', 'sales_q2.csv', 'sales_q3.csv']
prompts = ['Describe quarterly sales trends', 'Find top performing products', 'Detect sales anomalies']

for dataset, prompt in zip(datasets, prompts):
    results = analyzer.analyze_with_prompt(prompt, dataset)
    print(f"Analysis complete for {dataset}")
```

### Integration with Business Systems

```python
# Example: Integration with company database
analyzer = AIDataAnalyzer()

# Connect to company database
db_connection = "postgresql://user:pass@company-db:5432/analytics"
analyzer.load_data(db_connection, table_name="sales_data")

# Scheduled analysis
results = analyzer.analyze_with_prompt("Weekly sales performance analysis")

# Send results to stakeholders (implement your notification system)
send_report_email(results['report']['formatted_reports']['html'])
```

## 🎯 Use Cases

### Business Intelligence
- **Sales Analysis**: "Identify factors driving sales performance"
- **Customer Segmentation**: "Find distinct customer groups in our data"
- **Market Trends**: "Analyze seasonal patterns in revenue"

### Data Science
- **Feature Engineering**: "Which variables are most predictive of customer churn?"
- **Model Building**: "Build a model to predict product demand"
- **Data Quality**: "Find outliers and data quality issues"

### Operations
- **Performance Monitoring**: "Detect anomalies in system metrics"
- **Resource Planning**: "Predict future resource requirements"
- **Quality Control**: "Identify defects in manufacturing data"

## 🔒 Security & Privacy

- **Local Processing**: All analysis runs locally by default
- **Configurable API Usage**: Optional OpenAI integration for advanced NLP (configure your own API key)
- **Data Privacy**: No data is sent to external services without explicit configuration
- **Audit Trail**: Complete logging of all analysis operations

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ tests/

# Type checking
mypy src/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

**Q: "No module named 'src.data_ingestion'"**  
A: Make sure you're running from the project root directory and have installed the package with `pip install -e .`

**Q: "OpenAI API key error"**  
A: Set your OpenAI API key in the environment: `export OPENAI_API_KEY=your_key_here` (optional feature)

**Q: "Visualization not showing"**  
A: Check that you have the required visualization libraries installed and configured

### Getting Help

- 📖 [Documentation](docs/)
- 🐛 [Report Issues](https://github.com/your-org/ai-data-analyzer/issues)
- 💬 [Discussions](https://github.com/your-org/ai-data-analyzer/discussions)
- 📧 Email: support@aidatasolutions.com

## 🙏 Acknowledgments

- Built with powerful open-source libraries: pandas, scikit-learn, plotly, matplotlib
- Inspired by the need for democratized data analysis
- Thanks to the Python data science community

---

**Made with ❤️ for data analysts and business users everywhere**

## 🔄 Changelog

### v1.0.0 (2024-08-22)
- Initial release
- Core analysis engines implemented
- Natural language prompt parsing
- Automated visualization generation
- Multi-format report generation
- Interactive and command-line interfaces

---

*Start analyzing your data with AI today! 🚀*
