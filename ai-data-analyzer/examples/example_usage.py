"""
Example Usage of AI Data Analyzer
Demonstrates various ways to use the system
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ai_data_analyzer import AIDataAnalyzer
import pandas as pd
import numpy as np

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample business data
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 12, n_samples).astype(int),
        'income': np.random.lognormal(10, 0.5, n_samples).astype(int),
        'education_years': np.random.normal(14, 3, n_samples).astype(int),
        'products_purchased': np.random.poisson(3, n_samples),
        'satisfaction_score': np.random.uniform(1, 10, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'customer_type': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples, p=[0.2, 0.5, 0.3]),
        'join_date': pd.date_range('2020-01-01', periods=n_samples, freq='D')[:n_samples],
        'total_spent': np.random.exponential(500, n_samples)
    }
    
    # Add some correlations
    # Make satisfaction score somewhat dependent on income and education
    data['satisfaction_score'] = (data['satisfaction_score'] + 
                                 (data['income'] / 50000) * 2 + 
                                 (data['education_years'] / 20) * 3) / 2
    
    # Make total spent dependent on income and products purchased
    data['total_spent'] = (data['total_spent'] + 
                          data['income'] * 0.01 + 
                          data['products_purchased'] * 100)
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    for idx in missing_indices:
        data['satisfaction_score'][idx] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    sample_data_path = Path(__file__).parent.parent / 'data' / 'sample_customer_data.csv'
    sample_data_path.parent.mkdir(exist_ok=True)
    df.to_csv(sample_data_path, index=False)
    
    print(f"✅ Sample dataset created: {sample_data_path}")
    return str(sample_data_path)

def run_examples():
    """Run various analysis examples"""
    
    print("🤖 AI Data Analyzer - Example Usage")
    print("=" * 50)
    
    # Create sample data
    data_path = create_sample_dataset()
    
    # Initialize analyzer
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    analyzer = AIDataAnalyzer(str(config_path))
    
    # Load the sample data
    print("\\n📊 Loading sample data...")
    load_result = analyzer.load_data(data_path)
    if not load_result['success']:
        print(f"❌ Error loading data: {load_result['error']}")
        return
    
    print(f"✅ {load_result['message']}")
    
    # Example analyses
    examples = [
        {
            'name': 'Descriptive Analysis',
            'prompt': 'Describe my customer data and show basic statistics',
            'description': 'Basic overview of the dataset'
        },
        {
            'name': 'Correlation Analysis',
            'prompt': 'Show correlations between customer income, satisfaction, and spending',
            'description': 'Find relationships between key variables'
        },
        {
            'name': 'Customer Segmentation',
            'prompt': 'Find customer segments using income, age, and total spent',
            'description': 'Cluster customers into groups'
        },
        {
            'name': 'Predictive Modeling',
            'prompt': 'Predict customer satisfaction using income, age, and education',
            'description': 'Build a model to predict satisfaction scores'
        },
        {
            'name': 'Anomaly Detection',
            'prompt': 'Detect unusual spending patterns in customer data',
            'description': 'Find outliers in customer behavior'
        }
    ]
    
    # Run each example
    for i, example in enumerate(examples, 1):
        print(f"\\n{i}. {example['name']}")
        print(f"   Description: {example['description']}")
        print(f"   Prompt: '{example['prompt']}'")
        print("   🔄 Analyzing...")
        
        try:
            results = analyzer.analyze_with_prompt(example['prompt'])
            
            if results['success']:
                print("   ✅ Analysis completed!")
                
                # Show key insights
                insights = results['analysis_result'].insights
                if insights:
                    print("   🔍 Key Insights:")
                    for insight in insights[:2]:  # Show top 2
                        print(f"      • {insight}")
                
                # Show report location
                report_files = results['report'].get('formatted_reports', {})
                if report_files.get('html'):
                    print(f"   📄 Report: {report_files['html']}")
                
            else:
                print(f"   ❌ Analysis failed: {results['error']}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print("   " + "-" * 40)
    
    print(f"\\n🎉 All examples completed!")
    print(f"💾 Check the outputs directory for generated reports: {analyzer.config['output']['directory']}")
    
    # Show suggestions for further analysis
    print("\\n💡 Suggested follow-up analyses:")
    suggestions = analyzer.suggest_analyses()
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"   {i}. {suggestion.raw_prompt}")

def run_interactive_demo():
    """Run a simulated interactive session"""
    
    print("\\n🎮 Interactive Demo Simulation")
    print("=" * 40)
    
    # Simulate interactive commands
    demo_commands = [
        "load data/sample_customer_data.csv",
        "info",
        "suggest",
        "Describe my data",
        "Show strong correlations",
        "Find 4 customer clusters",
        "history"
    ]
    
    print("Simulated interactive session:")
    for cmd in demo_commands:
        print(f"Analyzer> {cmd}")
        # In real usage, these would be processed by InteractiveAnalyzer
    
    print("\\nTo run real interactive mode:")
    print("python src/ai_data_analyzer.py --interactive")

if __name__ == "__main__":
    try:
        # Run the examples
        run_examples()
        
        # Show interactive demo
        run_interactive_demo()
        
        print("\\n✨ Example session completed successfully!")
        print("\\nNext steps:")
        print("1. Try the interactive mode: python src/ai_data_analyzer.py --interactive")
        print("2. Use your own data: python src/ai_data_analyzer.py --data your_file.csv --prompt 'your question'")
        print("3. Customize the configuration in config/config.yaml")
        
    except KeyboardInterrupt:
        print("\\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error in example: {str(e)}")
        print("Make sure you have installed all dependencies: pip install -r requirements.txt")
