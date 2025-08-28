"""
AI Data Analyzer - Quick Demo
Showcases the key capabilities of the system
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def create_demo_data():
    """Create demonstration data"""
    print("📊 Creating sample business dataset...")
    
    np.random.seed(42)
    n = 500
    
    # Simulate realistic business data
    data = {
        'customer_id': range(1, n + 1),
        'age': np.random.normal(40, 15, n).astype(int),
        'annual_income': np.random.lognormal(11, 0.3, n).astype(int),
        'credit_score': np.random.normal(650, 100, n).astype(int),
        'months_as_customer': np.random.poisson(24, n),
        'products_owned': np.random.poisson(2.5, n),
        'monthly_charges': np.random.normal(75, 20, n),
        'total_charges': None,  # Will calculate
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'customer_segment': np.random.choice(['Enterprise', 'SMB', 'Consumer'], n, p=[0.15, 0.35, 0.5]),
        'satisfaction_rating': np.random.uniform(1, 10, n),
        'churn_risk': np.random.choice(['Low', 'Medium', 'High'], n, p=[0.6, 0.3, 0.1])
    }
    
    # Add realistic relationships
    # Total charges = monthly charges * months as customer + some noise
    data['total_charges'] = (data['monthly_charges'] * data['months_as_customer'] + 
                            np.random.normal(0, 100, n))
    
    # Satisfaction influenced by service quality (inverse of charges relative to income)
    charge_to_income_ratio = np.array(data['monthly_charges']) / (np.array(data['annual_income']) / 12)
    data['satisfaction_rating'] = (data['satisfaction_rating'] - 
                                 charge_to_income_ratio * 2 + 
                                 np.array(data['credit_score']) / 200)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values realistically
    missing_indices = np.random.choice(n, int(n * 0.03), replace=False)
    df.loc[missing_indices, 'satisfaction_rating'] = np.nan
    
    # Save demo data
    demo_path = Path('data/demo_business_data.csv')
    demo_path.parent.mkdir(exist_ok=True)
    df.to_csv(demo_path, index=False)
    
    print(f"✅ Demo dataset created: {demo_path}")
    print(f"   📈 {n} customers with {len(df.columns)} attributes")
    return str(demo_path)

def run_demo():
    """Run the main demonstration"""
    
    print("🤖 AI Data Analyzer - Live Demo")
    print("=" * 50)
    print("Welcome to the AI-powered data analysis demonstration!")
    print("This tool analyzes your data using simple natural language prompts.\n")
    
    # Create demo data
    data_path = create_demo_data()
    
    try:
        from ai_data_analyzer import AIDataAnalyzer
        
        # Initialize
        print("🔧 Initializing AI Data Analyzer...")
        analyzer = AIDataAnalyzer()
        
        # Load data
        print(f"📥 Loading demo data...")
        load_result = analyzer.load_data(data_path)
        
        if not load_result['success']:
            print(f"❌ Could not load data: {load_result['error']}")
            return
        
        print(f"✅ Loaded: {load_result['message']}")
        
        # Show what the AI can do
        print("\n🎯 Here's what the AI can analyze with simple prompts:")
        
        demo_prompts = [
            {
                'prompt': 'Describe this business dataset',
                'description': '📋 Basic data overview and statistics'
            },
            {
                'prompt': 'Show correlations between customer income, charges, and satisfaction',
                'description': '🔗 Relationship analysis between key metrics'
            },
            {
                'prompt': 'Find customer segments using income, age, and satisfaction',
                'description': '👥 Customer segmentation using machine learning'
            },
            {
                'prompt': 'Predict satisfaction rating using customer characteristics',
                'description': '🎯 Predictive modeling for customer satisfaction'
            }
        ]
        
        # Run demonstration analyses
        for i, demo in enumerate(demo_prompts, 1):
            print(f"\n{i}. {demo['description']}")
            print(f"   💬 Prompt: '{demo['prompt']}'")
            print("   🔄 Analyzing...")
            
            try:
                results = analyzer.analyze_with_prompt(demo['prompt'])
                
                if results['success']:
                    print("   ✅ Analysis complete!")
                    
                    # Show top insights
                    insights = results['analysis_result'].insights
                    if insights:
                        print("   🔍 Key Findings:")
                        for insight in insights[:2]:
                            print(f"      • {insight}")
                    
                    # Show if reports were generated
                    reports = results['report'].get('formatted_reports', {})
                    if reports.get('html'):
                        print(f"   📄 Report saved: {Path(reports['html']).name}")
                
                else:
                    print(f"   ⚠️ Analysis issue: {results['error']}")
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
            
            print("   " + "─" * 40)
        
        # Show additional capabilities
        print("\n🌟 Additional Capabilities:")
        print("   🔍 Anomaly Detection: 'Find unusual patterns in customer data'")
        print("   📊 Custom Analysis: 'Analyze only high-value customers'")
        print("   📈 Trend Analysis: 'Show trends in monthly charges over time'")
        print("   🎨 Visualizations: Automatic charts and interactive plots")
        print("   📝 Smart Reports: HTML, Markdown, and JSON formats")
        
        # Show outputs
        output_dir = Path(analyzer.config['output']['directory'])
        if output_dir.exists():
            files = list(output_dir.glob('*'))
            if files:
                print(f"\n💾 Generated {len(files)} output files in {output_dir}/")
                for file in files[:5]:  # Show first 5
                    print(f"   📄 {file.name}")
        
        print("\n✨ Demo completed! The AI Data Analyzer is ready for your data.")
        print("\n🚀 Next Steps:")
        print("   1. Try with your own data files")
        print("   2. Use interactive mode: python src/ai_data_analyzer.py --interactive")
        print("   3. Explore configuration options in config/config.yaml")
        print("   4. Check the README.md for complete documentation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install dependencies first: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
        print("💡 Check that all files are in place and dependencies are installed")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
