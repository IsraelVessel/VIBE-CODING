"""
AI Data Analyzer - Main Application
Ties everything together with a simple prompt-based interface
"""

import pandas as pd
import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import argparse
from datetime import datetime

# Import our modules
from data_ingestion import DataIngestor, DataIngestionError
from prompt_parser import PromptParser, AnalysisIntent, AnalysisType
from analysis_engines import AnalysisEngineFactory, AnalysisResult
from visualization_generator import VisualizationFactory, VisualizationResult
from report_generator import ReportGenerator

# Set up logging
def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_file = log_config.get('file', './logs/analyzer.log')
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

class AIDataAnalyzer:
    """Main AI Data Analyzer application"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the AI Data Analyzer"""
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = self._get_default_config()
        
        # Setup logging
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_ingestor = DataIngestor(self.config)
        self.prompt_parser = PromptParser(self.config)
        self.report_generator = ReportGenerator(self.config)
        
        # Current data state
        self.current_data = None
        self.data_summary = None
        
        self.logger.info("AI Data Analyzer initialized successfully")
    
    def analyze_with_prompt(self, prompt: str, data_source: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Main method to analyze data based on natural language prompt
        
        Args:
            prompt: Natural language description of desired analysis
            data_source: Path to data file, URL, or database connection string
            
        Returns:
            Dictionary containing analysis results, visualizations, and report
        """
        
        try:
            self.logger.info(f"Starting analysis for prompt: '{prompt}'")
            
            # Load data if new source provided
            if data_source:
                self.load_data(data_source)
            elif self.current_data is None:
                return {
                    'success': False,
                    'error': 'No data loaded. Please provide a data source.',
                    'prompt': prompt
                }
            
            # Parse the prompt
            intent = self.prompt_parser.parse_prompt(prompt, list(self.current_data.columns))
            self.logger.info(f"Parsed intent: {intent.analysis_type.value} (confidence: {intent.confidence:.2f})")
            
            # Validate intent against data
            is_valid, issues = self.prompt_parser.validate_intent(intent, self.current_data)
            if not is_valid:
                return {
                    'success': False,
                    'error': f'Cannot perform analysis: {"; ".join(issues)}',
                    'intent': intent,
                    'prompt': prompt
                }
            
            # Perform analysis
            analysis_result = self._execute_analysis(intent)
            
            # Generate visualizations
            visualizations = self._generate_visualizations(analysis_result)
            
            # Generate report
            report = self.report_generator.generate_report(
                analysis_result, visualizations, self.data_summary, prompt
            )
            
            # Compile results
            results = {
                'success': True,
                'prompt': prompt,
                'intent': intent,
                'analysis_result': analysis_result,
                'visualizations': visualizations,
                'report': report,
                'data_summary': self.data_summary,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'prompt': prompt,
                'timestamp': datetime.now().isoformat()
            }
    
    def load_data(self, data_source: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Load data from various sources
        
        Args:
            data_source: Path, URL, or connection string
            **kwargs: Additional parameters for data loading
            
        Returns:
            Dictionary with loading results and data summary
        """
        
        try:
            self.logger.info(f"Loading data from: {data_source}")
            
            # Load data using data ingestor
            self.current_data = self.data_ingestor.load_data(data_source, **kwargs)
            
            # Generate data summary
            self.data_summary = self.data_ingestor.get_data_summary(self.current_data)
            
            self.logger.info(f"Data loaded successfully: {self.current_data.shape}")
            
            return {
                'success': True,
                'message': f'Data loaded successfully: {self.current_data.shape[0]} rows, {self.current_data.shape[1]} columns',
                'data_summary': self.data_summary
            }
            
        except DataIngestionError as e:
            self.logger.error(f"Data ingestion error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
        except Exception as e:
            self.logger.error(f"Unexpected error loading data: {str(e)}")
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def suggest_analyses(self) -> List[AnalysisIntent]:
        """Suggest possible analyses based on current data"""
        
        if self.current_data is None:
            return []
        
        return self.prompt_parser.suggest_analysis(self.current_data)
    
    def get_data_info(self) -> Optional[Dict[str, Any]]:
        """Get information about currently loaded data"""
        
        if self.current_data is None:
            return None
        
        return {
            'shape': self.current_data.shape,
            'columns': list(self.current_data.columns),
            'dtypes': self.current_data.dtypes.to_dict(),
            'head': self.current_data.head().to_dict(),
            'summary': self.data_summary
        }
    
    def _execute_analysis(self, intent: AnalysisIntent) -> AnalysisResult:
        """Execute the analysis based on intent"""
        
        try:
            # Create appropriate analyzer
            analyzer = AnalysisEngineFactory.create_analyzer(intent.analysis_type.value, self.config)
            
            # Prepare arguments based on analysis type
            kwargs = {}
            
            if intent.analysis_type == AnalysisType.DESCRIPTIVE:
                if intent.features:
                    kwargs['columns'] = intent.features
            
            elif intent.analysis_type == AnalysisType.CORRELATION:
                if intent.features:
                    kwargs['columns'] = intent.features
                kwargs.update(intent.parameters or {})
            
            elif intent.analysis_type == AnalysisType.PREDICTIVE:
                if not intent.target_column:
                    raise ValueError("Predictive analysis requires a target column")
                kwargs['target_column'] = intent.target_column
                if intent.features:
                    kwargs['feature_columns'] = intent.features
                kwargs.update(intent.parameters or {})
            
            elif intent.analysis_type == AnalysisType.CLUSTERING:
                if intent.features:
                    kwargs['columns'] = intent.features
                kwargs.update(intent.parameters or {})
            
            elif intent.analysis_type == AnalysisType.ANOMALY:
                if intent.features:
                    kwargs['columns'] = intent.features
                kwargs.update(intent.parameters or {})
            
            # Apply filters if specified
            data_to_analyze = self.current_data
            if intent.filters:
                data_to_analyze = self._apply_filters(data_to_analyze, intent.filters)
            
            # Execute analysis
            result = analyzer.analyze(data_to_analyze, **kwargs)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing analysis: {str(e)}")
            return AnalysisResult(
                analysis_type=intent.analysis_type.value,
                results={},
                insights=[],
                warnings=[f"Analysis failed: {str(e)}"],
                metadata={},
                success=False
            )
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the dataframe"""
        
        filtered_df = df.copy()
        
        for column, filter_spec in filters.items():
            if column not in df.columns:
                self.logger.warning(f"Filter column '{column}' not found in data")
                continue
            
            operator = filter_spec.get('operator', '=')
            value = filter_spec.get('value')
            
            try:
                if operator == '=':
                    filtered_df = filtered_df[filtered_df[column] == value]
                elif operator == '>':
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif operator == '<':
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif operator == '>=':
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif operator == '<=':
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif operator == '!=':
                    filtered_df = filtered_df[filtered_df[column] != value]
                
                self.logger.info(f"Applied filter: {column} {operator} {value}")
                
            except Exception as e:
                self.logger.warning(f"Could not apply filter {column} {operator} {value}: {str(e)}")
        
        return filtered_df
    
    def _generate_visualizations(self, analysis_result: AnalysisResult) -> List[VisualizationResult]:
        """Generate visualizations for the analysis result"""
        
        try:
            if not self.config.get('output', {}).get('include_visualizations', True):
                return []
            
            visualizations = VisualizationFactory.create_visualization(
                analysis_result, self.current_data, self.config
            )
            
            self.logger.info(f"Generated {len(visualizations)} visualizations")
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            return []
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if no config file provided"""
        
        return {
            'analysis': {
                'alpha': 0.05,
                'max_features': 100,
                'correlation_threshold': 0.7,
                'missing_data_threshold': 0.9,
                'test_size': 0.2,
                'cv_folds': 5,
                'random_state': 42
            },
            'visualization': {
                'style': 'default',
                'palette': 'viridis',
                'figsize': [12, 8],
                'dpi': 300
            },
            'output': {
                'directory': './outputs',
                'report_format': 'html',
                'include_visualizations': True,
                'save_raw_results': True
            },
            'data': {
                'max_file_size': 500,
                'supported_formats': ['csv', 'xlsx', 'xls', 'json', 'parquet'],
                'auto_dtype': True,
                'parse_dates': True
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': './logs/analyzer.log'
            }
        }

# Command Line Interface
def main():
    """Command line interface for the AI Data Analyzer"""
    
    parser = argparse.ArgumentParser(description='AI Data Analyzer - Automated data analysis with natural language prompts')
    
    parser.add_argument('--data', '-d', type=str, required=True,
                       help='Path to data file, URL, or database connection string')
    parser.add_argument('--prompt', '-p', type=str, required=True,
                       help='Natural language prompt describing the desired analysis')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to configuration file (optional)')
    parser.add_argument('--output', '-o', type=str, default='./outputs',
                       help='Output directory for results (default: ./outputs)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        config_path = args.config or str(Path(__file__).parent.parent / 'config' / 'config.yaml')
        analyzer = AIDataAnalyzer(config_path)
        
        # Override output directory if specified
        if args.output != './outputs':
            analyzer.config['output']['directory'] = args.output
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        print("🤖 AI Data Analyzer Starting...")
        print(f"📊 Loading data from: {args.data}")
        
        # Load data
        load_result = analyzer.load_data(args.data)
        if not load_result['success']:
            print(f"❌ Error loading data: {load_result['error']}")
            sys.exit(1)
        
        print(f"✅ {load_result['message']}")
        print(f"🔍 Analyzing with prompt: '{args.prompt}'")
        
        # Perform analysis
        results = analyzer.analyze_with_prompt(args.prompt)
        
        if results['success']:
            print("✅ Analysis completed successfully!")
            print(f"📈 Analysis type: {results['analysis_result'].analysis_type}")
            print(f"📊 Generated {len(results.get('visualizations', []))} visualizations")
            
            # Print key insights
            insights = results['analysis_result'].insights
            if insights:
                print("\n🔍 Key Insights:")
                for i, insight in enumerate(insights[:5], 1):  # Show top 5
                    print(f"   {i}. {insight}")
            
            # Print warnings if any
            warnings = results['analysis_result'].warnings
            if warnings:
                print("\n⚠️  Warnings:")
                for warning in warnings:
                    print(f"   • {warning}")
            
            # Print output locations
            formatted_reports = results['report'].get('formatted_reports', {})
            if formatted_reports:
                print("\n📄 Reports generated:")
                for format_type, file_path in formatted_reports.items():
                    print(f"   • {format_type.upper()}: {file_path}")
            
            print(f"\n💾 All outputs saved to: {analyzer.config['output']['directory']}")
            
        else:
            print(f"❌ Analysis failed: {results['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

# Interactive Mode
class InteractiveAnalyzer:
    """Interactive mode for the AI Data Analyzer"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.analyzer = AIDataAnalyzer(config_path)
        self.session_history = []
    
    def run(self):
        """Run interactive analysis session"""
        
        print("🤖 Welcome to AI Data Analyzer - Interactive Mode")
        print("Type 'help' for available commands, 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\nAnalyzer> ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower().startswith('load '):
                    data_path = user_input[5:].strip()
                    self._handle_load_command(data_path)
                    continue
                
                elif user_input.lower() == 'info':
                    self._show_data_info()
                    continue
                
                elif user_input.lower() == 'suggest':
                    self._show_suggestions()
                    continue
                
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                # Handle analysis prompt
                if self.analyzer.current_data is None:
                    print("❌ No data loaded. Use 'load <file_path>' to load data first.")
                    continue
                
                # Perform analysis
                print("🔄 Analyzing...")
                results = self.analyzer.analyze_with_prompt(user_input)
                
                # Display results
                self._display_results(results)
                
                # Add to history
                self.session_history.append({
                    'prompt': user_input,
                    'timestamp': datetime.now().isoformat(),
                    'success': results['success']
                })
                
            except KeyboardInterrupt:
                print("\n🛑 Use 'quit' to exit")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def _handle_load_command(self, data_path: str):
        """Handle load data command"""
        result = self.analyzer.load_data(data_path)
        if result['success']:
            print(f"✅ {result['message']}")
        else:
            print(f"❌ {result['error']}")
    
    def _show_data_info(self):
        """Show information about currently loaded data"""
        info = self.analyzer.get_data_info()
        if info:
            print(f"📊 Data Shape: {info['shape'][0]:,} rows × {info['shape'][1]} columns")
            print(f"📋 Columns: {', '.join(info['columns'][:10])}{'...' if len(info['columns']) > 10 else ''}")
            print(f"🏷️  Data Types: {len(info['dtypes'])} unique types")
        else:
            print("❌ No data loaded")
    
    def _show_suggestions(self):
        """Show suggested analyses"""
        suggestions = self.analyzer.suggest_analyses()
        if suggestions:
            print("💡 Suggested analyses:")
            for i, suggestion in enumerate(suggestions[:5], 1):
                print(f"   {i}. {suggestion.raw_prompt}")
        else:
            print("❌ No data loaded or no suggestions available")
    
    def _show_history(self):
        """Show session history"""
        if self.session_history:
            print("📜 Session History:")
            for i, item in enumerate(self.session_history[-10:], 1):  # Last 10
                status = "✅" if item['success'] else "❌"
                print(f"   {i}. {status} {item['prompt'][:50]}{'...' if len(item['prompt']) > 50 else ''}")
        else:
            print("📜 No analysis history")
    
    def _display_results(self, results: Dict[str, Any]):
        """Display analysis results"""
        
        if results['success']:
            print("✅ Analysis completed!")
            
            # Show insights
            insights = results['analysis_result'].insights
            if insights:
                print("\n🔍 Key Insights:")
                for insight in insights[:3]:  # Top 3
                    print(f"   • {insight}")
            
            # Show warnings
            warnings = results['analysis_result'].warnings
            if warnings:
                print("\n⚠️  Warnings:")
                for warning in warnings[:2]:  # Top 2
                    print(f"   • {warning}")
            
            # Show report location
            report_files = results['report'].get('formatted_reports', {})
            if report_files:
                print(f"\n📄 Full report: {report_files.get('html', 'Not generated')}")
            
        else:
            print(f"❌ Analysis failed: {results['error']}")
    
    def _show_help(self):
        """Show help information"""
        
        help_text = """
🤖 AI Data Analyzer - Interactive Mode Help

Available Commands:
  help                    - Show this help message
  load <file_path>       - Load data from file, URL, or database
  info                   - Show information about loaded data
  suggest                - Show suggested analyses for current data
  history                - Show analysis history for this session
  quit/exit/q           - Exit the application

Analysis Examples:
  "Describe my data"
  "Show correlations between all variables"
  "Predict sales using all features"
  "Find clusters in the customer data"
  "Detect anomalies in the transaction data"
  "Create a histogram of prices"
  "Compare revenue by region"

Tips:
  • Load data first before running analysis
  • Be specific about columns you want to analyze
  • Use natural language - the AI will understand your intent
  • Check suggestions if you're not sure what to analyze
        """
        
        print(help_text)

if __name__ == "__main__":
    # Check if running in interactive mode
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['--interactive', '-i']):
        # Interactive mode
        interactive = InteractiveAnalyzer()
        interactive.run()
    else:
        # Command line mode
        main()
