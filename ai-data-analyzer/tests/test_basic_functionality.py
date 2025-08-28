"""
Basic functionality tests for AI Data Analyzer
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_ingestion import DataIngestor
from prompt_parser import PromptParser, AnalysisType
from analysis_engines import DescriptiveAnalyzer, CorrelationAnalyzer
from ai_data_analyzer import AIDataAnalyzer

# Test configuration
TEST_CONFIG = {
    'analysis': {
        'alpha': 0.05,
        'correlation_threshold': 0.7,
        'test_size': 0.2,
        'random_state': 42
    },
    'visualization': {
        'style': 'default',
        'palette': 'viridis',
        'figsize': [8, 6]
    },
    'output': {
        'directory': './test_outputs',
        'include_visualizations': False  # Disable for tests
    },
    'data': {
        'max_file_size': 100,
        'auto_dtype': True,
        'parse_dates': True
    },
    'logging': {
        'level': 'WARNING'  # Reduce log noise in tests
    }
}

@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing"""
    np.random.seed(42)
    n = 100
    
    data = {
        'numeric_col1': np.random.normal(50, 10, n),
        'numeric_col2': np.random.normal(30, 5, n),
        'categorical_col': np.random.choice(['A', 'B', 'C'], n),
        'date_col': pd.date_range('2020-01-01', periods=n, freq='D'),
        'target': np.random.randint(0, 2, n)
    }
    
    # Add correlation between numeric columns
    data['numeric_col2'] = data['numeric_col1'] * 0.8 + np.random.normal(0, 2, n)
    
    return pd.DataFrame(data)

class TestDataIngestion:
    """Test data ingestion functionality"""
    
    def test_csv_ingestion(self, sample_dataframe, tmp_path):
        """Test CSV file ingestion"""
        # Save sample data to CSV
        csv_path = tmp_path / "test_data.csv"
        sample_dataframe.to_csv(csv_path, index=False)
        
        # Test ingestion
        ingestor = DataIngestor(TEST_CONFIG)
        df = ingestor.load_data(csv_path)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0
    
    def test_data_summary(self, sample_dataframe):
        """Test data summary generation"""
        ingestor = DataIngestor(TEST_CONFIG)
        summary = ingestor.get_data_summary(sample_dataframe)
        
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert summary['shape'] == sample_dataframe.shape

class TestPromptParser:
    """Test prompt parsing functionality"""
    
    def test_descriptive_intent_parsing(self):
        """Test parsing descriptive analysis prompts"""
        parser = PromptParser(TEST_CONFIG)
        
        prompts = [
            "describe my data",
            "show basic statistics",
            "what's in this dataset"
        ]
        
        for prompt in prompts:
            intent = parser.parse_prompt(prompt)
            assert intent.analysis_type == AnalysisType.DESCRIPTIVE
    
    def test_correlation_intent_parsing(self):
        """Test parsing correlation analysis prompts"""
        parser = PromptParser(TEST_CONFIG)
        
        prompts = [
            "show correlations",
            "how are variables related",
            "correlation matrix"
        ]
        
        for prompt in prompts:
            intent = parser.parse_prompt(prompt)
            assert intent.analysis_type == AnalysisType.CORRELATION
    
    def test_predictive_intent_parsing(self):
        """Test parsing predictive analysis prompts"""
        parser = PromptParser(TEST_CONFIG)
        
        prompts = [
            "predict target using features",
            "build a model",
            "forecast values"
        ]
        
        for prompt in prompts:
            intent = parser.parse_prompt(prompt)
            assert intent.analysis_type == AnalysisType.PREDICTIVE
    
    def test_column_extraction(self, sample_dataframe):
        """Test column name extraction from prompts"""
        parser = PromptParser(TEST_CONFIG)
        columns = list(sample_dataframe.columns)
        
        prompt = "predict target using numeric_col1"
        intent = parser.parse_prompt(prompt, columns)
        
        assert intent.target_column == 'target'
        assert 'numeric_col1' in intent.features or intent.features is None

class TestAnalysisEngines:
    """Test analysis engine functionality"""
    
    def test_descriptive_analyzer(self, sample_dataframe):
        """Test descriptive analysis"""
        analyzer = DescriptiveAnalyzer(TEST_CONFIG)
        result = analyzer.analyze(sample_dataframe)
        
        assert result.success
        assert result.analysis_type == "descriptive"
        assert 'shape' in result.results
        assert 'columns' in result.results
        assert len(result.insights) > 0
    
    def test_correlation_analyzer(self, sample_dataframe):
        """Test correlation analysis"""
        analyzer = CorrelationAnalyzer(TEST_CONFIG)
        
        # Use only numeric columns
        numeric_cols = ['numeric_col1', 'numeric_col2']
        result = analyzer.analyze(sample_dataframe, columns=numeric_cols)
        
        assert result.success
        assert result.analysis_type == "correlation"
        assert 'correlation_matrix' in result.results
        assert len(result.insights) > 0

class TestIntegration:
    """Test full system integration"""
    
    def test_full_analysis_workflow(self, sample_dataframe, tmp_path):
        """Test complete analysis workflow"""
        # Save sample data
        csv_path = tmp_path / "integration_test.csv"
        sample_dataframe.to_csv(csv_path, index=False)
        
        # Initialize analyzer with test config
        analyzer = AIDataAnalyzer()
        analyzer.config = TEST_CONFIG
        analyzer.config['output']['directory'] = str(tmp_path / 'outputs')
        
        # Load data
        load_result = analyzer.load_data(csv_path)
        assert load_result['success']
        
        # Perform analysis
        results = analyzer.analyze_with_prompt("describe my data")
        
        assert results['success']
        assert 'analysis_result' in results
        assert 'report' in results
        assert results['analysis_result'].success
    
    def test_multiple_analysis_types(self, sample_dataframe, tmp_path):
        """Test multiple analysis types"""
        # Save sample data
        csv_path = tmp_path / "multi_test.csv"
        sample_dataframe.to_csv(csv_path, index=False)
        
        # Initialize analyzer
        analyzer = AIDataAnalyzer()
        analyzer.config = TEST_CONFIG
        analyzer.config['output']['directory'] = str(tmp_path / 'outputs')
        
        # Load data
        analyzer.load_data(csv_path)
        
        # Test different analysis types
        test_cases = [
            ("describe the data", AnalysisType.DESCRIPTIVE),
            ("show correlations", AnalysisType.CORRELATION),
            ("find clusters", AnalysisType.CLUSTERING)
        ]
        
        for prompt, expected_type in test_cases:
            results = analyzer.analyze_with_prompt(prompt)
            
            # Check that analysis completes (may not always succeed depending on data)
            assert 'analysis_result' in results
            
            # Parse the prompt to verify intent detection
            intent = analyzer.prompt_parser.parse_prompt(prompt, list(sample_dataframe.columns))
            assert intent.analysis_type == expected_type

class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_data_source(self):
        """Test handling of invalid data sources"""
        analyzer = AIDataAnalyzer()
        analyzer.config = TEST_CONFIG
        
        result = analyzer.load_data("nonexistent_file.csv")
        assert not result['success']
        assert 'error' in result
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframes"""
        analyzer = DescriptiveAnalyzer(TEST_CONFIG)
        empty_df = pd.DataFrame()
        
        result = analyzer.analyze(empty_df)
        assert not result.success
    
    def test_invalid_analysis_intent(self, sample_dataframe):
        """Test validation of analysis intents"""
        parser = PromptParser(TEST_CONFIG)
        
        # Test predictive analysis without target column
        intent = parser.parse_prompt("predict something", list(sample_dataframe.columns))
        intent.analysis_type = AnalysisType.PREDICTIVE
        intent.target_column = None
        
        is_valid, issues = parser.validate_intent(intent, sample_dataframe)
        assert not is_valid
        assert len(issues) > 0

# Test runner
if __name__ == "__main__":
    print("🧪 Running AI Data Analyzer Tests")
    print("=" * 40)
    
    # Run tests with pytest
    pytest.main([__file__, "-v"])
