"""
AI Data Analyzer Package
Automated data analysis with natural language prompts
"""

__version__ = "1.0.0"
__author__ = "AI Data Solutions"
__email__ = "info@aidatasolutions.com"

from .ai_data_analyzer import AIDataAnalyzer, InteractiveAnalyzer
from .data_ingestion import DataIngestor, DataIngestionError
from .prompt_parser import PromptParser, AnalysisIntent, AnalysisType, VisualizationType
from .analysis_engines import AnalysisEngineFactory, AnalysisResult
from .visualization_generator import VisualizationFactory, VisualizationResult
from .report_generator import ReportGenerator

__all__ = [
    'AIDataAnalyzer',
    'InteractiveAnalyzer', 
    'DataIngestor',
    'DataIngestionError',
    'PromptParser',
    'AnalysisIntent',
    'AnalysisType',
    'VisualizationType',
    'AnalysisEngineFactory',
    'AnalysisResult',
    'VisualizationFactory',
    'VisualizationResult',
    'ReportGenerator'
]
