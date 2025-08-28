"""
Prompt Parser and Intent Recognition Module
Understands user prompts and maps them to specific analysis types
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of analysis that can be performed"""
    DESCRIPTIVE = "descriptive"
    STATISTICAL = "statistical"
    CORRELATION = "correlation"
    TREND = "trend"
    PREDICTIVE = "predictive"
    CLUSTERING = "clustering"
    ANOMALY = "anomaly"
    VISUALIZATION = "visualization"
    COMPARISON = "comparison"
    DISTRIBUTION = "distribution"
    TIME_SERIES = "time_series"
    CUSTOM = "custom"

class VisualizationType(Enum):
    """Types of visualizations"""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    HEATMAP = "heatmap"
    PIE_CHART = "pie_chart"
    CORRELATION_MATRIX = "correlation_matrix"
    PAIR_PLOT = "pair_plot"
    TIME_SERIES_PLOT = "time_series_plot"

@dataclass
class AnalysisIntent:
    """Represents the parsed intent from a user prompt"""
    analysis_type: AnalysisType
    visualization_type: Optional[VisualizationType] = None
    target_column: Optional[str] = None
    features: List[str] = None
    filters: Dict[str, Any] = None
    parameters: Dict[str, Any] = None
    confidence: float = 0.0
    raw_prompt: str = ""

class PromptParser:
    """Main class for parsing user prompts and extracting analysis intent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize regex patterns for intent recognition"""
        
        # Analysis type patterns
        self.analysis_patterns = {
            AnalysisType.DESCRIPTIVE: [
                r'describ|summar|overview|basic|general|profile',
                r'what.*data|show.*data|explore.*data',
                r'statistics|stats|info|information'
            ],
            AnalysisType.STATISTICAL: [
                r'test|significant|p.value|hypothesis|statistical',
                r't.test|chi.square|anova|regression|correlation',
                r'confidence|interval|distribution'
            ],
            AnalysisType.CORRELATION: [
                r'correlat|relationship|associat|connect',
                r'how.*related|relationship between|depends on',
                r'correlation matrix|corr'
            ],
            AnalysisType.TREND: [
                r'trend|pattern|over time|temporal|time series',
                r'growing|declining|increasing|decreasing',
                r'seasonal|cyclical|moving average'
            ],
            AnalysisType.PREDICTIVE: [
                r'predict|forecast|estimate|model|machine learning',
                r'future|next|upcoming|project|anticipate',
                r'classification|regression|neural network|random forest'
            ],
            AnalysisType.CLUSTERING: [
                r'cluster|group|segment|category|classify',
                r'similar|k.means|hierarchical|dbscan',
                r'pattern|segment|group by similarity'
            ],
            AnalysisType.ANOMALY: [
                r'anomaly|outlier|unusual|abnormal|detect',
                r'strange|weird|different|exception',
                r'fraud|outlier detection'
            ],
            AnalysisType.VISUALIZATION: [
                r'plot|chart|graph|visual|show|display',
                r'histogram|scatter|bar|line|pie',
                r'visualiz|draw|create.*chart'
            ],
            AnalysisType.COMPARISON: [
                r'compar|versus|vs|difference|against',
                r'better|worse|higher|lower',
                r'benchmark|baseline'
            ],
            AnalysisType.DISTRIBUTION: [
                r'distribution|spread|normal|uniform|skew',
                r'histogram|density|bell curve',
                r'quartile|percentile'
            ]
        }
        
        # Visualization type patterns
        self.viz_patterns = {
            VisualizationType.BAR_CHART: [r'bar|column|categorical'],
            VisualizationType.LINE_CHART: [r'line|trend|time.*series'],
            VisualizationType.SCATTER_PLOT: [r'scatter|relationship|correlation'],
            VisualizationType.HISTOGRAM: [r'histogram|distribution|frequency'],
            VisualizationType.BOX_PLOT: [r'box|quartile|outlier'],
            VisualizationType.HEATMAP: [r'heatmap|correlation.*matrix|heat map'],
            VisualizationType.PIE_CHART: [r'pie|proportion|percentage'],
            VisualizationType.PAIR_PLOT: [r'pair.*plot|matrix.*plot'],
            VisualizationType.TIME_SERIES_PLOT: [r'time.*series|temporal|over.*time']
        }
        
        # Column extraction patterns
        self.column_patterns = [
            r'column[:\s]+(["\']?)(\w+)\1',
            r'field[:\s]+(["\']?)(\w+)\1',
            r'variable[:\s]+(["\']?)(\w+)\1',
            r'feature[:\s]+(["\']?)(\w+)\1',
            r'target[:\s]+(["\']?)(\w+)\1',
            r'predict[:\s]+(["\']?)(\w+)\1',
            r'for[:\s]+(["\']?)(\w+)\1',
            r'(["\'])(\w+)\1\s+column',
            r'analyze[:\s]+(["\']?)(\w+)\1'
        ]
        
        # Filter patterns
        self.filter_patterns = [
            r'where\s+(\w+)\s*(=|>|<|>=|<=|!=)\s*(["\']?)([^"\']+)\3',
            r'filter.*(\w+)\s*(=|>|<|>=|<=|!=)\s*(["\']?)([^"\']+)\3',
            r'only.*(\w+)\s*(=|>|<|>=|<=|!=)\s*(["\']?)([^"\']+)\3'
        ]
    
    def parse_prompt(self, prompt: str, data_columns: Optional[List[str]] = None) -> AnalysisIntent:
        """
        Parse a user prompt and extract analysis intent
        
        Args:
            prompt: User's natural language prompt
            data_columns: Available columns in the dataset (for context)
            
        Returns:
            AnalysisIntent: Parsed intent with analysis type and parameters
        """
        prompt_lower = prompt.lower()
        
        # Extract analysis type
        analysis_type, analysis_confidence = self._extract_analysis_type(prompt_lower)
        
        # Extract visualization type if mentioned
        visualization_type, viz_confidence = self._extract_visualization_type(prompt_lower)
        
        # Extract target column and features
        target_column = self._extract_target_column(prompt, data_columns)
        features = self._extract_features(prompt, data_columns, target_column)
        
        # Extract filters
        filters = self._extract_filters(prompt)
        
        # Extract additional parameters
        parameters = self._extract_parameters(prompt_lower)
        
        # Calculate overall confidence
        confidence = max(analysis_confidence, viz_confidence)
        
        # Create intent object
        intent = AnalysisIntent(
            analysis_type=analysis_type,
            visualization_type=visualization_type,
            target_column=target_column,
            features=features,
            filters=filters,
            parameters=parameters,
            confidence=confidence,
            raw_prompt=prompt
        )
        
        logger.info(f"Parsed intent: {analysis_type.value} (confidence: {confidence:.2f})")
        
        return intent
    
    def _extract_analysis_type(self, prompt: str) -> Tuple[AnalysisType, float]:
        """Extract the primary analysis type from the prompt"""
        scores = {}
        
        for analysis_type, patterns in self.analysis_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt, re.IGNORECASE))
                score += matches
            
            if score > 0:
                # Weight based on pattern specificity
                if analysis_type in [AnalysisType.PREDICTIVE, AnalysisType.CLUSTERING, 
                                   AnalysisType.ANOMALY]:
                    score *= 1.5  # Boost specific analysis types
                scores[analysis_type] = score
        
        if not scores:
            # Default to descriptive analysis
            return AnalysisType.DESCRIPTIVE, 0.5
        
        # Get the analysis type with highest score
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / 3.0, 1.0)  # Normalize to 0-1
        
        return best_type, confidence
    
    def _extract_visualization_type(self, prompt: str) -> Tuple[Optional[VisualizationType], float]:
        """Extract visualization type if mentioned"""
        scores = {}
        
        for viz_type, patterns in self.viz_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt, re.IGNORECASE))
                score += matches
            
            if score > 0:
                scores[viz_type] = score
        
        if not scores:
            return None, 0.0
        
        best_viz = max(scores, key=scores.get)
        confidence = min(scores[best_viz] / 2.0, 1.0)
        
        return best_viz, confidence
    
    def _extract_target_column(self, prompt: str, data_columns: Optional[List[str]]) -> Optional[str]:
        """Extract target column from the prompt"""
        
        # Try pattern matching first
        for pattern in self.column_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                # Get the column name (handle tuple matches)
                column = matches[0]
                if isinstance(column, tuple):
                    column = column[-1]  # Get the last group (actual column name)
                
                # Validate against available columns if provided
                if data_columns and column.lower() in [col.lower() for col in data_columns]:
                    return self._get_exact_column_name(column, data_columns)
                elif not data_columns:
                    return column
        
        # If no patterns match, look for column names directly mentioned
        if data_columns:
            for col in data_columns:
                if col.lower() in prompt.lower():
                    return col
        
        return None
    
    def _extract_features(self, prompt: str, data_columns: Optional[List[str]], 
                         target_column: Optional[str]) -> List[str]:
        """Extract feature columns from the prompt"""
        features = []
        
        # Look for multiple column mentions
        if data_columns:
            mentioned_cols = []
            for col in data_columns:
                if col.lower() in prompt.lower() and col != target_column:
                    mentioned_cols.append(col)
            features.extend(mentioned_cols)
        
        # Look for "all columns" or "all features" mentions
        if re.search(r'all\s+(columns|features|variables)', prompt, re.IGNORECASE):
            if data_columns:
                features = [col for col in data_columns if col != target_column]
        
        return list(set(features))  # Remove duplicates
    
    def _extract_filters(self, prompt: str) -> Dict[str, Any]:
        """Extract filter conditions from the prompt"""
        filters = {}
        
        for pattern in self.filter_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                if len(match) >= 4:
                    column = match[0]
                    operator = match[1]
                    value = match[3].strip()
                    
                    # Try to convert value to appropriate type
                    try:
                        if value.replace('.', '', 1).isdigit():
                            value = float(value) if '.' in value else int(value)
                    except:
                        pass  # Keep as string
                    
                    filters[column] = {'operator': operator, 'value': value}
        
        return filters
    
    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract analysis-specific parameters"""
        parameters = {}
        
        # Look for confidence level
        conf_match = re.search(r'confidence.*?(\d+(?:\.\d+)?)', prompt)
        if conf_match:
            parameters['confidence_level'] = float(conf_match.group(1)) / 100
        
        # Look for number of clusters
        cluster_match = re.search(r'(\d+)\s+clusters?', prompt)
        if cluster_match:
            parameters['n_clusters'] = int(cluster_match.group(1))
        
        # Look for prediction horizon
        horizon_match = re.search(r'next\s+(\d+)', prompt)
        if horizon_match:
            parameters['forecast_horizon'] = int(horizon_match.group(1))
        
        # Look for test size
        test_match = re.search(r'test.*?(\d+(?:\.\d+)?)%?', prompt)
        if test_match:
            value = float(test_match.group(1))
            parameters['test_size'] = value / 100 if value > 1 else value
        
        return parameters
    
    def _get_exact_column_name(self, column_lower: str, data_columns: List[str]) -> str:
        """Get the exact column name with proper casing"""
        for col in data_columns:
            if col.lower() == column_lower.lower():
                return col
        return column_lower
    
    def suggest_analysis(self, df: pd.DataFrame) -> List[AnalysisIntent]:
        """
        Suggest possible analyses based on data characteristics
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of suggested analysis intents
        """
        suggestions = []
        columns = list(df.columns)
        
        # Basic descriptive analysis
        suggestions.append(AnalysisIntent(
            analysis_type=AnalysisType.DESCRIPTIVE,
            features=columns,
            parameters={'include_all': True},
            confidence=1.0,
            raw_prompt="Automatically suggested: Data overview"
        ))
        
        # Correlation analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            suggestions.append(AnalysisIntent(
                analysis_type=AnalysisType.CORRELATION,
                visualization_type=VisualizationType.HEATMAP,
                features=numeric_cols,
                confidence=0.9,
                raw_prompt="Automatically suggested: Correlation analysis"
            ))
        
        # Time series analysis if date columns exist
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols and numeric_cols:
            suggestions.append(AnalysisIntent(
                analysis_type=AnalysisType.TIME_SERIES,
                visualization_type=VisualizationType.TIME_SERIES_PLOT,
                features=date_cols + numeric_cols[:3],  # Limit to first 3 numeric
                confidence=0.8,
                raw_prompt="Automatically suggested: Time series analysis"
            ))
        
        # Distribution analysis for numeric columns
        if numeric_cols:
            suggestions.append(AnalysisIntent(
                analysis_type=AnalysisType.DISTRIBUTION,
                visualization_type=VisualizationType.HISTOGRAM,
                features=numeric_cols[:5],  # Limit to first 5
                confidence=0.7,
                raw_prompt="Automatically suggested: Distribution analysis"
            ))
        
        # Clustering if enough numeric features
        if len(numeric_cols) >= 2:
            suggestions.append(AnalysisIntent(
                analysis_type=AnalysisType.CLUSTERING,
                features=numeric_cols,
                parameters={'n_clusters': min(5, max(2, len(df) // 100))},
                confidence=0.6,
                raw_prompt="Automatically suggested: Clustering analysis"
            ))
        
        return suggestions
    
    def validate_intent(self, intent: AnalysisIntent, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that the analysis intent can be executed with the given data
        
        Args:
            intent: Analysis intent to validate
            df: DataFrame to validate against
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        columns = list(df.columns)
        
        # Check target column exists
        if intent.target_column and intent.target_column not in columns:
            issues.append(f"Target column '{intent.target_column}' not found in data")
        
        # Check feature columns exist
        if intent.features:
            missing_features = [f for f in intent.features if f not in columns]
            if missing_features:
                issues.append(f"Feature columns not found: {missing_features}")
        
        # Check data requirements for specific analysis types
        if intent.analysis_type == AnalysisType.CORRELATION:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                issues.append("Correlation analysis requires at least 2 numeric columns")
        
        elif intent.analysis_type == AnalysisType.TIME_SERIES:
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if not date_cols:
                issues.append("Time series analysis requires at least one date/time column")
        
        elif intent.analysis_type == AnalysisType.PREDICTIVE:
            if not intent.target_column:
                issues.append("Predictive analysis requires a target column")
            if not intent.features:
                issues.append("Predictive analysis requires feature columns")
        
        elif intent.analysis_type == AnalysisType.CLUSTERING:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                issues.append("Clustering analysis requires at least 2 numeric columns")
        
        return len(issues) == 0, issues
