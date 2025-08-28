"""
Natural Language Report Generator
Converts analysis results into human-readable insights and recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
import json
from jinja2 import Template

from analysis_engines import AnalysisResult
from visualization_generator import VisualizationResult
from prompt_parser import AnalysisType

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates human-readable reports from analysis results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_config = config.get('output', {})
        self.output_dir = Path(self.output_config.get('directory', './outputs'))
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(self, 
                       analysis_result: AnalysisResult,
                       visualizations: List[VisualizationResult],
                       df_summary: Dict[str, Any],
                       prompt: str = "") -> Dict[str, Any]:
        """
        Generate a comprehensive report from analysis results
        
        Args:
            analysis_result: Results from analysis engine
            visualizations: List of generated visualizations
            df_summary: Basic data summary
            prompt: Original user prompt
            
        Returns:
            Dictionary containing report content and metadata
        """
        
        try:
            # Generate different sections of the report
            executive_summary = self._generate_executive_summary(analysis_result, df_summary, prompt)
            detailed_findings = self._generate_detailed_findings(analysis_result)
            recommendations = self._generate_recommendations(analysis_result)
            methodology = self._generate_methodology_section(analysis_result)
            
            # Compile full report
            report = {
                'title': self._generate_title(analysis_result.analysis_type, prompt),
                'executive_summary': executive_summary,
                'detailed_findings': detailed_findings,
                'recommendations': recommendations,
                'methodology': methodology,
                'metadata': {
                    'analysis_type': analysis_result.analysis_type,
                    'generated_at': datetime.now().isoformat(),
                    'original_prompt': prompt,
                    'data_shape': df_summary.get('shape', 'Unknown'),
                    'success': analysis_result.success
                }
            }
            
            # Add visualizations info
            if visualizations:
                report['visualizations'] = [
                    {
                        'title': viz.title,
                        'type': viz.viz_type,
                        'description': self._get_visualization_description(viz)
                    }
                    for viz in visualizations
                ]
            
            # Generate formatted reports
            formatted_reports = self._create_formatted_reports(report, visualizations)
            report['formatted_reports'] = formatted_reports
            
            logger.info(f"Report generated successfully for {analysis_result.analysis_type} analysis")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {
                'title': 'Error in Report Generation',
                'executive_summary': f'An error occurred while generating the report: {str(e)}',
                'detailed_findings': [],
                'recommendations': [],
                'methodology': '',
                'metadata': {'success': False, 'error': str(e)}
            }
    
    def _generate_title(self, analysis_type: str, prompt: str) -> str:
        """Generate an appropriate title for the report"""
        
        titles = {
            'descriptive': 'Data Analysis Overview Report',
            'correlation': 'Correlation Analysis Report',
            'predictive': 'Predictive Modeling Report',
            'clustering': 'Clustering Analysis Report',
            'anomaly': 'Anomaly Detection Report'
        }
        
        base_title = titles.get(analysis_type, 'Data Analysis Report')
        
        if prompt and len(prompt) < 100:
            return f"{base_title}: {prompt.capitalize()}"
        else:
            return base_title
    
    def _generate_executive_summary(self, analysis_result: AnalysisResult, 
                                  df_summary: Dict[str, Any], prompt: str) -> str:
        """Generate executive summary"""
        
        try:
            summary_parts = []
            
            # Opening statement
            if prompt:
                summary_parts.append(f"This report presents the results of a {analysis_result.analysis_type} analysis "
                                   f"in response to the query: '{prompt}'")
            else:
                summary_parts.append(f"This report presents the results of a {analysis_result.analysis_type} analysis "
                                   f"of the provided dataset.")
            
            # Data overview
            shape = df_summary.get('shape', (0, 0))
            summary_parts.append(f"The analysis was performed on a dataset containing {shape[0]:,} observations "
                                f"across {shape[1]} variables.")
            
            # Key findings based on analysis type
            if analysis_result.analysis_type == 'descriptive':
                summary_parts.append(self._summarize_descriptive_findings(analysis_result))
            elif analysis_result.analysis_type == 'correlation':
                summary_parts.append(self._summarize_correlation_findings(analysis_result))
            elif analysis_result.analysis_type == 'predictive':
                summary_parts.append(self._summarize_predictive_findings(analysis_result))
            elif analysis_result.analysis_type == 'clustering':
                summary_parts.append(self._summarize_clustering_findings(analysis_result))
            elif analysis_result.analysis_type == 'anomaly':
                summary_parts.append(self._summarize_anomaly_findings(analysis_result))
            
            # Success/warning note
            if not analysis_result.success:
                summary_parts.append("⚠️ Note: The analysis encountered some issues, and results should be interpreted with caution.")
            elif analysis_result.warnings:
                summary_parts.append(f"⚠️ Note: {len(analysis_result.warnings)} warning(s) were encountered during analysis.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return f"Executive summary could not be generated due to an error: {str(e)}"
    
    def _generate_detailed_findings(self, analysis_result: AnalysisResult) -> List[Dict[str, Any]]:
        """Generate detailed findings section"""
        
        findings = []
        
        try:
            # Add key insights
            for insight in analysis_result.insights:
                findings.append({
                    'type': 'insight',
                    'title': 'Key Insight',
                    'content': insight,
                    'importance': 'high'
                })
            
            # Add analysis-specific findings
            if analysis_result.analysis_type == 'descriptive':
                findings.extend(self._get_descriptive_findings(analysis_result))
            elif analysis_result.analysis_type == 'correlation':
                findings.extend(self._get_correlation_findings(analysis_result))
            elif analysis_result.analysis_type == 'predictive':
                findings.extend(self._get_predictive_findings(analysis_result))
            elif analysis_result.analysis_type == 'clustering':
                findings.extend(self._get_clustering_findings(analysis_result))
            elif analysis_result.analysis_type == 'anomaly':
                findings.extend(self._get_anomaly_findings(analysis_result))
            
            # Add warnings
            for warning in analysis_result.warnings:
                findings.append({
                    'type': 'warning',
                    'title': 'Warning',
                    'content': warning,
                    'importance': 'medium'
                })
            
            return findings
            
        except Exception as e:
            logger.error(f"Error generating detailed findings: {str(e)}")
            return [{'type': 'error', 'title': 'Error', 'content': str(e), 'importance': 'high'}]
    
    def _generate_recommendations(self, analysis_result: AnalysisResult) -> List[Dict[str, str]]:
        """Generate recommendations based on analysis results"""
        
        recommendations = []
        
        try:
            # General recommendations based on analysis type
            if analysis_result.analysis_type == 'descriptive':
                recommendations.extend(self._get_descriptive_recommendations(analysis_result))
            elif analysis_result.analysis_type == 'correlation':
                recommendations.extend(self._get_correlation_recommendations(analysis_result))
            elif analysis_result.analysis_type == 'predictive':
                recommendations.extend(self._get_predictive_recommendations(analysis_result))
            elif analysis_result.analysis_type == 'clustering':
                recommendations.extend(self._get_clustering_recommendations(analysis_result))
            elif analysis_result.analysis_type == 'anomaly':
                recommendations.extend(self._get_anomaly_recommendations(analysis_result))
            
            # Add warning-based recommendations
            if analysis_result.warnings:
                recommendations.append({
                    'category': 'Data Quality',
                    'title': 'Address Data Quality Issues',
                    'description': 'Several warnings were encountered during analysis. Review data quality and '
                                 'consider data cleaning or preprocessing steps.',
                    'priority': 'high'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return [{'category': 'Error', 'title': 'Report Generation Issue',
                    'description': f'Could not generate recommendations: {str(e)}', 'priority': 'high'}]
    
    def _generate_methodology_section(self, analysis_result: AnalysisResult) -> str:
        """Generate methodology section"""
        
        methodologies = {
            'descriptive': """
            **Descriptive Analysis Methodology:**
            - Calculated summary statistics for all numeric variables
            - Analyzed missing data patterns and data types
            - Generated distribution plots and identified outliers using statistical measures
            - Computed data quality metrics including duplicates and memory usage
            """,
            'correlation': """
            **Correlation Analysis Methodology:**
            - Computed Pearson correlation coefficients between all numeric variables
            - Applied statistical significance testing (p-values) where applicable
            - Identified strong correlations above the threshold (default: 0.7)
            - Generated correlation matrix visualization and network diagrams for strong relationships
            """,
            'predictive': """
            **Predictive Modeling Methodology:**
            - Automatically detected task type (classification vs. regression) based on target variable characteristics
            - Applied standard preprocessing: missing value imputation and feature scaling
            - Used Random Forest as the base model for robust performance
            - Performed train/test split and cross-validation for model evaluation
            - Calculated feature importance scores to identify key predictors
            """,
            'clustering': """
            **Clustering Analysis Methodology:**
            - Applied K-means clustering algorithm with standardized features
            - Automatically determined optimal number of clusters based on data size
            - Used Principal Component Analysis (PCA) for dimensionality reduction and visualization
            - Calculated silhouette scores to measure cluster quality
            - Generated cluster statistics and comparative analysis
            """,
            'anomaly': """
            **Anomaly Detection Methodology:**
            - Applied isolation forest algorithm for unsupervised anomaly detection
            - Used statistical methods (Z-score) as alternative approach
            - Standardized features to ensure equal contribution to anomaly scoring
            - Identified top anomalies and analyzed feature contributions
            - Compared normal vs. anomalous data point characteristics
            """
        }
        
        return methodologies.get(analysis_result.analysis_type, 
                               "Methodology information not available for this analysis type.")
    
    # Analysis-specific summary methods
    def _summarize_descriptive_findings(self, analysis_result: AnalysisResult) -> str:
        """Summarize descriptive analysis findings"""
        
        results = analysis_result.results
        shape = results.get('shape', (0, 0))
        
        summary = f"The dataset analysis reveals {shape[1]} variables with varying characteristics."
        
        # Missing data summary
        missing_info = results.get('missing_values', {})
        if missing_info.get('count'):
            total_missing = sum(missing_info['count'].values())
            if total_missing > 0:
                summary += f" Missing data was identified in {total_missing:,} cells."
        
        return summary
    
    def _summarize_correlation_findings(self, analysis_result: AnalysisResult) -> str:
        """Summarize correlation analysis findings"""
        
        strong_corrs = analysis_result.results.get('strong_correlations', [])
        if strong_corrs:
            return f"The analysis identified {len(strong_corrs)} strong correlation relationships between variables."
        else:
            return "No strong correlations were found between the analyzed variables."
    
    def _summarize_predictive_findings(self, analysis_result: AnalysisResult) -> str:
        """Summarize predictive analysis findings"""
        
        results = analysis_result.results
        task_type = results.get('task_type', 'unknown')
        
        if task_type == 'classification':
            accuracy = results.get('accuracy', 0)
            return f"A {task_type} model was built with an accuracy of {accuracy:.1%}."
        elif task_type == 'regression':
            r2 = results.get('r2_score', 0)
            return f"A {task_type} model was built with an R² score of {r2:.3f}."
        else:
            return "A predictive model was successfully trained on the dataset."
    
    def _summarize_clustering_findings(self, analysis_result: AnalysisResult) -> str:
        """Summarize clustering analysis findings"""
        
        n_clusters = analysis_result.results.get('n_clusters', 0)
        silhouette = analysis_result.results.get('silhouette_score')
        
        summary = f"The data was segmented into {n_clusters} distinct clusters."
        if silhouette:
            if silhouette > 0.7:
                quality = "excellent"
            elif silhouette > 0.5:
                quality = "good"
            else:
                quality = "moderate"
            summary += f" The clustering quality is {quality} (silhouette score: {silhouette:.3f})."
        
        return summary
    
    def _summarize_anomaly_findings(self, analysis_result: AnalysisResult) -> str:
        """Summarize anomaly detection findings"""
        
        n_anomalies = analysis_result.results.get('n_anomalies', 0)
        anomaly_pct = analysis_result.results.get('anomaly_percentage', 0)
        
        if n_anomalies > 0:
            return f"The analysis detected {n_anomalies} anomalous data points ({anomaly_pct:.1f}% of the dataset)."
        else:
            return "No significant anomalies were detected in the dataset."
    
    # Detailed findings methods
    def _get_descriptive_findings(self, analysis_result: AnalysisResult) -> List[Dict[str, Any]]:
        """Get detailed descriptive findings"""
        findings = []
        results = analysis_result.results
        
        # Data composition
        if 'columns' in results:
            findings.append({
                'type': 'finding',
                'title': 'Dataset Composition',
                'content': f"Dataset contains {len(results['columns'])} variables: "
                          f"{results['columns'][:5]} {'...' if len(results['columns']) > 5 else ''}",
                'importance': 'medium'
            })
        
        # Missing data analysis
        missing_info = results.get('missing_values', {})
        if missing_info.get('percentage'):
            high_missing = {k: v for k, v in missing_info['percentage'].items() if v > 10}
            if high_missing:
                findings.append({
                    'type': 'finding',
                    'title': 'Missing Data Analysis',
                    'content': f"Variables with significant missing data (>10%): {list(high_missing.keys())}",
                    'importance': 'high'
                })
        
        return findings
    
    def _get_correlation_findings(self, analysis_result: AnalysisResult) -> List[Dict[str, Any]]:
        """Get detailed correlation findings"""
        findings = []
        strong_corrs = analysis_result.results.get('strong_correlations', [])
        
        for corr in strong_corrs[:5]:  # Top 5
            direction = "positively" if corr['correlation'] > 0 else "negatively"
            strength = "strongly" if abs(corr['correlation']) > 0.8 else "moderately"
            
            findings.append({
                'type': 'finding',
                'title': 'Strong Correlation',
                'content': f"{corr['variable1']} and {corr['variable2']} are {strength} {direction} "
                          f"correlated (r = {corr['correlation']:.3f})",
                'importance': 'high' if abs(corr['correlation']) > 0.8 else 'medium'
            })
        
        return findings
    
    def _get_predictive_findings(self, analysis_result: AnalysisResult) -> List[Dict[str, Any]]:
        """Get detailed predictive findings"""
        findings = []
        results = analysis_result.results
        
        # Feature importance
        if 'feature_importance' in results:
            top_features = list(results['feature_importance'].items())[:3]
            findings.append({
                'type': 'finding',
                'title': 'Most Important Features',
                'content': f"Top predictive features: {[f[0] for f in top_features]}",
                'importance': 'high'
            })
        
        # Model performance details
        if results.get('task_type') == 'classification' and 'cv_accuracy' in results:
            cv_acc = results['cv_accuracy']
            findings.append({
                'type': 'finding',
                'title': 'Cross-Validation Performance',
                'content': f"Model shows consistent performance with {cv_acc['mean']:.3f} ± {cv_acc['std']:.3f} accuracy across folds",
                'importance': 'medium'
            })
        elif results.get('task_type') == 'regression' and 'cv_r2' in results:
            cv_r2 = results['cv_r2']
            findings.append({
                'type': 'finding',
                'title': 'Cross-Validation Performance',
                'content': f"Model shows consistent performance with {cv_r2['mean']:.3f} ± {cv_r2['std']:.3f} R² score across folds",
                'importance': 'medium'
            })
        
        return findings
    
    def _get_clustering_findings(self, analysis_result: AnalysisResult) -> List[Dict[str, Any]]:
        """Get detailed clustering findings"""
        findings = []
        cluster_stats = analysis_result.results.get('cluster_statistics', {})
        
        # Cluster size distribution
        if cluster_stats:
            sizes = [stats['size'] for stats in cluster_stats.values()]
            min_size, max_size = min(sizes), max(sizes)
            
            if max_size / min_size > 5:
                findings.append({
                    'type': 'finding',
                    'title': 'Cluster Size Imbalance',
                    'content': f"Cluster sizes vary significantly: smallest={min_size}, largest={max_size}",
                    'importance': 'medium'
                })
        
        return findings
    
    def _get_anomaly_findings(self, analysis_result: AnalysisResult) -> List[Dict[str, Any]]:
        """Get detailed anomaly findings"""
        findings = []
        results = analysis_result.results
        
        # Feature contribution to anomalies
        if 'feature_contribution' in results:
            top_feature = list(results['feature_contribution'].items())[0]
            findings.append({
                'type': 'finding',
                'title': 'Primary Anomaly Driver',
                'content': f"'{top_feature[0]}' contributes most to anomalous patterns",
                'importance': 'high'
            })
        
        return findings
    
    # Recommendation methods
    def _get_descriptive_recommendations(self, analysis_result: AnalysisResult) -> List[Dict[str, str]]:
        """Get descriptive analysis recommendations"""
        recommendations = []
        
        # Based on missing data
        missing_info = analysis_result.results.get('missing_values', {})
        if missing_info.get('percentage'):
            high_missing = {k: v for k, v in missing_info['percentage'].items() if v > 50}
            if high_missing:
                recommendations.append({
                    'category': 'Data Quality',
                    'title': 'Address High Missing Data',
                    'description': f'Consider removing or imputing variables with >50% missing data: {list(high_missing.keys())}',
                    'priority': 'high'
                })
        
        recommendations.append({
            'category': 'Next Steps',
            'title': 'Further Analysis',
            'description': 'Consider correlation analysis to understand relationships between variables, or predictive modeling for specific outcomes.',
            'priority': 'medium'
        })
        
        return recommendations
    
    def _get_correlation_recommendations(self, analysis_result: AnalysisResult) -> List[Dict[str, str]]:
        """Get correlation analysis recommendations"""
        recommendations = []
        
        strong_corrs = analysis_result.results.get('strong_correlations', [])
        if strong_corrs:
            perfect_corrs = [c for c in strong_corrs if abs(c['correlation']) > 0.95]
            if perfect_corrs:
                recommendations.append({
                    'category': 'Feature Engineering',
                    'title': 'Address Multicollinearity',
                    'description': 'Consider removing one variable from highly correlated pairs to avoid multicollinearity in modeling.',
                    'priority': 'high'
                })
        
        return recommendations
    
    def _get_predictive_recommendations(self, analysis_result: AnalysisResult) -> List[Dict[str, str]]:
        """Get predictive analysis recommendations"""
        recommendations = []
        
        results = analysis_result.results
        if results.get('task_type') == 'classification':
            accuracy = results.get('accuracy', 0)
            if accuracy < 0.7:
                recommendations.append({
                    'category': 'Model Improvement',
                    'title': 'Improve Model Performance',
                    'description': 'Consider feature engineering, hyperparameter tuning, or trying different algorithms to improve accuracy.',
                    'priority': 'high'
                })
        
        return recommendations
    
    def _get_clustering_recommendations(self, analysis_result: AnalysisResult) -> List[Dict[str, str]]:
        """Get clustering analysis recommendations"""
        recommendations = []
        
        silhouette = analysis_result.results.get('silhouette_score')
        if silhouette and silhouette < 0.5:
            recommendations.append({
                'category': 'Clustering Quality',
                'title': 'Optimize Cluster Number',
                'description': 'Consider trying different numbers of clusters or clustering algorithms to improve separation.',
                'priority': 'medium'
            })
        
        return recommendations
    
    def _get_anomaly_recommendations(self, analysis_result: AnalysisResult) -> List[Dict[str, str]]:
        """Get anomaly detection recommendations"""
        recommendations = []
        
        n_anomalies = analysis_result.results.get('n_anomalies', 0)
        if n_anomalies > 0:
            recommendations.append({
                'category': 'Data Investigation',
                'title': 'Investigate Anomalies',
                'description': 'Review identified anomalies to determine if they represent errors, outliers, or genuine unusual patterns.',
                'priority': 'high'
            })
        
        return recommendations
    
    def _get_visualization_description(self, viz: VisualizationResult) -> str:
        """Get description for visualization"""
        
        descriptions = {
            'overview_dashboard': 'Comprehensive dashboard showing dataset composition, missing values, and key statistics',
            'correlation_heatmap': 'Interactive heatmap showing correlation strengths between all numeric variables',
            'cluster_scatter': 'Scatter plot visualization of clusters using principal component analysis',
            'missing_values_heatmap': 'Pattern visualization of missing data across variables and observations'
        }
        
        return descriptions.get(viz.viz_type, f'{viz.viz_type} visualization')
    
    def _create_formatted_reports(self, report: Dict[str, Any], 
                                visualizations: List[VisualizationResult]) -> Dict[str, str]:
        """Create formatted report outputs"""
        
        formatted_reports = {}
        
        try:
            # HTML Report
            html_content = self._generate_html_report(report, visualizations)
            html_path = self.output_dir / 'analysis_report.html'
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            formatted_reports['html'] = str(html_path)
            
            # Markdown Report
            md_content = self._generate_markdown_report(report)
            md_path = self.output_dir / 'analysis_report.md'
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            formatted_reports['markdown'] = str(md_path)
            
            # JSON Report (structured data)
            json_path = self.output_dir / 'analysis_report.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            formatted_reports['json'] = str(json_path)
            
        except Exception as e:
            logger.error(f"Error creating formatted reports: {str(e)}")
        
        return formatted_reports
    
    def _generate_html_report(self, report: Dict[str, Any], 
                            visualizations: List[VisualizationResult]) -> str:
        """Generate HTML report"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
                .section { margin-bottom: 30px; }
                .finding { background-color: #e8f4fd; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; }
                .warning { background-color: #fff3cd; padding: 15px; margin: 10px 0; border-left: 4px solid #ffc107; }
                .recommendation { background-color: #d4edda; padding: 15px; margin: 10px 0; border-left: 4px solid #28a745; }
                .high-priority { border-left-color: #dc3545 !important; }
                .metadata { color: #666; font-size: 0.9em; }
                h1, h2 { color: #333; }
                pre { background-color: #f8f9fa; padding: 15px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <div class="metadata">
                    Generated: {{ metadata.generated_at }}<br>
                    Analysis Type: {{ metadata.analysis_type }}<br>
                    Data Shape: {{ metadata.data_shape }}
                </div>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{{ executive_summary }}</p>
            </div>
            
            <div class="section">
                <h2>Detailed Findings</h2>
                {% for finding in detailed_findings %}
                    <div class="{{ finding.type }} {% if finding.importance == 'high' %}high-priority{% endif %}">
                        <strong>{{ finding.title }}:</strong> {{ finding.content }}
                    </div>
                {% endfor %}
            </div>
            
            {% if recommendations %}
            <div class="section">
                <h2>Recommendations</h2>
                {% for rec in recommendations %}
                    <div class="recommendation {% if rec.priority == 'high' %}high-priority{% endif %}">
                        <strong>{{ rec.title }}</strong> ({{ rec.category }})<br>
                        {{ rec.description }}
                    </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if visualizations %}
            <div class="section">
                <h2>Visualizations</h2>
                {% for viz in visualizations %}
                    <h3>{{ viz.title }}</h3>
                    <p>{{ viz.description }}</p>
                {% endfor %}
            </div>
            {% endif %}
            
            <div class="section">
                <h2>Methodology</h2>
                <pre>{{ methodology }}</pre>
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        return template.render(**report)
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate Markdown report"""
        
        md_content = f"""# {report['title']}

*Generated: {report['metadata'].get('generated_at', 'Unknown')}*  
*Analysis Type: {report['metadata'].get('analysis_type', 'Unknown')}*  
*Data Shape: {report['metadata'].get('data_shape', 'Unknown')}*

## Executive Summary

{report['executive_summary']}

## Detailed Findings

"""
        
        for finding in report['detailed_findings']:
            priority_marker = "🔴" if finding.get('importance') == 'high' else "🟡" if finding.get('importance') == 'medium' else "🟢"
            md_content += f"{priority_marker} **{finding['title']}**: {finding['content']}\n\n"
        
        if report['recommendations']:
            md_content += "## Recommendations\n\n"
            for rec in report['recommendations']:
                priority_marker = "🔴" if rec.get('priority') == 'high' else "🟡" if rec.get('priority') == 'medium' else "🟢"
                md_content += f"{priority_marker} **{rec['title']}** ({rec['category']})\n{rec['description']}\n\n"
        
        if report.get('visualizations'):
            md_content += "## Visualizations\n\n"
            for viz in report['visualizations']:
                md_content += f"### {viz['title']}\n{viz['description']}\n\n"
        
        md_content += f"## Methodology\n\n{report['methodology']}\n"
        
        return md_content
