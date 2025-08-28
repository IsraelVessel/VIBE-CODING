"""
Visualization Generator
Creates automated charts and graphs based on analysis type and data characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import base64
import io
from abc import ABC, abstractmethod

from prompt_parser import VisualizationType, AnalysisType
from analysis_engines import AnalysisResult

logger = logging.getLogger(__name__)

class VisualizationResult:
    """Container for visualization results"""
    def __init__(self, 
                 viz_type: str,
                 title: str,
                 figure_path: Optional[str] = None,
                 html_content: Optional[str] = None,
                 base64_image: Optional[str] = None,
                 metadata: Dict[str, Any] = None):
        self.viz_type = viz_type
        self.title = title
        self.figure_path = figure_path
        self.html_content = html_content
        self.base64_image = base64_image
        self.metadata = metadata or {}

class BaseVisualizer(ABC):
    """Base class for all visualizers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.output_dir = Path(config.get('output', {}).get('directory', './outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use(self.viz_config.get('style', 'default'))
        sns.set_palette(self.viz_config.get('palette', 'viridis'))
    
    @abstractmethod
    def create_visualization(self, df: pd.DataFrame, analysis_result: AnalysisResult, 
                           **kwargs) -> VisualizationResult:
        """Create the visualization"""
        pass
    
    def _save_matplotlib_figure(self, fig, filename: str) -> Tuple[str, str]:
        """Save matplotlib figure and return path and base64 string"""
        # Save to file
        filepath = self.output_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=self.viz_config.get('dpi', 300), bbox_inches='tight')
        
        # Convert to base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.viz_config.get('dpi', 300), 
                   bbox_inches='tight')
        buffer.seek(0)
        base64_string = base64.b64encode(buffer.read()).decode()
        buffer.close()
        
        plt.close(fig)
        return str(filepath), base64_string
    
    def _save_plotly_figure(self, fig, filename: str) -> Tuple[str, str]:
        """Save plotly figure and return path and HTML content"""
        # Save as HTML
        filepath = self.output_dir / f"{filename}.html"
        fig.write_html(str(filepath))
        
        # Get HTML content
        html_content = fig.to_html(include_plotlyjs='cdn')
        
        return str(filepath), html_content

class DescriptiveVisualizer(BaseVisualizer):
    """Creates visualizations for descriptive analysis"""
    
    def create_visualization(self, df: pd.DataFrame, analysis_result: AnalysisResult, 
                           **kwargs) -> List[VisualizationResult]:
        """Create descriptive visualizations"""
        results = []
        
        try:
            # Overview dashboard
            fig = self._create_overview_dashboard(df, analysis_result)
            if fig:
                filepath, base64_img = self._save_matplotlib_figure(fig, 'descriptive_overview')
                results.append(VisualizationResult(
                    viz_type="overview_dashboard",
                    title="Data Overview Dashboard",
                    figure_path=filepath,
                    base64_image=base64_img,
                    metadata={'type': 'matplotlib'}
                ))
            
            # Missing values heatmap if there are missing values
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                fig = self._create_missing_values_heatmap(df)
                if fig:
                    filepath, base64_img = self._save_matplotlib_figure(fig, 'missing_values_heatmap')
                    results.append(VisualizationResult(
                        viz_type="missing_values_heatmap",
                        title="Missing Values Pattern",
                        figure_path=filepath,
                        base64_image=base64_img,
                        metadata={'type': 'matplotlib'}
                    ))
            
            # Distribution plots for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = self._create_distribution_plots(df[numeric_cols])
                if fig:
                    filepath, base64_img = self._save_matplotlib_figure(fig, 'distributions')
                    results.append(VisualizationResult(
                        viz_type="distribution_plots",
                        title="Variable Distributions",
                        figure_path=filepath,
                        base64_image=base64_img,
                        metadata={'type': 'matplotlib'}
                    ))
            
        except Exception as e:
            logger.error(f"Error creating descriptive visualizations: {str(e)}")
        
        return results
    
    def _create_overview_dashboard(self, df: pd.DataFrame, analysis_result: AnalysisResult):
        """Create an overview dashboard"""
        try:
            figsize = self.viz_config.get('figsize', [15, 10])
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            fig.suptitle('Data Overview Dashboard', fontsize=16, fontweight='bold')
            
            # Data types pie chart
            dtype_counts = df.dtypes.value_counts()
            axes[0, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Data Types Distribution')
            
            # Missing values bar chart
            missing_counts = df.isnull().sum().sort_values(ascending=False)
            if missing_counts.sum() > 0:
                top_missing = missing_counts.head(10)
                axes[0, 1].bar(range(len(top_missing)), top_missing.values)
                axes[0, 1].set_xticks(range(len(top_missing)))
                axes[0, 1].set_xticklabels(top_missing.index, rotation=45, ha='right')
                axes[0, 1].set_title('Missing Values by Column')
                axes[0, 1].set_ylabel('Missing Count')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                               transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title('Missing Values')
            
            # Memory usage
            memory_usage = df.memory_usage(deep=True).sort_values(ascending=False)
            top_memory = memory_usage.head(10)
            axes[0, 2].bar(range(len(top_memory)), top_memory.values / 1024)  # KB
            axes[0, 2].set_xticks(range(len(top_memory)))
            axes[0, 2].set_xticklabels(top_memory.index, rotation=45, ha='right')
            axes[0, 2].set_title('Memory Usage by Column (KB)')
            axes[0, 2].set_ylabel('Memory (KB)')
            
            # Numeric columns summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                axes[1, 0].bar(['Count', 'Mean', 'Std', 'Min', 'Max'], 
                              [len(numeric_cols), 
                               df[numeric_cols].mean().mean(),
                               df[numeric_cols].std().mean(),
                               df[numeric_cols].min().min(),
                               df[numeric_cols].max().max()])
                axes[1, 0].set_title(f'Numeric Summary ({len(numeric_cols)} columns)')
                axes[1, 0].tick_params(axis='x', rotation=45)
            else:
                axes[1, 0].text(0.5, 0.5, 'No Numeric Columns', ha='center', va='center',
                               transform=axes[1, 0].transAxes, fontsize=12)
                axes[1, 0].set_title('Numeric Summary')
            
            # Categorical columns summary
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                unique_counts = [df[col].nunique() for col in cat_cols[:10]]
                axes[1, 1].bar(range(len(unique_counts)), unique_counts)
                axes[1, 1].set_xticks(range(len(unique_counts)))
                axes[1, 1].set_xticklabels(cat_cols[:10], rotation=45, ha='right')
                axes[1, 1].set_title(f'Unique Values per Category ({len(cat_cols)} columns)')
                axes[1, 1].set_ylabel('Unique Count')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Categorical Columns', ha='center', va='center',
                               transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('Categorical Summary')
            
            # Data shape and info
            info_text = f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
            info_text += f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n"
            info_text += f"Duplicates: {df.duplicated().sum():,}\n"
            info_text += f"Numeric: {len(numeric_cols)} columns\n"
            info_text += f"Categorical: {len(cat_cols)} columns"
            
            axes[1, 2].text(0.1, 0.9, info_text, transform=axes[1, 2].transAxes, 
                           fontsize=10, va='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
            axes[1, 2].set_title('Dataset Information')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating overview dashboard: {str(e)}")
            return None
    
    def _create_missing_values_heatmap(self, df: pd.DataFrame):
        """Create missing values heatmap"""
        try:
            missing_data = df.isnull()
            if missing_data.sum().sum() == 0:
                return None
            
            # Sample data if too large
            if len(df) > 1000:
                sample_df = df.sample(1000, random_state=42)
                missing_data = sample_df.isnull()
            
            figsize = self.viz_config.get('figsize', [12, 8])
            plt.figure(figsize=figsize)
            
            # Create heatmap
            sns.heatmap(missing_data.T, cbar=True, cmap='viridis_r', 
                       yticklabels=True, xticklabels=False)
            plt.title('Missing Values Pattern\n(Yellow = Missing, Purple = Present)')
            plt.ylabel('Columns')
            plt.xlabel('Observations')
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error creating missing values heatmap: {str(e)}")
            return None
    
    def _create_distribution_plots(self, numeric_df: pd.DataFrame):
        """Create distribution plots for numeric columns"""
        try:
            n_cols = min(len(numeric_df.columns), 12)  # Limit to 12 columns
            n_rows = (n_cols + 2) // 3  # 3 columns per row
            
            figsize = self.viz_config.get('figsize', [15, 4 * n_rows])
            fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle('Distribution of Numeric Variables', fontsize=16, fontweight='bold')
            
            for i, col in enumerate(numeric_df.columns[:n_cols]):
                row, col_idx = divmod(i, 3)
                ax = axes[row, col_idx]
                
                # Create histogram with KDE
                data = numeric_df[col].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=30, alpha=0.7, color='skyblue', density=True)
                    
                    # Add KDE if enough data points
                    if len(data) > 10:
                        try:
                            sns.kdeplot(data=data, ax=ax, color='red', linewidth=2)
                        except:
                            pass
                    
                    ax.set_title(f'{col}')
                    ax.set_ylabel('Density')
                    
                    # Add statistics text
                    stats_text = f'μ={data.mean():.2f}\nσ={data.std():.2f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(f'{col} (No Data)')
            
            # Hide unused subplots
            for i in range(n_cols, n_rows * 3):
                row, col_idx = divmod(i, 3)
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating distribution plots: {str(e)}")
            return None

class CorrelationVisualizer(BaseVisualizer):
    """Creates correlation visualizations"""
    
    def create_visualization(self, df: pd.DataFrame, analysis_result: AnalysisResult, 
                           **kwargs) -> List[VisualizationResult]:
        """Create correlation visualizations"""
        results = []
        
        try:
            # Correlation heatmap
            fig = self._create_correlation_heatmap(analysis_result)
            if fig:
                filepath, html_content = self._save_plotly_figure(fig, 'correlation_heatmap')
                results.append(VisualizationResult(
                    viz_type="correlation_heatmap",
                    title="Correlation Heatmap",
                    figure_path=filepath,
                    html_content=html_content,
                    metadata={'type': 'plotly'}
                ))
            
            # Strong correlations network (if any)
            strong_corrs = analysis_result.results.get('strong_correlations', [])
            if strong_corrs:
                fig = self._create_correlation_network(strong_corrs)
                if fig:
                    filepath, html_content = self._save_plotly_figure(fig, 'correlation_network')
                    results.append(VisualizationResult(
                        viz_type="correlation_network",
                        title="Strong Correlations Network",
                        figure_path=filepath,
                        html_content=html_content,
                        metadata={'type': 'plotly'}
                    ))
            
        except Exception as e:
            logger.error(f"Error creating correlation visualizations: {str(e)}")
        
        return results
    
    def _create_correlation_heatmap(self, analysis_result: AnalysisResult):
        """Create interactive correlation heatmap"""
        try:
            corr_matrix = analysis_result.results.get('correlation_matrix', {})
            if not corr_matrix:
                return None
            
            corr_df = pd.DataFrame(corr_matrix)
            
            # Create interactive heatmap
            fig = px.imshow(corr_df,
                          labels=dict(x="Variables", y="Variables", color="Correlation"),
                          x=corr_df.columns,
                          y=corr_df.index,
                          color_continuous_scale='RdBu_r',
                          aspect="auto",
                          title="Correlation Matrix Heatmap")
            
            fig.update_layout(
                title_x=0.5,
                width=800,
                height=600,
                xaxis_title="Variables",
                yaxis_title="Variables"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return None
    
    def _create_correlation_network(self, strong_correlations: List[Dict]):
        """Create network visualization of strong correlations"""
        try:
            # Extract nodes and edges
            nodes = set()
            edges = []
            
            for corr in strong_correlations:
                var1, var2 = corr['variable1'], corr['variable2']
                correlation = corr['correlation']
                
                nodes.add(var1)
                nodes.add(var2)
                edges.append((var1, var2, correlation))
            
            # Create network layout (simple circular layout)
            import math
            n_nodes = len(nodes)
            node_positions = {}
            
            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / n_nodes
                x = math.cos(angle)
                y = math.sin(angle)
                node_positions[node] = (x, y)
            
            # Create traces for edges
            edge_traces = []
            for var1, var2, corr in edges:
                x0, y0 = node_positions[var1]
                x1, y1 = node_positions[var2]
                
                # Color based on correlation strength
                color = 'red' if corr > 0 else 'blue'
                width = min(abs(corr) * 10, 8)
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color=color),
                    hoverinfo='none',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
            
            # Create trace for nodes
            node_x = [node_positions[node][0] for node in nodes]
            node_y = [node_positions[node][1] for node in nodes]
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=list(nodes),
                textposition='middle center',
                marker=dict(size=30, color='lightblue', line=dict(width=2)),
                hoverinfo='text',
                hovertext=list(nodes),
                showlegend=False
            )
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            for edge_trace in edge_traces:
                fig.add_trace(edge_trace)
            
            # Add nodes
            fig.add_trace(node_trace)
            
            fig.update_layout(
                title="Strong Correlations Network<br><sub>Red=Positive, Blue=Negative, Width=Strength</sub>",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="Red lines: Positive correlations<br>Blue lines: Negative correlations<br>Line width: Correlation strength",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(size=10)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=800,
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation network: {str(e)}")
            return None

class ClusteringVisualizer(BaseVisualizer):
    """Creates clustering visualizations"""
    
    def create_visualization(self, df: pd.DataFrame, analysis_result: AnalysisResult, 
                           **kwargs) -> List[VisualizationResult]:
        """Create clustering visualizations"""
        results = []
        
        try:
            # Cluster scatter plot (2D using PCA)
            if 'pca_coordinates' in analysis_result.results:
                fig = self._create_cluster_scatter_plot(analysis_result)
                if fig:
                    filepath, html_content = self._save_plotly_figure(fig, 'cluster_scatter')
                    results.append(VisualizationResult(
                        viz_type="cluster_scatter",
                        title="Clusters Visualization (PCA)",
                        figure_path=filepath,
                        html_content=html_content,
                        metadata={'type': 'plotly'}
                    ))
            
            # Cluster statistics comparison
            fig = self._create_cluster_statistics(analysis_result)
            if fig:
                filepath, base64_img = self._save_matplotlib_figure(fig, 'cluster_statistics')
                results.append(VisualizationResult(
                    viz_type="cluster_statistics",
                    title="Cluster Statistics Comparison",
                    figure_path=filepath,
                    base64_image=base64_img,
                    metadata={'type': 'matplotlib'}
                ))
            
        except Exception as e:
            logger.error(f"Error creating clustering visualizations: {str(e)}")
        
        return results
    
    def _create_cluster_scatter_plot(self, analysis_result: AnalysisResult):
        """Create scatter plot of clusters using PCA coordinates"""
        try:
            pca_coords = analysis_result.results.get('pca_coordinates', {})
            cluster_labels = analysis_result.results.get('cluster_labels', [])
            
            if not pca_coords or not cluster_labels:
                return None
            
            # Create DataFrame
            df_plot = pd.DataFrame({
                'PC1': pca_coords['x'],
                'PC2': pca_coords['y'],
                'Cluster': [f'Cluster {c}' for c in cluster_labels]
            })
            
            # Create scatter plot
            fig = px.scatter(df_plot, x='PC1', y='PC2', color='Cluster',
                           title="Clusters Visualization (Principal Components)",
                           labels={'PC1': f'PC1 ({pca_coords["explained_variance_ratio"][0]:.1%} variance)',
                                  'PC2': f'PC2 ({pca_coords["explained_variance_ratio"][1]:.1%} variance)'})
            
            fig.update_layout(
                title_x=0.5,
                width=800,
                height=600,
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cluster scatter plot: {str(e)}")
            return None
    
    def _create_cluster_statistics(self, analysis_result: AnalysisResult):
        """Create cluster statistics comparison"""
        try:
            cluster_stats = analysis_result.results.get('cluster_statistics', {})
            if not cluster_stats:
                return None
            
            # Get cluster sizes
            cluster_names = list(cluster_stats.keys())
            cluster_sizes = [cluster_stats[name]['size'] for name in cluster_names]
            cluster_percentages = [cluster_stats[name]['percentage'] for name in cluster_names]
            
            figsize = self.viz_config.get('figsize', [12, 8])
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle('Cluster Analysis Summary', fontsize=16, fontweight='bold')
            
            # Cluster sizes bar chart
            ax1.bar(range(len(cluster_sizes)), cluster_sizes)
            ax1.set_xticks(range(len(cluster_names)))
            ax1.set_xticklabels([f'C{i}' for i in range(len(cluster_names))])
            ax1.set_title('Cluster Sizes')
            ax1.set_ylabel('Number of Points')
            
            # Add value labels on bars
            for i, v in enumerate(cluster_sizes):
                ax1.text(i, v + max(cluster_sizes) * 0.01, str(v), ha='center', va='bottom')
            
            # Cluster percentages pie chart
            ax2.pie(cluster_percentages, labels=[f'Cluster {i}' for i in range(len(cluster_names))], 
                   autopct='%1.1f%%', startangle=90)
            ax2.set_title('Cluster Distribution')
            
            # Feature means comparison (first few features)
            if cluster_names:
                first_cluster = cluster_stats[cluster_names[0]]
                feature_names = list(first_cluster['mean_values'].keys())[:5]  # First 5 features
                
                if feature_names:
                    cluster_means = {}
                    for cluster_name in cluster_names:
                        means = cluster_stats[cluster_name]['mean_values']
                        cluster_means[cluster_name] = [means.get(feat, 0) for feat in feature_names]
                    
                    x = np.arange(len(feature_names))
                    width = 0.8 / len(cluster_names)
                    
                    for i, (cluster_name, means) in enumerate(cluster_means.items()):
                        ax3.bar(x + i * width, means, width, label=f'C{i}', alpha=0.8)
                    
                    ax3.set_xticks(x + width * (len(cluster_names) - 1) / 2)
                    ax3.set_xticklabels(feature_names, rotation=45, ha='right')
                    ax3.set_title('Feature Means by Cluster')
                    ax3.set_ylabel('Mean Value')
                    ax3.legend()
                else:
                    ax3.text(0.5, 0.5, 'No Feature Data', ha='center', va='center', 
                            transform=ax3.transAxes)
                    ax3.set_title('Feature Means by Cluster')
            
            # Summary statistics
            total_points = sum(cluster_sizes)
            n_clusters = len(cluster_names)
            avg_cluster_size = total_points / n_clusters
            min_cluster_size = min(cluster_sizes)
            max_cluster_size = max(cluster_sizes)
            
            summary_text = f"Total Points: {total_points:,}\n"
            summary_text += f"Number of Clusters: {n_clusters}\n"
            summary_text += f"Average Cluster Size: {avg_cluster_size:.1f}\n"
            summary_text += f"Smallest Cluster: {min_cluster_size}\n"
            summary_text += f"Largest Cluster: {max_cluster_size}\n"
            
            # Add silhouette score if available
            silhouette_score = analysis_result.results.get('silhouette_score')
            if silhouette_score is not None:
                summary_text += f"Silhouette Score: {silhouette_score:.3f}"
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                    fontsize=12, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
            ax4.set_title('Clustering Summary')
            ax4.axis('off')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cluster statistics: {str(e)}")
            return None

# Visualization Factory
class VisualizationFactory:
    """Factory for creating visualizations"""
    
    @staticmethod
    def create_visualizer(analysis_type: str, config: Dict[str, Any]) -> BaseVisualizer:
        """Create appropriate visualizer based on analysis type"""
        
        visualizers = {
            'descriptive': DescriptiveVisualizer,
            'correlation': CorrelationVisualizer,
            'clustering': ClusteringVisualizer,
        }
        
        # Default to descriptive for unsupported types
        visualizer_class = visualizers.get(analysis_type, DescriptiveVisualizer)
        return visualizer_class(config)
    
    @staticmethod
    def create_visualization(analysis_result: AnalysisResult, df: pd.DataFrame, 
                           config: Dict[str, Any]) -> List[VisualizationResult]:
        """Create appropriate visualizations for analysis result"""
        
        try:
            visualizer = VisualizationFactory.create_visualizer(analysis_result.analysis_type, config)
            return visualizer.create_visualization(df, analysis_result)
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return []
