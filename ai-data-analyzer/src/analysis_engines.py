"""
Core Analysis Engines
Implements different analysis modules: descriptive statistics, trend analysis, 
anomaly detection, predictive modeling, clustering, etc.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    analysis_type: str
    results: Dict[str, Any]
    insights: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    success: bool = True

class BaseAnalyzer(ABC):
    """Base class for all analysis engines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analysis_config = config.get('analysis', {})
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame, **kwargs) -> AnalysisResult:
        """Perform the analysis"""
        pass
    
    def _validate_data(self, df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> List[str]:
        """Validate data requirements"""
        issues = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return issues
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                issues.append(f"Required columns missing: {missing_cols}")
        
        return issues

class DescriptiveAnalyzer(BaseAnalyzer):
    """Performs descriptive statistical analysis"""
    
    def analyze(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> AnalysisResult:
        """Generate descriptive statistics"""
        
        try:
            issues = self._validate_data(df)
            if issues:
                return AnalysisResult("descriptive", {}, [], issues, {}, False)
            
            if columns:
                df = df[columns]
            
            results = {}
            insights = []
            warnings_list = []
            
            # Basic info
            results['shape'] = df.shape
            results['columns'] = list(df.columns)
            results['dtypes'] = df.dtypes.to_dict()
            
            # Missing values analysis
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df) * 100).round(2)
            results['missing_values'] = {
                'count': missing_data.to_dict(),
                'percentage': missing_pct.to_dict()
            }
            
            # Insights about missing data
            high_missing = missing_pct[missing_pct > 50]
            if len(high_missing) > 0:
                insights.append(f"Columns with >50% missing data: {list(high_missing.index)}")
                warnings_list.append("High missing data detected in some columns")
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                numeric_stats = df[numeric_cols].describe()
                results['numeric_summary'] = numeric_stats.to_dict()
                
                # Additional numeric insights
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        skewness = stats.skew(col_data)
                        kurtosis = stats.kurtosis(col_data)
                        
                        if abs(skewness) > 1:
                            insights.append(f"Column '{col}' is highly skewed (skewness: {skewness:.2f})")
                        
                        if kurtosis > 3:
                            insights.append(f"Column '{col}' has heavy tails (kurtosis: {kurtosis:.2f})")
            
            # Categorical columns analysis
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                cat_summary = {}
                for col in categorical_cols:
                    value_counts = df[col].value_counts().head(10)
                    unique_count = df[col].nunique()
                    cat_summary[col] = {
                        'unique_count': unique_count,
                        'top_values': value_counts.to_dict()
                    }
                    
                    # Insights about categorical data
                    if unique_count == 1:
                        warnings_list.append(f"Column '{col}' has only one unique value")
                    elif unique_count == len(df):
                        insights.append(f"Column '{col}' appears to be a unique identifier")
                
                results['categorical_summary'] = cat_summary
            
            # Date columns analysis
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_summary = {}
                for col in date_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        date_summary[col] = {
                            'min_date': col_data.min(),
                            'max_date': col_data.max(),
                            'date_range': (col_data.max() - col_data.min()).days
                        }
                
                results['date_summary'] = date_summary
            
            # Memory usage
            results['memory_usage'] = df.memory_usage(deep=True).sum()
            
            # Generate general insights
            insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
            insights.append(f"Numeric columns: {len(numeric_cols)}, Categorical: {len(categorical_cols)}, Date: {len(date_cols)}")
            
            if df.duplicated().sum() > 0:
                dup_count = df.duplicated().sum()
                insights.append(f"Found {dup_count} duplicate rows ({dup_count/len(df)*100:.1f}%)")
            
            metadata = {
                'analysis_timestamp': pd.Timestamp.now(),
                'rows_analyzed': len(df),
                'columns_analyzed': len(df.columns)
            }
            
            return AnalysisResult(
                analysis_type="descriptive",
                results=results,
                insights=insights,
                warnings=warnings_list,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in descriptive analysis: {str(e)}")
            return AnalysisResult("descriptive", {}, [], [f"Analysis failed: {str(e)}"], {}, False)

class CorrelationAnalyzer(BaseAnalyzer):
    """Performs correlation analysis"""
    
    def analyze(self, df: pd.DataFrame, columns: Optional[List[str]] = None, 
                method: str = 'pearson') -> AnalysisResult:
        """Analyze correlations between variables"""
        
        try:
            # Get numeric columns
            if columns:
                numeric_df = df[columns].select_dtypes(include=[np.number])
            else:
                numeric_df = df.select_dtypes(include=[np.number])
            
            issues = self._validate_data(numeric_df)
            if len(numeric_df.columns) < 2:
                issues.append("Need at least 2 numeric columns for correlation analysis")
            
            if issues:
                return AnalysisResult("correlation", {}, [], issues, {}, False)
            
            results = {}
            insights = []
            warnings_list = []
            
            # Calculate correlation matrix
            if method == 'pearson':
                corr_matrix = numeric_df.corr(method='pearson')
            elif method == 'spearman':
                corr_matrix = numeric_df.corr(method='spearman')
            else:
                corr_matrix = numeric_df.corr(method='pearson')
                warnings_list.append(f"Unknown method '{method}', using Pearson")
            
            results['correlation_matrix'] = corr_matrix.to_dict()
            
            # Find strong correlations
            threshold = self.analysis_config.get('correlation_threshold', 0.7)
            strong_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        strong_correlations.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            results['strong_correlations'] = strong_correlations
            
            # Generate insights
            if strong_correlations:
                insights.append(f"Found {len(strong_correlations)} strong correlations (|r| > {threshold})")
                for corr in strong_correlations[:5]:  # Show top 5
                    insights.append(f"{corr['variable1']} ↔ {corr['variable2']}: {corr['correlation']:.3f}")
            else:
                insights.append(f"No strong correlations found (threshold: {threshold})")
            
            # Check for perfect correlations (potential multicollinearity)
            perfect_corrs = [c for c in strong_correlations if abs(c['correlation']) > 0.95]
            if perfect_corrs:
                warnings_list.append("Found near-perfect correlations - potential multicollinearity issues")
            
            # Statistical significance testing
            if method == 'pearson':
                p_values = {}
                for col1 in numeric_df.columns:
                    for col2 in numeric_df.columns:
                        if col1 != col2:
                            try:
                                _, p_val = pearsonr(numeric_df[col1].dropna(), numeric_df[col2].dropna())
                                p_values[f"{col1}_vs_{col2}"] = p_val
                            except:
                                pass
                
                results['p_values'] = p_values
                
                # Count significant correlations
                alpha = self.analysis_config.get('alpha', 0.05)
                significant_corrs = sum(1 for p in p_values.values() if p < alpha)
                insights.append(f"Significant correlations at α={alpha}: {significant_corrs}/{len(p_values)}")
            
            metadata = {
                'method': method,
                'variables_analyzed': len(numeric_df.columns),
                'threshold_used': threshold
            }
            
            return AnalysisResult(
                analysis_type="correlation",
                results=results,
                insights=insights,
                warnings=warnings_list,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return AnalysisResult("correlation", {}, [], [f"Analysis failed: {str(e)}"], {}, False)

class PredictiveAnalyzer(BaseAnalyzer):
    """Performs predictive modeling"""
    
    def analyze(self, df: pd.DataFrame, target_column: str, 
                feature_columns: Optional[List[str]] = None,
                model_type: str = 'auto') -> AnalysisResult:
        """Build and evaluate predictive models"""
        
        try:
            issues = self._validate_data(df, [target_column])
            if issues:
                return AnalysisResult("predictive", {}, [], issues, {}, False)
            
            results = {}
            insights = []
            warnings_list = []
            
            # Prepare data
            if feature_columns:
                feature_df = df[feature_columns]
            else:
                feature_df = df.drop(columns=[target_column])
            
            # Remove non-numeric columns for now (could be enhanced later)
            numeric_features = feature_df.select_dtypes(include=[np.number])
            
            if numeric_features.empty:
                return AnalysisResult("predictive", {}, [], 
                                    ["No numeric features available for modeling"], {}, False)
            
            # Handle missing values
            X = numeric_features.fillna(numeric_features.median())
            y = df[target_column].fillna(df[target_column].mode().iloc[0] if not df[target_column].mode().empty else 0)
            
            # Determine if classification or regression
            is_classification = self._is_classification_task(y)
            
            # Split data
            test_size = self.analysis_config.get('test_size', 0.2)
            random_state = self.analysis_config.get('random_state', 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Choose and train model
            if model_type == 'auto':
                if is_classification:
                    model = RandomForestClassifier(random_state=random_state)
                else:
                    model = RandomForestRegressor(random_state=random_state)
            else:
                # Could add more model types here
                if is_classification:
                    model = RandomForestClassifier(random_state=random_state)
                else:
                    model = RandomForestRegressor(random_state=random_state)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate model
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                results['accuracy'] = accuracy
                results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
                insights.append(f"Model accuracy: {accuracy:.3f}")
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                results['confusion_matrix'] = cm.tolist()
                
            else:
                r2 = r2_score(y_test, y_pred)
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)
                
                results['r2_score'] = r2
                results['mse'] = mse
                results['rmse'] = rmse
                
                insights.append(f"Model R² score: {r2:.3f}")
                insights.append(f"RMSE: {rmse:.3f}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                results['feature_importance'] = dict(sorted_features)
                
                top_features = [f[0] for f in sorted_features[:5]]
                insights.append(f"Most important features: {', '.join(top_features)}")
            
            # Cross-validation
            cv_folds = self.analysis_config.get('cv_folds', 5)
            if is_classification:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy')
                results['cv_accuracy'] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                }
                insights.append(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            else:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
                results['cv_r2'] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                }
                insights.append(f"Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Model metadata
            results['model_type'] = type(model).__name__
            results['task_type'] = 'classification' if is_classification else 'regression'
            results['features_used'] = list(X.columns)
            results['train_size'] = len(X_train)
            results['test_size'] = len(X_test)
            
            metadata = {
                'target_column': target_column,
                'num_features': len(X.columns),
                'task_type': 'classification' if is_classification else 'regression',
                'model_used': type(model).__name__
            }
            
            return AnalysisResult(
                analysis_type="predictive",
                results=results,
                insights=insights,
                warnings=warnings_list,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in predictive analysis: {str(e)}")
            return AnalysisResult("predictive", {}, [], [f"Analysis failed: {str(e)}"], {}, False)
    
    def _is_classification_task(self, y: pd.Series) -> bool:
        """Determine if the target variable indicates a classification task"""
        # If target is string/object, it's classification
        if y.dtype == 'object':
            return True
        
        # If target has few unique values relative to total, likely classification
        unique_ratio = y.nunique() / len(y)
        if unique_ratio < 0.05 and y.nunique() <= 20:
            return True
        
        # If all values are integers and there are few unique values
        if y.dtype in ['int64', 'int32'] and y.nunique() <= 20:
            return True
        
        return False

class ClusteringAnalyzer(BaseAnalyzer):
    """Performs clustering analysis"""
    
    def analyze(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                n_clusters: Optional[int] = None, method: str = 'kmeans') -> AnalysisResult:
        """Perform clustering analysis"""
        
        try:
            # Get numeric columns
            if columns:
                numeric_df = df[columns].select_dtypes(include=[np.number])
            else:
                numeric_df = df.select_dtypes(include=[np.number])
            
            issues = self._validate_data(numeric_df)
            if len(numeric_df.columns) < 2:
                issues.append("Need at least 2 numeric columns for clustering")
            
            if issues:
                return AnalysisResult("clustering", {}, [], issues, {}, False)
            
            results = {}
            insights = []
            warnings_list = []
            
            # Handle missing values
            X = numeric_df.fillna(numeric_df.median())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine number of clusters if not specified
            if n_clusters is None:
                n_clusters = min(5, max(2, len(X) // 100))
                insights.append(f"Auto-selected {n_clusters} clusters based on data size")
            
            # Perform clustering
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = model.fit_predict(X_scaled)
                
                # Calculate cluster centers in original space
                cluster_centers = scaler.inverse_transform(model.cluster_centers_)
                results['cluster_centers'] = pd.DataFrame(cluster_centers, columns=X.columns).to_dict()
                
                # Inertia (within-cluster sum of squares)
                results['inertia'] = model.inertia_
                
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
                clusters = model.fit_predict(X_scaled)
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                results['n_clusters_found'] = n_clusters
                
                noise_points = sum(1 for c in clusters if c == -1)
                if noise_points > 0:
                    insights.append(f"DBSCAN found {noise_points} noise points")
            
            else:
                warnings_list.append(f"Unknown clustering method '{method}', using K-means")
                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(X_scaled)
            
            # Add clusters to results
            results['cluster_labels'] = clusters.tolist()
            results['n_clusters'] = n_clusters
            
            # Cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_mask = clusters == i
                cluster_data = X[cluster_mask]
                
                if len(cluster_data) > 0:
                    cluster_stats[f'cluster_{i}'] = {
                        'size': len(cluster_data),
                        'percentage': len(cluster_data) / len(X) * 100,
                        'mean_values': cluster_data.mean().to_dict(),
                        'std_values': cluster_data.std().to_dict()
                    }
            
            results['cluster_statistics'] = cluster_stats
            
            # Generate insights
            cluster_sizes = [sum(1 for c in clusters if c == i) for i in range(n_clusters)]
            insights.append(f"Created {n_clusters} clusters with sizes: {cluster_sizes}")
            
            # Check for very uneven clusters
            max_size, min_size = max(cluster_sizes), min(cluster_sizes)
            if max_size / min_size > 10:
                warnings_list.append("Clusters are very uneven in size")
            
            # Silhouette analysis (if scikit-learn version supports it)
            try:
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(X_scaled, clusters)
                results['silhouette_score'] = silhouette_avg
                
                if silhouette_avg > 0.7:
                    insights.append(f"Excellent cluster separation (silhouette: {silhouette_avg:.3f})")
                elif silhouette_avg > 0.5:
                    insights.append(f"Good cluster separation (silhouette: {silhouette_avg:.3f})")
                else:
                    insights.append(f"Moderate cluster separation (silhouette: {silhouette_avg:.3f})")
                    
            except ImportError:
                pass
            
            # PCA for visualization (if more than 2 dimensions)
            if len(X.columns) > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                results['pca_coordinates'] = {
                    'x': X_pca[:, 0].tolist(),
                    'y': X_pca[:, 1].tolist(),
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
                }
                
                total_variance = sum(pca.explained_variance_ratio_)
                insights.append(f"PCA visualization captures {total_variance:.1%} of variance")
            
            metadata = {
                'method': method,
                'features_used': list(X.columns),
                'n_samples': len(X),
                'n_features': len(X.columns)
            }
            
            return AnalysisResult(
                analysis_type="clustering",
                results=results,
                insights=insights,
                warnings=warnings_list,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
            return AnalysisResult("clustering", {}, [], [f"Analysis failed: {str(e)}"], {}, False)

class AnomalyAnalyzer(BaseAnalyzer):
    """Performs anomaly detection"""
    
    def analyze(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                method: str = 'isolation_forest') -> AnalysisResult:
        """Detect anomalies in the data"""
        
        try:
            # Get numeric columns
            if columns:
                numeric_df = df[columns].select_dtypes(include=[np.number])
            else:
                numeric_df = df.select_dtypes(include=[np.number])
            
            issues = self._validate_data(numeric_df)
            if numeric_df.empty:
                issues.append("No numeric columns available for anomaly detection")
            
            if issues:
                return AnalysisResult("anomaly", {}, [], issues, {}, False)
            
            results = {}
            insights = []
            warnings_list = []
            
            # Handle missing values
            X = numeric_df.fillna(numeric_df.median())
            
            # Scale features for distance-based methods
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                
                model = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = model.fit_predict(X_scaled)
                anomaly_scores = model.decision_function(X_scaled)
                
            elif method == 'statistical':
                # Z-score based anomaly detection
                z_scores = np.abs(stats.zscore(X_scaled))
                threshold = 3  # Standard 3-sigma rule
                anomaly_labels = np.where((z_scores > threshold).any(axis=1), -1, 1)
                anomaly_scores = np.max(z_scores, axis=1)  # Max z-score across features
                
            else:
                warnings_list.append(f"Unknown method '{method}', using statistical method")
                z_scores = np.abs(stats.zscore(X_scaled))
                threshold = 3
                anomaly_labels = np.where((z_scores > threshold).any(axis=1), -1, 1)
                anomaly_scores = np.max(z_scores, axis=1)
            
            # Process results
            anomalies = anomaly_labels == -1
            n_anomalies = sum(anomalies)
            anomaly_percentage = n_anomalies / len(X) * 100
            
            results['anomaly_labels'] = anomaly_labels.tolist()
            results['anomaly_scores'] = anomaly_scores.tolist()
            results['n_anomalies'] = n_anomalies
            results['anomaly_percentage'] = anomaly_percentage
            
            # Get anomalous data points
            if n_anomalies > 0:
                anomaly_indices = np.where(anomalies)[0]
                results['anomaly_indices'] = anomaly_indices.tolist()
                
                # Show top anomalies
                top_anomalies_idx = np.argsort(anomaly_scores)[::-1][:min(10, n_anomalies)]
                anomaly_data = X.iloc[top_anomalies_idx].to_dict('records')
                results['top_anomalies'] = anomaly_data
                
                insights.append(f"Detected {n_anomalies} anomalies ({anomaly_percentage:.1f}% of data)")
                
                # Analyze which features contribute most to anomalies
                if method == 'statistical':
                    anomaly_z_scores = np.abs(stats.zscore(X_scaled[anomalies]))
                    avg_z_scores = np.mean(anomaly_z_scores, axis=0)
                    feature_contribution = dict(zip(X.columns, avg_z_scores))
                    sorted_features = sorted(feature_contribution.items(), key=lambda x: x[1], reverse=True)
                    results['feature_contribution'] = dict(sorted_features)
                    
                    top_feature = sorted_features[0][0]
                    insights.append(f"'{top_feature}' contributes most to anomalies")
            
            else:
                insights.append("No anomalies detected with current parameters")
            
            # Summary statistics for normal vs anomalous points
            if n_anomalies > 0:
                normal_data = X[~anomalies]
                anomaly_data = X[anomalies]
                
                comparison = {}
                for col in X.columns:
                    comparison[col] = {
                        'normal_mean': normal_data[col].mean(),
                        'anomaly_mean': anomaly_data[col].mean(),
                        'normal_std': normal_data[col].std(),
                        'anomaly_std': anomaly_data[col].std()
                    }
                
                results['normal_vs_anomaly'] = comparison
            
            metadata = {
                'method': method,
                'features_analyzed': list(X.columns),
                'threshold_used': threshold if method == 'statistical' else 'auto',
                'total_samples': len(X)
            }
            
            return AnalysisResult(
                analysis_type="anomaly",
                results=results,
                insights=insights,
                warnings=warnings_list,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return AnalysisResult("anomaly", {}, [], [f"Analysis failed: {str(e)}"], {}, False)

# Analysis Engine Factory
class AnalysisEngineFactory:
    """Factory for creating analysis engines"""
    
    @staticmethod
    def create_analyzer(analysis_type: str, config: Dict[str, Any]) -> BaseAnalyzer:
        """Create the appropriate analyzer based on analysis type"""
        
        analyzers = {
            'descriptive': DescriptiveAnalyzer,
            'correlation': CorrelationAnalyzer,
            'predictive': PredictiveAnalyzer,
            'clustering': ClusteringAnalyzer,
            'anomaly': AnomalyAnalyzer,
        }
        
        analyzer_class = analyzers.get(analysis_type)
        if analyzer_class:
            return analyzer_class(config)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
