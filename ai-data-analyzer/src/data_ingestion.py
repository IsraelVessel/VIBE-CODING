"""
Data Ingestion Module for AI Data Analyzer
Handles various data formats and standardizes them for analysis
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from sqlalchemy import create_engine
import pymongo
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class DataIngestionError(Exception):
    """Custom exception for data ingestion errors"""
    pass

class DataIngestor:
    """Main class for data ingestion from various sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = config.get('data', {}).get('supported_formats', 
                                                            ['csv', 'xlsx', 'xls', 'json', 'parquet'])
        self.max_file_size = config.get('data', {}).get('max_file_size', 500)  # MB
        
    def load_data(self, source: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from various sources
        
        Args:
            source: File path, URL, or connection string
            **kwargs: Additional parameters for specific loaders
            
        Returns:
            pandas.DataFrame: Loaded and preprocessed data
        """
        try:
            # Determine source type
            if isinstance(source, (str, Path)):
                source_path = Path(source)
                
                # Check if it's a file
                if source_path.exists():
                    return self._load_from_file(source_path, **kwargs)
                
                # Check if it's a URL
                if str(source).startswith(('http://', 'https://')):
                    return self._load_from_url(str(source), **kwargs)
                
                # Check if it's a database connection string
                if any(db in str(source).lower() for db in ['postgresql', 'mysql', 'sqlite', 'mongodb']):
                    return self._load_from_database(str(source), **kwargs)
            
            raise DataIngestionError(f"Unable to determine source type for: {source}")
            
        except Exception as e:
            logger.error(f"Error loading data from {source}: {str(e)}")
            raise DataIngestionError(f"Failed to load data: {str(e)}")
    
    def _load_from_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from local file"""
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size:
            raise DataIngestionError(f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({self.max_file_size}MB)")
        
        # Get file extension
        file_ext = file_path.suffix.lower().lstrip('.')
        
        if file_ext not in self.supported_formats:
            raise DataIngestionError(f"Unsupported file format: {file_ext}")
        
        # Load based on file type
        if file_ext == 'csv':
            return self._load_csv(file_path, **kwargs)
        elif file_ext in ['xlsx', 'xls']:
            return self._load_excel(file_path, **kwargs)
        elif file_ext == 'json':
            return self._load_json(file_path, **kwargs)
        elif file_ext == 'parquet':
            return self._load_parquet(file_path, **kwargs)
        else:
            raise DataIngestionError(f"No loader implemented for format: {file_ext}")
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file with intelligent parameter detection"""
        try:
            # Try to detect encoding
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise DataIngestionError("Could not decode CSV file with any supported encoding")
            
            return self._preprocess_dataframe(df)
            
        except Exception as e:
            raise DataIngestionError(f"Error loading CSV file: {str(e)}")
    
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file"""
        try:
            # If no sheet specified, load all sheets and combine
            sheet_name = kwargs.get('sheet_name', None)
            
            if sheet_name is None:
                # Get all sheet names
                excel_file = pd.ExcelFile(file_path)
                if len(excel_file.sheet_names) == 1:
                    df = pd.read_excel(file_path, **kwargs)
                else:
                    # Load first sheet by default, but inform user about other sheets
                    df = pd.read_excel(file_path, sheet_name=0, **kwargs)
                    logger.info(f"Multiple sheets found: {excel_file.sheet_names}. Loaded first sheet.")
            else:
                df = pd.read_excel(file_path, **kwargs)
            
            return self._preprocess_dataframe(df)
            
        except Exception as e:
            raise DataIngestionError(f"Error loading Excel file: {str(e)}")
    
    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # If dict has a key that contains list data, use that
                list_keys = [k for k, v in data.items() if isinstance(v, list)]
                if list_keys:
                    # Use the first list found
                    df = pd.DataFrame(data[list_keys[0]])
                else:
                    # Convert single record to DataFrame
                    df = pd.DataFrame([data])
            else:
                raise DataIngestionError("JSON structure not supported for DataFrame conversion")
            
            return self._preprocess_dataframe(df)
            
        except Exception as e:
            raise DataIngestionError(f"Error loading JSON file: {str(e)}")
    
    def _load_parquet(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file"""
        try:
            df = pd.read_parquet(file_path, **kwargs)
            return self._preprocess_dataframe(df)
        except Exception as e:
            raise DataIngestionError(f"Error loading Parquet file: {str(e)}")
    
    def _load_from_url(self, url: str, **kwargs) -> pd.DataFrame:
        """Load data from URL"""
        try:
            # Determine file type from URL
            parsed_url = urlparse(url)
            file_ext = Path(parsed_url.path).suffix.lower().lstrip('.')
            
            if file_ext == 'csv':
                df = pd.read_csv(url, **kwargs)
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(url, **kwargs)
            else:
                # Try CSV as default
                df = pd.read_csv(url, **kwargs)
            
            return self._preprocess_dataframe(df)
            
        except Exception as e:
            raise DataIngestionError(f"Error loading data from URL: {str(e)}")
    
    def _load_from_database(self, connection_string: str, query: Optional[str] = None, 
                          table_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from database"""
        try:
            if 'mongodb' in connection_string.lower():
                return self._load_from_mongodb(connection_string, **kwargs)
            else:
                return self._load_from_sql_database(connection_string, query, table_name, **kwargs)
                
        except Exception as e:
            raise DataIngestionError(f"Error loading data from database: {str(e)}")
    
    def _load_from_sql_database(self, connection_string: str, query: Optional[str], 
                              table_name: Optional[str], **kwargs) -> pd.DataFrame:
        """Load data from SQL database"""
        try:
            engine = create_engine(connection_string)
            
            if query:
                df = pd.read_sql(query, engine, **kwargs)
            elif table_name:
                df = pd.read_sql_table(table_name, engine, **kwargs)
            else:
                # Get list of tables and use the first one
                inspector = engine.inspect()
                tables = inspector.get_table_names()
                if not tables:
                    raise DataIngestionError("No tables found in database")
                df = pd.read_sql_table(tables[0], engine, **kwargs)
                logger.info(f"No table specified, loaded: {tables[0]}")
            
            return self._preprocess_dataframe(df)
            
        except Exception as e:
            raise DataIngestionError(f"Error loading from SQL database: {str(e)}")
    
    def _load_from_mongodb(self, connection_string: str, collection: Optional[str] = None, 
                          **kwargs) -> pd.DataFrame:
        """Load data from MongoDB"""
        try:
            client = pymongo.MongoClient(connection_string)
            
            # Get database name from connection string or use first available
            db_name = connection_string.split('/')[-1] if '/' in connection_string else None
            if not db_name:
                db_name = client.list_database_names()[0]
            
            db = client[db_name]
            
            # Get collection
            if not collection:
                collection = db.list_collection_names()[0]
                logger.info(f"No collection specified, using: {collection}")
            
            # Load data
            cursor = db[collection].find()
            data = list(cursor)
            
            if not data:
                raise DataIngestionError("No data found in collection")
            
            df = pd.DataFrame(data)
            
            # Remove MongoDB ObjectId if present
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            return self._preprocess_dataframe(df)
            
        except Exception as e:
            raise DataIngestionError(f"Error loading from MongoDB: {str(e)}")
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the loaded dataframe"""
        try:
            # Basic cleaning
            df_clean = df.copy()
            
            # Remove completely empty rows and columns
            df_clean = df_clean.dropna(how='all')
            df_clean = df_clean.dropna(axis=1, how='all')
            
            # Auto-detect data types if configured
            if self.config.get('data', {}).get('auto_dtype', True):
                df_clean = self._auto_detect_types(df_clean)
            
            # Parse dates if configured
            if self.config.get('data', {}).get('parse_dates', True):
                df_clean = self._auto_parse_dates(df_clean)
            
            # Log basic info
            logger.info(f"Data loaded successfully: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
            
            return df_clean
            
        except Exception as e:
            logger.warning(f"Error in preprocessing, returning original dataframe: {str(e)}")
            return df
    
    def _auto_detect_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect and convert data types"""
        try:
            df_typed = df.copy()
            
            for col in df_typed.columns:
                # Skip if already numeric
                if pd.api.types.is_numeric_dtype(df_typed[col]):
                    continue
                
                # Try to convert to numeric
                try:
                    df_typed[col] = pd.to_numeric(df_typed[col], errors='ignore')
                except:
                    pass
            
            return df_typed
        except Exception as e:
            logger.warning(f"Error in auto type detection: {str(e)}")
            return df
    
    def _auto_parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect and parse date columns"""
        try:
            df_dated = df.copy()
            
            for col in df_dated.columns:
                # Skip if already datetime
                if pd.api.types.is_datetime64_any_dtype(df_dated[col]):
                    continue
                
                # Look for date-like column names
                date_keywords = ['date', 'time', 'created', 'updated', 'timestamp']
                if any(keyword in col.lower() for keyword in date_keywords):
                    try:
                        df_dated[col] = pd.to_datetime(df_dated[col], errors='ignore')
                    except:
                        pass
            
            return df_dated
        except Exception as e:
            logger.warning(f"Error in auto date parsing: {str(e)}")
            return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a summary of the loaded data"""
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
        }
        
        # Add numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        return summary
