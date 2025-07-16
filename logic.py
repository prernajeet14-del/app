# Standard Library Imports
import re
import logging
import os
import json
import time
from datetime import datetime, date
import warnings
import ast

# Third-party Library Imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import create_engine, inspect, text
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Any, Dict, List, Optional, Type
from scipy.stats import zscore
from ydata_profiling import ProfileReport

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add these with your other sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # Added GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression # Added LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mpl_toolkits.mplot3d import Axes3D

# FIXED: Added missing import for Market Basket Analysis
from mlxtend.frequent_patterns import apriori, association_rules

# --- LangChain, Memory, and RAG Imports ---
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory


# --- Configuration ---
# Replace with your real Azure SQL credentials
AZURE_SERVER = 'marketinganalytics1.database.windows.net'
AZURE_DATABASE = 'analyticsDB'
AZURE_USERNAME = 'serveradmin'
AZURE_PASSWORD = 'hotgex-fAkpu0-saxjug'  # Ensure correct escaping if special characters are present
AZURE_DRIVER = 'ODBC Driver 18 for SQL Server'

# Build SQLAlchemy connection string
connection_string = (
    f"mssql+pyodbc://{AZURE_USERNAME}:{AZURE_PASSWORD}@{AZURE_SERVER}:1433/{AZURE_DATABASE}"
    f"?driver={AZURE_DRIVER.replace(' ', '+')}&Encrypt=yes&TrustServerCertificate=no"
)

# Set API Key
# Replace with your actual key and provider.
os.environ["GEMINI_API_KEY"] = "AIzaSyAwoVq3n3sepoEJB2VfJ36Dwn1rm-nW-_8"

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from crewai import LLM
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- JSON Serialization Helper ---
def make_json_serializable(obj):
    """Convert non-serializable objects to JSON-serializable format"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'tolist'):  # Handle numpy arrays and similar
        return obj.tolist()
    else:
        try:
            return str(obj)
        except:
            return None


def sanitize_dict_keys(obj):
    """Recursively convert all keys of a dictionary to strings."""
    if isinstance(obj, dict):
        return {str(k): sanitize_dict_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_dict_keys(elem) for elem in obj]
    else:
        return obj


# --- Alternative to pandas to_markdown() ---
def dataframe_to_markdown(df, max_rows=20):
    """Convert DataFrame to markdown format without tabulate dependency"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "| No data available |\n|---|\n"

    # Limit rows for readability
    df_display = df.head(max_rows) if len(df) > max_rows else df

    # Create header
    headers = df_display.columns.tolist()
    markdown = "| " + " | ".join(str(h) for h in headers) + " |\n"
    markdown += "| " + " | ".join("---" for _ in headers) + " |\n"

    # Add rows
    for _, row in df_display.iterrows():
        markdown += "| " + " | ".join(str(val) for val in row) + " |\n"

    if len(df) > max_rows:
        markdown += f"\n*Showing first {max_rows} rows out of {len(df)} total rows*\n"

    return markdown

def parse_agent_output(raw_output: str) -> dict:
    """
    Safely parses a string that might contain a Python dictionary or JSON.
    Handles markdown code blocks and extraneous text around the dictionary.
    """
    if isinstance(raw_output, dict):
        return raw_output  # Already a dictionary

    if not isinstance(raw_output, str):
        return {"error": "Output is not a string or dictionary", "raw_content": str(raw_output)}

    # Use regex to find the dictionary within the raw output string
    match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if match:
        clean_output = match.group(0)
    else:
        # Fallback for outputs that are not wrapped in brackets but might still be parsable
        clean_output = re.sub(r'^```(json|python)?\n|\n```$', '', raw_output.strip())

    try:
        # ast.literal_eval is safer than eval() and can handle Python dict syntax
        return ast.literal_eval(clean_output)
    except (ValueError, SyntaxError, MemoryError) as e:
        # If parsing fails, return an error dictionary with the raw content
        return {"error": f"Failed to parse tool output with ast.literal_eval: {e}", "raw_content": raw_output}


# --- Database Connection Helper ---
def get_db_engine():
    """Create and return a database engine with proper error handling"""
    try:
        engine = create_engine(connection_string, pool_pre_ping=True)
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        logging.error(f"Failed to create database engine: {e}")
        raise


# --- Tool Classes ---

class ClarificationTool(BaseTool):
    name: str = "Clarification Question Context Provider"
    description: str = (
        "Provides structured context for an LLM to generate a clarifying question. "
        "It takes the user's query, analysis type, conversation history, and a question guide, "
        "and formats this into a prompt for the agent to determine the next logical question."
    )
    
    class ClarificationArgs(BaseModel):
        query: str = Field(..., description="The user's original business query.")
        analysis_type: str = Field(..., description="The type of analysis chosen (e.g., 'RFM', 'Churn').")
        history: List[Dict[str, str]] = Field(default_factory=list, description="List of prior questions and answers.")
        question_guide: str = Field(..., description="Newline-separated questions from a mapping file to use as a guide.")
    
    args_schema: Type[BaseModel] = ClarificationArgs

    def _run(self, query: str, analysis_type: str, history: List[Dict[str, str]], question_guide: str) -> str:
        history_str = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in history]) or "No questions have been asked yet."
        
        additional_prompt = ""
        if history and history[-1]["question"] == "USER_REQUEST_MORE_QUESTIONS":
            additional_prompt = (
                "The user was not satisfied with the initial questions and has requested more. "
                "Please generate another 4-5 relevant and distinct clarifying questions based on the full context. "
                "If you genuinely cannot think of more questions, return the completion phrase."
            )
        
        context_for_llm = (
            f"User's Original Query: \"{query}\"\n"
            f"Chosen Analysis Type: {analysis_type}\n"
            f"Conversation History:\n{history_str}\n\n"
            f"Question Guide for '{analysis_type}' Analysis:\n{question_guide}\n\n"
            "Based on the context, formulate the single, most logical next clarifying question. "
            "Do not ask about something already answered. If all guide questions are covered or you have enough info, "
            "return the exact phrase 'I don't have any more questions.'\n\n"
            f"{additional_prompt}\n\n"
            "IMPORTANT: Your output MUST ONLY be the question text or the completion phrase. No other text."
        )
        return context_for_llm


class SchemaAnalyzerTool(BaseTool):
    name: str = "Schema Analyzer Tool"
    description: str = "Connects to DB via URL, analyzes schema, returns table/column info."
    
    class SchemaAnalyzerArgs(BaseModel):
        database_url: str = Field(..., description="SQLAlchemy database URL")
    
    args_schema: Type[BaseModel] = SchemaAnalyzerArgs
    
    def _run(self, database_url: str) -> Dict[str, Any]:
        logging.info("SchemaAnalyzer: Connecting to DB...")
        max_retries = 3
        retry_delay_seconds = 5

        for attempt in range(max_retries):
            engine = None
            try:
                engine = get_db_engine()
                inspector = inspect(engine)
                table_names = inspector.get_table_names()
                logging.debug(f"Tables found: {table_names}")

                if not table_names:
                    logging.warning(f"SchemaAnalyzer: No tables found in database on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay_seconds)
                        continue
                    return {"error": "No tables found in database after multiple attempts"}

                schema = {}
                for table_name in table_names:
                    try:
                        columns = inspector.get_columns(table_name)
                        schema[table_name] = [col['name'] for col in columns]
                    except Exception as e:
                        logging.warning(f"Could not get columns for table {table_name}: {e}")
                        continue
                
                if not schema:
                    return {"error": "No table schemas could be retrieved"}
                
                logging.info(f"SchemaAnalyzer: Success for tables: {list(schema.keys())}")
                return {"schema_data": schema}
                
            except Exception as e:
                logging.error(f"SchemaAnalyzer: Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_seconds)
                else:
                    return {"error": f"Schema analysis failed after {max_retries} attempts: {e}"}
            finally:
                if engine:
                    engine.dispose()

        return {"error": "Schema analysis failed unexpectedly."}

class PreprocessingTool(BaseTool):
    name: str = "Preprocessing Tool"
    description: str = "Cleans a given pandas DataFrame using missing value handling, outlier removal, and duplicate dropping. This is an internal tool and not meant for direct agent use."

    class PreprocessingArgs(BaseModel):
        df: pd.DataFrame = Field(..., description="The pandas DataFrame to preprocess.")
        table_name: str = Field(..., description="The original name of the table for logging purposes.")
        required_columns: List[str] = Field(..., description="A list of columns essential for the analysis.")

        model_config = ConfigDict(arbitrary_types_allowed=True)

    args_schema: Type[BaseModel] = PreprocessingArgs

    def _run(self, df: pd.DataFrame, table_name: str, required_columns: List[str]) -> pd.DataFrame:
        logging.info(f"✅ Preprocessing data from table: {table_name}")

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        df_process = df.copy()

        # Column Selection
        column_mapping = {}
        missing_cols = []
        for req_col in required_columns:
            found_match = False
            for actual_col in df_process.columns:
                if req_col.lower() == actual_col.lower():
                    column_mapping[req_col] = actual_col
                    found_match = True
                    break
            if not found_match:
                missing_cols.append(req_col)

        if missing_cols:
            raise ValueError(f"Required columns not found: {missing_cols}. Available: {list(df.columns)}")

        df_process = df_process[list(column_mapping.values())]
        logging.info(f"Selected columns: {list(df_process.columns)}")
        
        initial_rows = len(df_process)
        df_clean = df_process.drop_duplicates().copy()
        logging.info(f"Dropped {initial_rows - len(df_clean)} duplicate rows.")

        if len(df_clean) > 0:
            threshold = 0.5 * len(df_clean)
            cols_to_drop_nan = df_clean.columns[df_clean.isnull().sum() > threshold]
            df_clean.drop(columns=cols_to_drop_nan, inplace=True)
            if len(cols_to_drop_nan) > 0:
                logging.info(f"Dropped columns with >50% NaN: {list(cols_to_drop_nan)}")

        if df_clean.empty:
            raise ValueError("DataFrame became empty after cleaning.")

        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
        
        logging.info(f"Preprocessing complete. Final shape: {df_clean.shape}")
        return df_clean

class LoadAndPreprocessDataTool(BaseTool):
    name: str = "Load and Preprocess Data Tool"
    description: str = (
        "Executes a SQL query to load data from a table, preprocesses it, "
        "saves the cleaned DataFrame to a pickle file, and returns the file path."
    )

    class LoadAndPreprocessArgs(BaseModel):
        table_name: str = Field(..., description="The database table to load data from.")
        required_columns: List[str] = Field(..., description="Essential columns for the analysis.")

    args_schema: Type[BaseModel] = LoadAndPreprocessArgs

    def _run(self, table_name: str, required_columns: List[str]) -> str:
        logging.info(f"✅ Executing Load and Preprocess for table: {table_name}")
        
        query = f"SELECT * FROM {table_name}"
        df = pd.DataFrame()
        try:
            engine = get_db_engine()
            df = pd.read_sql_query(query, engine)
            logging.info(f"Loaded DataFrame with shape: {df.shape} from '{table_name}'")
        except Exception as e:
            logging.error(f"SQL Query failed for table '{table_name}': {e}")
            raise
        finally:
            if 'engine' in locals() and engine:
                engine.dispose()
        
        if df.empty:
            raise ValueError(f"No data loaded from table '{table_name}'.")

        preprocessor = PreprocessingTool()
        cleaned_df = preprocessor._run(df=df, table_name=table_name, required_columns=required_columns)
        
        output_dir = "analysis_temp_data"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"{table_name}_cleaned_{timestamp}.pkl")
        cleaned_df.to_pickle(file_path)
        
        logging.info(f"Saved cleaned DataFrame to '{file_path}'")
        return file_path

def sanitize_dict_keys(obj):
    """Recursively convert all keys of a dictionary to strings."""
    if isinstance(obj, dict):
        return {str(k): sanitize_dict_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_dict_keys(elem) for elem in obj]
    else:
        return obj

class EDAExplorationTool(BaseTool):
    name: str = "EDA Exploration Tool"
    description: str = "Loads a DataFrame from a file path and performs comprehensive EDA, including a detailed profile report."

    class EDAArgs(BaseModel):
        file_path: str = Field(..., description="Path to the pickle file containing the cleaned dataframe for EDA.")
        table_name: Optional[str] = Field(None, description="Name of the original table for report labeling. This is now optional.")

    args_schema: type[BaseModel] = EDAArgs

    def _run(self, file_path: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        logging.info(f"✅ Performing EDA on data from: {file_path}")
        
        if not table_name:
            match = re.search(r'([a-zA-Z0-9]+)_cleaned_', os.path.basename(file_path))
            if match:
                table_name = match.group(1)
                logging.info(f"Inferred table name as '{table_name}' from file path.")
            else:
                table_name = "EDA_Report" # Fallback name
                logging.warning("Could not infer table name from file path. Using default.")

        try:
            df = pd.read_pickle(file_path)
        except Exception as e:
            return {"error": f"Failed to load DataFrame from {file_path}: {e}"}

        if df.empty:
            return {"error": "DataFrame is empty"}

        # --- ydata-profiling Integration ---
        profile_report_path = None
        try:
            logging.info(f"Generating ydata-profiling report for {table_name}...")
            profile_dir = "profiling_reports"
            os.makedirs(profile_dir, exist_ok=True)

            profile = ProfileReport(df, title=f"Data Profile Report: {table_name}", explorative=True)

            profile_report_path = os.path.join(profile_dir, f"{table_name}_profile_report.html")
            profile.to_file(profile_report_path)
            logging.info(f"Successfully saved profile report to {profile_report_path}")
        except Exception as e:
            logging.warning(f"Could not generate or save ydata-profiling report: {e}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        eda_results = {
            "basic_stats": {}, "missing_values": {}, "correlations": {}, "distributions": {},
            "outliers": {}, "categorical_analysis": {}, "profile_report_path": profile_report_path
        }
        output_dir = "eda_visuals"
        os.makedirs(output_dir, exist_ok=True)

        if numeric_cols:
            try:
                eda_results["basic_stats"] = df[numeric_cols].describe().to_dict()
            except Exception as e:
                logging.warning(f"Could not generate basic statistics: {e}")

        missing_values = df.isnull().sum()
        eda_results["missing_values"] = missing_values[missing_values > 0].to_dict()

        if len(numeric_cols) > 1:
            try:
                correlation_matrix = df[numeric_cols].corr()
                eda_results["correlations"] = correlation_matrix.to_dict()
                plt.figure(figsize=(12, 10))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                plt.title(f'Correlation Matrix - {table_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{table_name}_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logging.warning(f"Could not generate correlation heatmap: {e}")

        for col in numeric_cols:
            try:
                if df[col].nunique() > 1:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df[col].dropna(), kde=True, bins=30)
                    plt.title(f'Distribution of {col}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'{table_name}_{col}_distribution.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                logging.warning(f"Could not plot distribution for {col}: {e}")

        if categorical_cols:
            cat_analysis = {}
            for col in categorical_cols:
                try:
                    value_counts = df[col].value_counts().head(10)
                    cat_analysis[col] = value_counts.to_dict()
                except Exception as e:
                    logging.warning(f"Could not analyze categorical column {col}: {e}")
            eda_results["categorical_analysis"] = cat_analysis

        logging.info(f"EDA completed. Visuals saved to '{output_dir}' directory.")
        return sanitize_dict_keys(eda_results)

# --- CORRECTED FeatureEngineeringTool ---
class FeatureEngineeringTool(BaseTool):
    name: str = "Feature Engineering Tool"
    description: str = "Loads data from a file, creates new features for analysis like RFM, performs binning and encoding, saves the engineered DataFrame, and returns the new file path."

    class FeatureEngineeringArgs(BaseModel):
        file_path: str = Field(..., description="Path to the pickle file with cleaned data.")
        customer_col: Optional[str] = Field(None, description="Name of the customer ID column.")
        date_col: Optional[str] = Field(None, description="Name of the date column.")
        monetary_col: Optional[str] = Field(None, description="Name of the monetary value column.")
        columns_to_bin: Optional[List[Dict[str, Any]]] = Field(None, description="A list of dictionaries for binning. Each dict should have 'column_name' and 'bins'. For example: [{'column_name': 'Age', 'bins': 4}]")
        columns_to_encode: Optional[List[str]] = Field(None, description="A list of categorical column names to be one-hot encoded.")

    args_schema: Type[BaseModel] = FeatureEngineeringArgs

    def _run(self, file_path: str, customer_col: Optional[str] = None,
             date_col: Optional[str] = None, monetary_col: Optional[str] = None,
             columns_to_bin: Optional[List[Dict[str, Any]]] = None,
             columns_to_encode: Optional[List[str]] = None) -> str:
        logging.info(f"✅ Performing Feature Engineering on data from {file_path}...")
        try:
            df = pd.read_pickle(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load DataFrame from {file_path}: {e}")

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        df_engineered = df.copy()

        def find_actual_column_name(df_columns, requested_name):
            """Finds the actual column name in a case-insensitive way."""
            if not requested_name:
                return None
            for col in df_columns:
                if col.lower().strip() == requested_name.lower().strip():
                    return col
            logging.warning(f"Column '{requested_name}' not found. Available columns: {list(df_columns)}")
            return None

        actual_date_col = find_actual_column_name(df_engineered.columns, date_col)
        actual_customer_col = find_actual_column_name(df_engineered.columns, customer_col)
        actual_monetary_col = find_actual_column_name(df_engineered.columns, monetary_col)

        if actual_date_col:
            try:
                df_engineered[actual_date_col] = pd.to_datetime(df_engineered[actual_date_col], errors='coerce')
                initial_rows = len(df_engineered)
                df_engineered.dropna(subset=[actual_date_col], inplace=True)
                dropped_rows = initial_rows - len(df_engineered)
                if dropped_rows > 0:
                    logging.info(f"Dropped {dropped_rows} rows with invalid dates in column '{actual_date_col}'")
                if not df_engineered.empty:
                    df_engineered['days_since_last_purchase'] = (datetime.now() - df_engineered[actual_date_col]).dt.days
                    df_engineered['year'] = df_engineered[actual_date_col].dt.year
                    df_engineered['month'] = df_engineered[actual_date_col].dt.month
                    df_engineered['day_of_week'] = df_engineered[actual_date_col].dt.dayofweek
                    logging.info(f"Created time-based features from '{actual_date_col}'.")
            except Exception as e:
                logging.warning(f"Could not process date column '{date_col}': {e}")

        if actual_customer_col and actual_monetary_col:
            try:
                df_engineered[actual_monetary_col] = pd.to_numeric(df_engineered[actual_monetary_col], errors='coerce')
                agg_features = df_engineered.groupby(actual_customer_col).agg({
                    actual_monetary_col: ['sum', 'mean', 'count', 'std'],
                }).reset_index()
                agg_features.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_features.columns]
                agg_features.rename(columns={
                    f'{actual_monetary_col}_sum': 'customer_total_spent',
                    f'{actual_monetary_col}_mean': 'customer_avg_order_value',
                    f'{actual_monetary_col}_count': 'customer_frequency',
                    f'{actual_monetary_col}_std': 'customer_spending_std'
                }, inplace=True)
                df_engineered = df_engineered.merge(agg_features, on=actual_customer_col, how='left')
                logging.info(f"Created customer-level aggregation features using '{actual_customer_col}' and '{actual_monetary_col}'.")
            except Exception as e:
                logging.warning(f"Could not create customer aggregation features: {e}")

        # Binning Section
        if columns_to_bin:
            for b in columns_to_bin:
                col_to_bin = find_actual_column_name(df_engineered.columns, b.get('column_name'))
                num_bins = b.get('bins')
                if col_to_bin and num_bins:
                    try:
                        df_engineered[f'{col_to_bin}_binned'] = pd.cut(df_engineered[col_to_bin], bins=num_bins, labels=False, include_lowest=True, duplicates='drop')
                        logging.info(f"Successfully binned column '{col_to_bin}' into {num_bins} bins.")
                    except Exception as e:
                        logging.warning(f"Could not bin column '{col_to_bin}': {e}")

        # Encoding Section
        if columns_to_encode:
            for col_to_encode_req in columns_to_encode:
                col_to_encode = find_actual_column_name(df_engineered.columns, col_to_encode_req)
                if col_to_encode and col_to_encode in df_engineered.columns:
                    try:
                        dummies = pd.get_dummies(df_engineered[col_to_encode], prefix=col_to_encode, drop_first=True)
                        df_engineered = pd.concat([df_engineered.drop(columns=[col_to_encode]), dummies], axis=1)
                        logging.info(f"Successfully one-hot encoded column '{col_to_encode}'.")
                    except Exception as e:
                        logging.warning(f"Could not encode column '{col_to_encode}': {e}")


        if 'days_since_last_purchase' in df_engineered.columns:
            try:
                df_engineered['churn'] = (df_engineered['days_since_last_purchase'] > 90).astype(int)
                churn_count = df_engineered['churn'].sum()
                total_customers = len(df_engineered)
                churn_rate = (churn_count / total_customers) * 100 if total_customers > 0 else 0
                logging.info(f"Applied churn definition (>90 days). Customers labeled as churned: {churn_count} ({churn_rate:.1f}%)")
            except Exception as e:
                logging.warning(f"Could not create churn feature: {e}")

        if all(col in df_engineered.columns for col in ['days_since_last_purchase', 'customer_frequency', 'customer_total_spent']):
            try:
                df_engineered['recency_score'] = pd.qcut(df_engineered['days_since_last_purchase'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
                df_engineered['frequency_score'] = pd.qcut(df_engineered['customer_frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                df_engineered['monetary_score'] = pd.qcut(df_engineered['customer_total_spent'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                df_engineered['recency_score'] = pd.to_numeric(df_engineered['recency_score'], errors='coerce')
                df_engineered['frequency_score'] = pd.to_numeric(df_engineered['frequency_score'], errors='coerce')
                df_engineered['monetary_score'] = pd.to_numeric(df_engineered['monetary_score'], errors='coerce')
                logging.info("Successfully created RFM scores.")
            except Exception as e:
                logging.warning(f"Could not create RFM scores: {e}")

        output_dir = "analysis_temp_data"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_file_path = os.path.join(output_dir, f"engineered_data_{timestamp}.pkl")
        df_engineered.to_pickle(new_file_path)

        logging.info(f"Feature Engineering complete. Saved to '{new_file_path}'")
        return new_file_path

# --- UPDATED: ChurnPredictionTool with Selectable Models ---
class ChurnPredictionTool(BaseTool):
    """
    A tool for predicting customer churn. It can either run a specific, pre-selected
    model or compare multiple models to find the best one.
    """
    name: str = "Churn Prediction Tool"
    description: str = (
        "Loads engineered data, trains one or more classification models (Random Forest, Gradient Boosting, Logistic Regression), "
        "and returns a detailed analysis. If no model is specified, it compares them to find the best performer."
    )

    class ChurnPredictionArgs(BaseModel):
        file_path: str = Field(..., description="Path to the pickle file with engineered data for churn modeling.")
        models_to_run: Optional[List[str]] = Field(None, description="A list of model names to run, e.g., ['Random Forest']. If None, all models are run and compared.")

    args_schema: Type[BaseModel] = ChurnPredictionArgs

    def _run(self, file_path: str, models_to_run: Optional[List[str]] = None) -> Dict[str, Any]:
        logging.info("✅ Starting Churn Prediction...")
        
        # --- 1. Data Loading and Validation ---
        try:
            df = pd.read_pickle(file_path)
            if 'churn' not in df.columns: df = df.rename(columns={'Churn': 'churn'}) # Normalize target column name
        except Exception as e:
            return {"error": f"Failed to load DataFrame from {file_path}: {e}"}

        if df.empty: return {"error": "DataFrame is empty."}
        if 'churn' not in df.columns: return {"error": "Target column 'churn' not found."}
        if df['churn'].nunique() < 2: return {"error": "Target variable 'churn' must have at least two unique classes."}

        customer_id_col = next((col for col in df.columns if 'customer' in col.lower() and 'id' in col.lower()), None)
        target = 'churn'
        features = [f for f in df.select_dtypes(include=np.number).columns if f != target and 'id' not in f.lower()]
        if not features: return {"error": "No valid numeric features found for modeling."}

        X = df[features].copy().fillna(df[features].median())
        y = df[target].copy()

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        except ValueError as e:
            return {"error": f"Could not split data. This can happen if one class has too few samples: {e}"}

        # --- 2. Model Initialization ---
        all_models = {
            "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
        }

        # --- 3. Determine Execution Path: Single Model or Comparison ---
        
        # If a specific model is chosen, run analysis for it directly
        if models_to_run and len(models_to_run) == 1:
            model_name = models_to_run[0]
            if model_name not in all_models:
                return {"error": f"Model '{model_name}' is not a valid choice. Available models are: {list(all_models.keys())}"}
            
            logging.info(f"Running analysis for single selected model: {model_name}")
            model = all_models[model_name]
            
            # Scale data if necessary
            X_train_processed, X_test_processed = X_train, X_test
            if isinstance(model, LogisticRegression):
                scaler = MinMaxScaler()
                X_train_processed = scaler.fit_transform(X_train)
                X_test_processed = scaler.transform(X_test)
            
            # Train and predict
            model.fit(X_train_processed, y_train)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate metrics and feature importances
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            churn_drivers = self._get_feature_importances(model, features)
            
            # Final output for a single model
            return {
                "selected_model": model_name,
                "evaluation_metrics": metrics,
                "churn_drivers": churn_drivers,
                "model_comparison": [], # No comparison to show
                "customers_at_risk": [] # Placeholder for at-risk customers
            }

        # --- 4. Comparison Mode: Run all specified (or all) models and find the best ---
        else:
            logging.info("Running in comparison mode to find the best model.")
            models_to_train = {name: model for name, model in all_models.items() if not models_to_run or name in models_to_run}
            if not models_to_train:
                logging.warning(f"No valid models in {models_to_run}. Falling back to all models.")
                models_to_train = all_models
            
            results = {}
            for name, model in models_to_train.items():
                try:
                    X_train_p, X_test_p = X_train, X_test
                    if isinstance(model, LogisticRegression):
                        scaler = MinMaxScaler()
                        X_train_p = scaler.fit_transform(X_train)
                        X_test_p = scaler.transform(X_test)
                    
                    model.fit(X_train_p, y_train)
                    y_pred_proba = model.predict_proba(X_test_p)[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    results[name] = {
                        "metrics": self._calculate_metrics(y_test, y_pred, y_pred_proba),
                        "model": model
                    }
                except Exception as e:
                    logging.error(f"Failed to train {name}: {e}")
                    results[name] = {"metrics": {"error": str(e)}, "model": None}
            
            # Select best model based on F1-score
            valid_results = {k: v for k, v in results.items() if "error" not in v["metrics"]}
            if not valid_results:
                return {"error": "All models failed to train. Please check data."}

            best_model_name = max(valid_results, key=lambda name: valid_results[name]["metrics"]["f1_score"])
            best_model_instance = valid_results[best_model_name]["model"]
            
            logging.info(f"🏆 Best model: {best_model_name}")

            # Prepare final output
            churn_drivers = self._get_feature_importances(best_model_instance, features)
            model_comparison_df = pd.DataFrame({model: res["metrics"] for model, res in valid_results.items()}).T.reset_index().rename(columns={"index": "Model"})
            
            return {
                "best_model": best_model_name,
                "model_comparison": model_comparison_df.to_dict('records'),
                "evaluation_metrics": results[best_model_name]["metrics"],
                "churn_drivers": churn_drivers,
                "customers_at_risk": [] # Placeholder
            }

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        """Helper to calculate a standard set of classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba)
        }

    def _get_feature_importances(self, model, features: List[str]) -> List[Dict]:
        """Helper to extract feature importances from a model."""
        if hasattr(model, 'feature_importances_'): # For tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'): # For linear models
            importances = np.abs(model.coef_[0])
        else:
            return []
        
        df = pd.DataFrame({
            'feature': features, 
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        return df.to_dict('records')

class RFMAnalysisTool(BaseTool):
    name: str = "RFM Analysis Tool"
    description: str = "Loads data from a file path and performs RFM analysis to segment customers."

    class RFMAnalysisArgs(BaseModel):
        file_path: str = Field(..., description="Path to the pickle file with engineered RFM features.")

    args_schema: Type[BaseModel] = RFMAnalysisArgs

    def _run(self, file_path: str) -> Dict[str, Any]:
        logging.info("✅ Performing RFM Analysis...")
        try:
            df = pd.read_pickle(file_path)
        except Exception as e:
            return {"error": f"Failed to load DataFrame from {file_path}: {e}"}

        if df.empty:
            return {"error": "DataFrame is empty"}
        required_cols = ['recency_score', 'frequency_score', 'monetary_score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Missing RFM columns: {missing_cols}. Please run feature engineering first."}

        # Find customer ID column
        customer_id_col = None
        for col in df.columns:
            if 'customer' in col.lower() and ('id' in col.lower() or 'key' in col.lower()):
                customer_id_col = col
                break
        if not customer_id_col:
            return {"error": "Could not identify a unique customer ID column in the data."}

        try:
            df_rfm = df.copy()
            df_rfm = df_rfm.dropna(subset=required_cols)
            if df_rfm.empty:
                return {"error": "No valid data after removing NaN values from RFM scores"}

            # Drop duplicates to ensure one record per customer for segmentation
            df_rfm = df_rfm.drop_duplicates(subset=[customer_id_col], keep='first')

            df_rfm['RFM_Score'] = (df_rfm['recency_score'].astype(str) + df_rfm['frequency_score'].astype(str) + df_rfm['monetary_score'].astype(str))
            df_rfm['RFM_Overall'] = (df_rfm['recency_score'].astype(float) + df_rfm['frequency_score'].astype(float) + df_rfm['monetary_score'].astype(float)) / 3
            def rfm_segment(row):
                if row['RFM_Overall'] >= 4.5: return 'Champions'
                elif row['RFM_Overall'] >= 4.0: return 'Loyal Customers'
                elif row['RFM_Overall'] >= 3.5: return 'Potential Loyalists'
                elif row['RFM_Overall'] >= 3.0: return 'New Customers'
                elif row['RFM_Overall'] >= 2.5: return 'At Risk'
                elif row['RFM_Overall'] >= 2.0: return 'Cannot Lose Them'
                else: return 'Lost'
            df_rfm['Customer_Segment'] = df_rfm.apply(rfm_segment, axis=1)

            required_agg_cols = ['customer_total_spent', 'customer_frequency', 'days_since_last_purchase']
            available_agg_cols = [col for col in required_agg_cols if col in df_rfm.columns]
            
            # Use customer_id_col for counting unique customers in each segment
            agg_dict = {customer_id_col: 'count'}
            for col in available_agg_cols:
                agg_dict[col] = 'mean'
            
            segment_analysis = df_rfm.groupby('Customer_Segment').agg(agg_dict).round(2)

            # Rename columns for clarity in the summary report
            column_mapping = { customer_id_col: 'Customer_Count', 'customer_total_spent': 'Avg_Monetary_Value', 'customer_frequency': 'Avg_Frequency', 'days_since_last_purchase': 'Avg_Recency_Days' }
            segment_analysis = segment_analysis.rename(columns=column_mapping)
            
            segment_analysis['Percentage'] = (segment_analysis['Customer_Count'] / len(df_rfm) * 100).round(2)

            output_dir = "rfm_visuals"
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(12, 8))
            segment_counts = df_rfm['Customer_Segment'].value_counts()
            plt.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Customer Segments Distribution')
            plt.savefig(os.path.join(output_dir, 'rfm_segments_pie.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Prepare detailed customer-level data for the second report
            customer_segment_details = df_rfm[[customer_id_col, 'Customer_Segment']]
            
            logging.info("RFM Analysis completed successfully")
            return {
                "segment_analysis": segment_analysis.reset_index().to_dict('records'),
                "total_customers": len(df_rfm),
                "rfm_summary": {
                    "avg_recency": float(df_rfm['recency_score'].mean()),
                    "avg_frequency": float(df_rfm['frequency_score'].mean()),
                    "avg_monetary": float(df_rfm['monetary_score'].mean())
                },
                "customer_segment_details": customer_segment_details.to_dict('records'),
                "customer_id_col_name": customer_id_col # Pass the column name for the report generator
            }
        except Exception as e:
            logging.error(f"RFM Analysis failed: {e}")
            return {"error": f"RFM Analysis failed: {e}"}

class CustomerSegmentationTool(BaseTool):
    name: str = "Customer Segmentation Tool"
    description: str = "Loads data from a file path and performs customer segmentation using K-means clustering."

    class SegmentationArgs(BaseModel):
        file_path: str = Field(..., description="Path to the pickle file with engineered data.")

    args_schema: Type[BaseModel] = SegmentationArgs

    def _run(self, file_path: str) -> Dict[str, Any]:
        logging.info("✅ Performing Customer Segmentation...")
        try:
            df = pd.read_pickle(file_path)
        except Exception as e:
            return {"error": f"Failed to load DataFrame from {file_path}: {e}"}

        if df.empty:
            return {"error": "DataFrame is empty"}
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['churn', 'churn_probability']
        features = [f for f in numeric_features if f not in exclude_cols and not any(keyword in f.lower() for keyword in ['id', 'index'])]
        if len(features) < 2:
            return {"error": "Not enough numeric features for clustering (minimum 2 required)."}
        X = df[features].copy()
        X = X.fillna(X.median())
        X = X.dropna(how='all')
        if X.empty:
            return {"error": "No valid data after cleaning for clustering"}
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        max_clusters = min(10, len(X) // 2)
        if max_clusters < 2:
            return {"error": "Not enough data points for clustering."}
        try:
            silhouette_scores = []
            K_range = range(2, max_clusters + 1)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(score)
            optimal_k = K_range[np.argmax(silhouette_scores)]
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            df_clustered = df.loc[X.index].copy()
            df_clustered['Cluster'] = cluster_labels
            cluster_analysis = df_clustered.groupby('Cluster')[features].mean().round(2)
            cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
            output_dir = "segmentation_visuals"
            os.makedirs(output_dir, exist_ok=True)
            if len(features) >= 2:
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter)
                plt.title(f'Customer Segmentation (K={optimal_k})')
                plt.xlabel(f'{features[0]} (standardized)')
                plt.ylabel(f'{features[1]} (standardized)')
                plt.savefig(os.path.join(output_dir, 'customer_segments_2d.png'), dpi=300, bbox_inches='tight')
                plt.close()
            logging.info(f"Customer Segmentation completed with {optimal_k} clusters")
            return { "optimal_clusters": int(optimal_k), "cluster_analysis": cluster_analysis.reset_index().to_dict('records'), "cluster_sizes": cluster_sizes.to_dict(), "silhouette_scores": dict(zip(K_range, silhouette_scores)), "best_silhouette_score": float(max(silhouette_scores)) }
        except Exception as e:
            logging.error(f"Customer Segmentation failed: {e}")
            return {"error": f"Customer Segmentation failed: {e}"}

# --- CORRECTED MarketBasketAnalysisTool ---
class MarketBasketAnalysisTool(BaseTool):
    name: str = "Market Basket Analysis Tool"
    description: str = "Loads transaction data from a file path and performs market basket analysis."

    class MarketBasketArgs(BaseModel):
        file_path: str = Field(..., description="Path to the pickle file with transaction data.")
        transaction_col: str = Field(..., description="Transaction ID column.")
        item_col: str = Field(..., description="Item/Product column.")

    args_schema: Type[BaseModel] = MarketBasketArgs

    def _run(self, file_path: str, transaction_col: str, item_col: str) -> Dict[str, Any]:
        logging.info("✅ Performing Market Basket Analysis...")
        try:
            df = pd.read_pickle(file_path)
        except Exception as e:
            return {"error": f"Failed to load DataFrame from {file_path}: {e}"}

        if df.empty:
            return {"error": "DataFrame is empty"}
        
        def find_actual_column_name(df_columns, requested_name):
            """Finds the actual column name in a case-insensitive way."""
            if not requested_name:
                raise ValueError("A required column name was not provided.")
            for col in df_columns:
                if col.lower().strip() == requested_name.lower().strip():
                    return col
            raise ValueError(f"Required column '{requested_name}' not found. Available columns: {list(df_columns)}")

        try:
            actual_transaction_col = find_actual_column_name(df.columns, transaction_col)
            actual_item_col = find_actual_column_name(df.columns, item_col)

            if 'quantity' not in df.columns:
                df['quantity'] = 1
            basket = df.groupby([actual_transaction_col, actual_item_col])['quantity'].sum().unstack().reset_index().fillna(0).set_index(actual_transaction_col)
            
            def encode_units(x):
                if x <= 0: return 0
                if x >= 1: return 1
            basket_sets = basket.applymap(encode_units)
            if basket_sets.empty:
                return {"error": "No transaction data available for analysis"}

            min_support = 0.01
            frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
            if frequent_itemsets.empty:
                return {"error": "Not enough frequent items for association analysis with current support."}
            
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            if rules.empty:
                return {"error": "No significant association rules found."}
            
            rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
            frequent_items_dict = frequent_itemsets.sort_values('support', ascending=False).head(20).copy()
            associations = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20).copy()
            associations['antecedents'] = associations['antecedents'].apply(lambda a: ', '.join(list(a)))
            associations['consequents'] = associations['consequents'].apply(lambda a: ', '.join(list(a)))
            frequent_items_dict['itemsets'] = frequent_items_dict['itemsets'].apply(lambda a: ', '.join(list(a)))
            
            logging.info(f"Market Basket Analysis completed with {len(rules)} associations found")
            return { "frequent_items": frequent_items_dict.to_dict('records'), "associations": associations.to_dict('records'), "total_transactions": len(basket_sets), "unique_items": len(basket_sets.columns) }
        except (Exception, ValueError) as e:
            logging.error(f"Market Basket Analysis failed: {e}")
            return {"error": f"Market Basket Analysis failed: {str(e)}"}


# --- Agent Definitions ---
# The agents now benefit from the more powerful LangChain LLM and will use ReAct-style reasoning.
# The `verbose=True` flag in the Crew will show the "Thought-Action-Observation" loop.

def create_data_analyst_agent():
    return Agent(
        role="Senior Data Analyst",
        goal=(
            "Rigorously analyze database schemas, and then load, preprocess, and explore the most relevant data. "
            "Your final output is a clean dataset file and a comprehensive EDA report."
        ),
        backstory=(
            "You are an expert data analyst with a keen eye for detail. You follow a strict process: "
            "first, understand the data landscape (schema), then efficiently load and clean the necessary data, "
            "saving the output to a file. Finally, you perform a thorough exploratory data analysis on that file "
            "to uncover initial insights. Your process ensures that all downstream tasks are built on a solid foundation."
        ),
        tools=[SchemaAnalyzerTool(), LoadAndPreprocessDataTool(), EDAExplorationTool()],
        verbose=True,
        llm=llm
    )


def create_feature_engineer_agent():
    return Agent(
        role="Feature Engineering Specialist",
        goal="Load a cleaned data file and engineer insightful features for advanced analytical models, then save the result to a new file.",
        backstory=(
            "You are a master feature engineer. You specialize in transforming raw, clean data into powerful, predictive features. "
            "You can create time-based features, customer-level aggregations, and RFM scores. "
            "Your work is the crucial link between clean data and high-performance models. "
            "You receive a file path and your output is another file path to the enriched dataset."
        ),
        tools=[FeatureEngineeringTool()],
        verbose=True,
        llm=llm
    )


def create_customer_analytics_agent():
    return Agent(
        role="Customer Analytics Expert",
        goal="Load an engineered dataset and execute advanced customer analytics like churn prediction, RFM analysis, or segmentation.",
        backstory=(
            "You are a customer analytics guru. You live and breathe customer data. "
            "Using a pre-engineered dataset from a file, you apply sophisticated models to predict customer churn, "
            "create detailed RFM segments, and perform cluster analysis to uncover hidden customer personas. "
            "Your insights are critical for targeted marketing and customer retention strategies."
        ),
        tools=[ChurnPredictionTool(), RFMAnalysisTool(), CustomerSegmentationTool()],
        verbose=True,
        llm=llm
    )


def create_marketing_analyst_agent():
    return Agent(
        role="Marketing Analytics Specialist",
        goal="Load transaction data from a file to find product associations and generate strategic marketing recommendations.",
        backstory=(
            "You are a marketing analytics specialist with a knack for finding hidden patterns in purchase data. "
            "You use Market Basket Analysis to understand which products are bought together, "
            "providing the business with a clear list of association rules to drive cross-selling and promotional strategies."
        ),
        tools=[MarketBasketAnalysisTool()],
        verbose=True,
        llm=llm
    )


def create_business_strategist_agent():
    return Agent(
        role="Business Strategy Advisor",
        goal=(
            "Synthesize all analytical findings into a clear, actionable, and structured business report. "
            "Your output MUST be a JSON object that follows a predefined technical report template."
        ),
        backstory=(
            "You are a senior business strategist who bridges the gap between data and decisions. "
            "You excel at translating complex analytical outputs (like EDA, model results, and segment profiles) "
            "into a coherent, structured narrative. You are tasked with populating a detailed technical report template "
            "in JSON format, ensuring all sections are filled with practical, impactful, and data-driven insights."
        ),

        tools=[],
        verbose=True,
        llm=llm,
        allow_delegation=False,
    )

# --- NEW RAG-Powered Q&A Agent ---
def create_rag_qa_agent(memory):
    """
    Creates a RAG-powered agent for answering questions about the analysis.
    This agent uses memory to be aware of the ongoing conversation.
    """
    return Agent(
        role="Analysis Q&A Specialist",
        goal=(
            "Answer user questions accurately by retrieving relevant information from the final analysis report. "
            "Use the conversation history to understand the context of the questions."
        ),
        backstory=(
            "You are an AI assistant specializing in explaining data analysis results. "
            "You are provided with a comprehensive analysis report and must use it as your single source of truth. "
            "You will retrieve relevant sections from the report to construct your answers. "
            "If the answer is not in the report, you must clearly state that. "
            "You are aware of the past few questions and answers to handle follow-up questions gracefully."
        ),
        tools=[],  # RAG is handled by a LangChain chain, not a crewAI tool
        memory=memory,  # Equip the agent with conversational memory
        verbose=True,
        llm=llm
    )

# --- New Function to Load Analysis Mappings from Excel ---
def load_analysis_mappings_from_excel(file_path: str) -> Dict[str, Any]:
    """
    Loads and parses the question mapping Excel file to create a structured
    dictionary for driving the analysis process.
    """
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        logging.error(f"FATAL: The question mapping file was not found at {file_path}")
        print(f"Error: The question mapping file '{file_path}' was not found.")
        return {}
    except Exception as e:
        logging.error(f"FATAL: Could not read the Excel file at {file_path}. Error: {e}")
        print(f"Error: Could not read the Excel file '{file_path}'. It might be corrupted or in an unsupported format.")
        return {}

    df.columns = [str(col).strip() for col in df.columns]
    
    q_col = 'Customer Value Questions:'
    data_col = 'DATA REQUIRED'
    approach_col = 'Approach/model used'
    clarification_col = 'questions asked'

    if not all(col in df.columns for col in [q_col, data_col, approach_col, clarification_col]):
        print(f"Error: The Excel file must contain the columns: '{q_col}', '{data_col}', '{approach_col}', '{clarification_col}'")
        return {}

    df.dropna(how='all', inplace=True)
    df = df[~df[q_col].str.contains(':', na=False)] 
    df.dropna(subset=[clarification_col, q_col], inplace=True) 
    df.reset_index(drop=True, inplace=True)

    analysis_mappings = {}
    for _, row in df.iterrows():
        question = row[q_col].strip()
        approach = str(row[approach_col]).lower() if pd.notna(row[approach_col]) else ""
        
        internal_type = "General"
        if 'rfm' in approach: internal_type = 'RFM'
        elif 'churn' in approach or 'classification' in approach or 'predictive' in approach: internal_type = 'Churn'
        elif 'cluster' in approach or 'segment' in approach: internal_type = 'Segmentation'
        elif 'basket' in approach or 'apriori' in approach: internal_type = 'Market_Basket'

        required_cols = [col.strip() for col in str(row[data_col]).split(',') if col.strip()] if pd.notna(row[data_col]) else []
        clarification_qs = [q.strip() for q in str(row[clarification_col]).split('\n') if q.strip()] if pd.notna(row[clarification_col]) else []
        
        if not clarification_qs: 
            continue

        analysis_mappings[question] = { "description": question, "internal_name": internal_type, "required_columns": required_cols, "questions": clarification_qs }
        
    return analysis_mappings


# Main Analysis Orchestrator
class MarketingAnalyticsOrchestrator:
    def __init__(self, analysis_mappings):
        # --- Memory Integration ---
        # Using LangChain's memory module to give the orchestrator context.
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
        
        self.agents = {
            "data_analyst": create_data_analyst_agent(),
            "feature_engineer": create_feature_engineer_agent(),
            "customer_analytics": create_customer_analytics_agent(),
            "marketing_analyst": create_marketing_analyst_agent(),
            "business_strategist": create_business_strategist_agent(),
            "rag_qa_agent": create_rag_qa_agent(self.memory) # New agent for RAG Q&A
        }
        self.clarification_tool = ClarificationTool()
        self.conversation_history = []
        self.schema_analyzer_tool = SchemaAnalyzerTool()
        self.analysis_mappings = analysis_mappings


    def present_database_schema(self):
        """Analyzes and presents the database schema to the user."""
        schema_info = self.schema_analyzer_tool._run(database_url=connection_string)
        return schema_info # Return the raw dictionary

    def get_next_clarifying_question(self, user_query: str, analysis_info: Dict[str, Any]) -> str:
        """Gets the next single clarifying question."""
        analysis_type = analysis_info['internal_name']
        questions = analysis_info["questions"]
        question_guide = "\n".join(questions)
        
        context = self.clarification_tool._run(
            query=user_query, analysis_type=analysis_type,
            history=self.conversation_history, question_guide=question_guide
        )
        
        temp_agent = Agent(
            role="Clarification Assistant", goal="Generate the next logical clarifying question",
            backstory="You are a helpful assistant that asks clarifying questions.",
            tools=[], verbose=False, llm=llm
        )
        
        question_task = Task(
            description=context, agent=temp_agent,
            expected_output="A single clarifying question or the phrase 'I don't have any more questions.'"
        )
        
        crew = Crew(agents=[temp_agent], tasks=[question_task], verbose=False, process=Process.sequential)
        
        result = crew.kickoff()
        question = str(result.tasks_output[0].raw).strip()
        
        return question

    def run_analysis(self, user_query: str, analysis_info: Dict[str, Any], db_schema: Dict[str, Any], analysis_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Main analysis execution with detailed result capture"""
        if analysis_params is None:
            analysis_params = {}
            
        analysis_type = analysis_info['internal_name']
        required_columns = analysis_info['required_columns']
        
        os.makedirs("analysis_temp_data", exist_ok=True)

        detailed_results = {
            "success": False, "analysis_type": analysis_type, "user_query": user_query,
            "conversation_history": self.conversation_history, "execution_steps": {},
            "data_summary": {}, "analysis_results": {}, "recommendations": {},
            "technical_details": {}, "visualizations": [],
            "timestamps": { "start_time": datetime.now().isoformat(), "end_time": None },
            "intermediate_files": {}
        }
        
        try:
            # Map required columns to specific roles to guide the feature engineering agent.
            customer_col_name = next((c for c in required_columns if 'customer' in c.lower() or 'id' in c.lower()), "customer id")
            date_col_name = next((c for c in required_columns if 'date' in c.lower()), "order date")
            monetary_col_name = next((c for c in required_columns if 'sales' in c.lower() or 'monetary' in c.lower() or 'price' in c.lower() or 'revenue' in c.lower()), "sales")

            # --- TASK FLOW with file-based handoff ---
            
            data_preprocessing_task = Task(
                description=(
                    f"You are analyzing: '{user_query}'.\n"
                    "1. Use the 'Schema Analyzer Tool' to understand the available database tables.\n"
                    f"2. From the schema, identify the single most relevant table for a '{analysis_type}' analysis which is likely to contain these columns: {required_columns}.\n"
                    f"3. Use the 'Load and Preprocess Data Tool' with the identified `table_name` and `required_columns`: {required_columns}. "
                    "This tool saves the cleaned data to a file and returns its path."
                ),
                agent=self.agents["data_analyst"],
                expected_output="A string containing the file path to the saved, cleaned pandas DataFrame in pickle format."
            )

            eda_task = Task(
                description=(
                    "Take the file path for the cleaned data from the previous task's output. "
                    "Use the 'EDA Exploration Tool' to perform a comprehensive exploratory data analysis. "
                    "You MUST pass the file path string from the context to the `file_path` argument of the tool. "
                    "Also, infer the original table name from the file path for labeling."
                ),
                agent=self.agents["data_analyst"],
                context=[data_preprocessing_task],
                expected_output="A dictionary of EDA results, including statistics and correlations."
            )
            
            binning_specs = [{'column_name': 'customer_total_spent', 'bins': 4}]
            encoding_specs = ['some_categorical_column']

            feature_task = Task(
                description=(
                    f"Take the file path for the cleaned data from the 'data_preprocessing_task' output. "
                    f"Use the 'FeatureEngineeringTool' to create features for a '{analysis_type}' analysis. "
                    f"You MUST pass this file path to the `file_path` argument. "
                    "To create the necessary features, you MUST provide the correct column names to the tool. "
                    f"Based on the analysis requirements ({required_columns}), use the following parameters:\n"
                    f"- `customer_col`: '{customer_col_name}'\n"
                    f"- `date_col`: '{date_col_name}'\n"
                    f"- `monetary_col`: '{monetary_col_name}'\n"
                    f"- `columns_to_bin`: {binning_specs}\n"
                    f"- `columns_to_encode`: {encoding_specs}\n"
                    "The tool will save the engineered DataFrame to a new file and return its path."
                ),
                agent=self.agents["feature_engineer"],
                context=[data_preprocessing_task],
                expected_output="A string containing the file path to the NEW saved, feature-engineered DataFrame."
            )
            
            analysis_tasks = []
            
            core_task_context = [feature_task]
            if analysis_type == "Market_Basket":
                core_task_context = [data_preprocessing_task]

            if analysis_type == "RFM":
                rfm_task = Task(
                    description=(
                        "Take the file path for the engineered DataFrame from the previous task's output. "
                        "Perform RFM analysis using the 'RFMAnalysisTool'. You MUST pass the file path string "
                        "from the context to the `file_path` argument of the tool."
                    ),
                    agent=self.agents["customer_analytics"], context=core_task_context,
                    expected_output="A dictionary of RFM analysis results with customer segments, characteristics, and detailed customer-level segment data."
                )
                analysis_tasks.append(rfm_task)
            
            elif analysis_type == "Churn":
                selected_models = analysis_params.get("selected_churn_models")
                churn_task = Task(
                    description=(
                        "Take the file path for the engineered DataFrame from the previous task's output. "
                        "Build a churn prediction model using the 'ChurnPredictionTool'. You MUST pass the file path string "
                        f"from the context to the `file_path` argument. Also pass `models_to_run`: {selected_models}."
                    ),
                    agent=self.agents["customer_analytics"], context=core_task_context,
                    expected_output="A dictionary of churn prediction results with model comparisons, the best model's metrics, drivers, and at-risk customers."
                )
                analysis_tasks.append(churn_task)
            
            elif analysis_type == "Segmentation":
                segmentation_task = Task(
                    description=(
                        "Take the file path for the engineered DataFrame from the previous task's output. "
                        "Perform customer segmentation using the 'CustomerSegmentationTool'. You MUST pass the file path string "
                        "from the context to the `file_path` argument of the tool."
                    ),
                    agent=self.agents["customer_analytics"], context=core_task_context,
                    expected_output="A dictionary of customer segmentation results with cluster analysis."
                )
                analysis_tasks.append(segmentation_task)
            
            elif analysis_type == "Market_Basket":
                transaction_col, item_col = "order_id", "product_name" 
                for item in self.conversation_history:
                    if "transaction id" in item["question"].lower(): transaction_col = item["answer"]
                    if "item or product" in item["question"].lower(): item_col = item["answer"]

                basket_task = Task(
                    description=(
                        "Take the file path for the preprocessed data from the initial data loading task's output. "
                        "Perform market basket analysis using the 'MarketBasketAnalysisTool'. "
                        f"You MUST pass the file path to the `file_path` argument, use '{transaction_col}' for `transaction_col`, and '{item_col}' for `item_col`."
                    ),
                    agent=self.agents["marketing_analyst"], context=core_task_context,
                    expected_output="A dictionary of market basket analysis results with product associations."
                )
                analysis_tasks.append(basket_task)
            
            strategy_context = [eda_task, feature_task] + analysis_tasks
            strategy_task = Task(
                description=(
                    f"You are creating a detailed technical report for the business query: '{user_query}'. "
                    "Synthesize all analytical findings from the prior EDA, feature engineering, and core analysis tasks. "
                    "Your response MUST be a valid JSON object only, without any markdown formatting, comments, or other text before or after the JSON. "
                    "Adhere strictly to the following structure. Fill each key with detailed, well-written markdown content. "
                    "If a section is not applicable, state that clearly.\n"
                    "{\n"
                    "  \"executive_summary\": {\n"
                    "    \"problem_statement\": \"...\",\n"
                    "    \"key_result\": \"...\",\n"
                    "    \"recommendation\": \"...\"\n"
                    "  },\n"
                    "  \"business_objective\": {\n"
                    "    \"problem_context\": \"...\",\n"
                    "    \"goal\": \"...\",\n"
                    "    \"success_metrics\": {\n"
                    "       \"technical\": \"e.g., Model F1-score > 0.75\",\n"
                    "       \"business\": \"e.g., Reduce churn by 5% in next quarter\"\n"
                    "     },\n"
                    "    \"constraints\": \"...\"\n"
                    "  },\n"
                    "  \"modeling_interpretation\": {\n"
                    "    \"feature_importance\": \"Interpret the feature importance results from the model...\",\n"
                    "    \"model_explainability\": \"Briefly explain the importance of model explainability like SHAP/LIME, even if not explicitly calculated in this run...\"\n"
                    "  },\n"
                    "  \"conclusion_recommendations\": {\n"
                    "    \"conclusion\": \"Summarize the project's findings.\",\n"
                    "    \"business_impact_analysis\": \"Analyze the potential business impact of implementing your recommendations.\",\n"
                    "    \"limitations_risks\": \"Discuss limitations of the analysis and potential risks.\",\n"
                    "    \"deployment_next_steps\": \"Suggest concrete next steps, e.g., 'Deploy model as API', 'Launch A/B test for marketing campaigns.'\"\n"
                    "  }\n"
                    "}"
                ),
                agent=self.agents["business_strategist"], context=strategy_context,
                expected_output="A raw and valid JSON string containing the structured business report with markdown content."
            )
            
            all_tasks = [data_preprocessing_task, eda_task, feature_task] + analysis_tasks + [strategy_task]
            
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=all_tasks, verbose=True, process=Process.sequential
            )
            
            
            crew_result = crew.kickoff()

            # --- CORRECTED RESULT PARSING ---
            # Use the helper function to convert string outputs to dictionaries
            eda_raw_output = eda_task.output.raw if eda_task.output else "{}"
            eda_result_dict = parse_agent_output(eda_raw_output)

            core_analysis_output = analysis_tasks[-1].output if analysis_tasks and analysis_tasks[-1].output else None
            core_analysis_raw = core_analysis_output.raw if core_analysis_output and hasattr(core_analysis_output, 'raw') else "{}"
            core_analysis_result_dict = parse_agent_output(core_analysis_raw)

            cleaned_data_path = data_preprocessing_task.output.raw if data_preprocessing_task.output else "Not generated"
            engineered_data_path = feature_task.output.raw if feature_task.output else "Not generated"

            # The strategist's output is also parsed
            recommendations_raw_str = crew_result.raw
            recommendations_dict = parse_agent_output(recommendations_raw_str)
            if 'error' in recommendations_dict:
                logging.error(f"Failed to parse JSON from strategist: {recommendations_raw_str}")
                # Fallback for malformed strategist output
                recommendations_dict = {"error": "Failed to parse JSON from strategist.", "raw_output": recommendations_raw_str}


            detailed_results.update({
                "success": True,
                "execution_steps": {
                    "data_load_and_preprocess": data_preprocessing_task.output.raw,
                    "exploratory_analysis": eda_result_dict,  # Use the parsed dictionary
                    "feature_engineering": feature_task.output.raw,
                    "core_analysis": core_analysis_result_dict, # Use the parsed dictionary
                },
                "recommendations": recommendations_dict,
                "analysis_results": core_analysis_result_dict, # Use the parsed dictionary
                "intermediate_files": {
                    "cleaned_data": cleaned_data_path,
                    "engineered_data": engineered_data_path,
                },
                "technical_details": { "database_connection": "Azure SQL Database", "analysis_method": f"{analysis_type} Analysis", "tools_used": ["CrewAI", "scikit-learn", "pandas", "mlxtend", "LangChain"], },
                "timestamps": { "start_time": detailed_results["timestamps"]["start_time"], "end_time": datetime.now().isoformat() }
            })
            
            return detailed_results
            
        except Exception as e:
            logging.error(f"Analysis orchestration failed: {e}", exc_info=True)
            detailed_results.update({
                "success": False, "error": str(e),
                "timestamps": { "start_time": detailed_results["timestamps"]["start_time"], "end_time": datetime.now().isoformat() }
            })
            return detailed_results


# --- UPDATED ReportGenerator to handle new structure ---
class ReportGenerator:
    def __init__(self):
        self.output_dir = "analysis_reports"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_json_report(self, analysis_results: Dict[str, Any], timestamp: str) -> str:
        json_filename = f"{analysis_results['analysis_type']}_analysis_{timestamp}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        serializable_results = make_json_serializable(analysis_results)
        serializable_results = sanitize_dict_keys(serializable_results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        return json_path
    
    def generate_markdown_report(self, analysis_results: Dict[str, Any], timestamp: str) -> str:
        """
        Generates the markdown report content, saves it to a file, 
        and returns the raw markdown string for direct use in Streamlit.
        """
        analysis_type = analysis_results['analysis_type']
        md_filename = f"{analysis_type}_analysis_report_{timestamp}.md"
        md_path = os.path.join(self.output_dir, md_filename)
        
        # 1. Generate the content
        md_content = self._generate_markdown_content(analysis_results, timestamp)
        
        # 2. Save the content to a file
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        # 3. Return the content itself (This is the fix)
        return md_content
    
    def generate_detailed_rfm_report(self, analysis_results: Dict[str, Any], timestamp: str) -> str:
        """Generates a detailed markdown report of customers within each RFM segment."""
        md_filename = f"RFM_Customer_Details_Report_{timestamp}.md"
        md_path = os.path.join(self.output_dir, md_filename)
        
        core_results = analysis_results.get('analysis_results', {})
        
        if isinstance(core_results, str):
            dict_start = core_results.find('{')
            dict_end = core_results.rfind('}') + 1
            if dict_start != -1 and dict_end > dict_start:
                dict_str = core_results[dict_start:dict_end]
                try:
                    import ast
                    core_results = ast.literal_eval(dict_str)
                except (ValueError, SyntaxError):
                    core_results = {}
            else:
                core_results = {}

        customer_details = core_results.get('customer_segment_details')
        customer_id_col = core_results.get('customer_id_col_name', 'Customer ID')

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# RFM Customer-Level Detail Report\n")
            f.write(f"**Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if not customer_details:
                f.write("No customer-level segment data was found in the analysis results.\n")
                return md_path
            
            df = pd.DataFrame(customer_details)
            if df.empty:
                f.write("The customer detail data is empty.\n")
                return md_path
                
            f.write("This report lists the individual customers belonging to each RFM segment.\n\n")
            
            grouped = df.groupby('Customer_Segment')
            
            for segment_name, segment_df in grouped:
                f.write(f"## Segment: {segment_name}\n\n")
                f.write(f"**Total Customers in this segment:** {len(segment_df)}\n\n")
                
                display_df = segment_df[[customer_id_col]].rename(columns={customer_id_col: 'Customer ID'})
                f.write(dataframe_to_markdown(display_df, max_rows=len(display_df)))
                f.write("\n---\n\n")
                
        return md_path
    def _generate_markdown_content(self, results: Dict[str, Any], timestamp: str) -> str:
        analysis_type = results['analysis_type']
        recommendations = results.get('recommendations', {})
        
        def get_rec(key_path, default="Not provided."):
            keys = key_path.split('.')
            value = recommendations
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key, default)
                else:
                    return default
            return value if value else default

        md_content = f"""# {analysis_type} Analysis: Technical Report
**Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Status:** {'✅ Success' if results['success'] else '❌ Failed'}
---
"""
        if not results['success']:
            md_content += f"""## ❌ Analysis Failure Details
**An error occurred during the analysis process.**
**Error Message:**
{results.get('error', 'Unknown error')}
"""
            return md_content

        # 1. Executive Summary
        md_content += "## 1. 📌 Executive Summary (TL;DR)\n"
        md_content += f"- **Problem Statement:** {get_rec('executive_summary.problem_statement')}\n"
        md_content += f"- **Key Result:** {get_rec('executive_summary.key_result')}\n"
        md_content += f"- **Recommendation:** {get_rec('executive_summary.recommendation')}\n\n"

        # 2. Business Objective
        md_content += "## 2. 🎯 Business Objective & Success Criteria\n"
        md_content += f"### Problem Context\n{get_rec('business_objective.problem_context')}\n\n"
        md_content += f"### Goal\n{get_rec('business_objective.goal')}\n\n"
        md_content += "### Success Metrics\n"
        md_content += f"- **Technical:** {get_rec('business_objective.success_metrics.technical')}\n"
        md_content += f"- **Business:** {get_rec('business_objective.success_metrics.business')}\n\n"
        md_content += f"### Constraints\n{get_rec('business_objective.constraints')}\n\n"

        # 3. Data Exploration & Feature Engineering
        md_content += "## 3. 📊 Data Exploration & Feature Engineering\n"
        
        eda_results = results.get('execution_steps', {}).get('exploratory_analysis', {})
        if not isinstance(eda_results, dict):
            eda_results = {}
        
        md_content += "### Data Sources & Preprocessing\n"
        cleaned_data_path = results.get('intermediate_files', {}).get('cleaned_data', '')
        table_name_match = re.search(r'([a-zA-Z0-9]+)_cleaned_', cleaned_data_path)
        db_table = table_name_match.group(1) if table_name_match else 'relevant'
        md_content += f"- The analysis utilized the `{db_table}` table from the database.\n"
        md_content += "- Preprocessing steps included handling missing values, removing duplicates, and selecting relevant columns.\n\n"
        
        md_content += "### Exploratory Data Analysis (EDA)\n"
        md_content += "Key initial findings from the data include:\n\n"
        
        # --- Display Basic Statistics ---
        stats_data = eda_results.get('basic_stats', {})
        if stats_data:
            stats_df = pd.DataFrame(stats_data).round(2)
            if not stats_df.empty:
                md_content += "#### Basic Statistics for Numeric Columns\n"
                # Reset the index to turn statistic names (e.g., 'count', 'mean') into a column
                stats_df = stats_df.reset_index()
                # Rename the new column for better readability in the report
                stats_df = stats_df.rename(columns={'index': 'Statistic'})
                md_content += dataframe_to_markdown(stats_df) + "\n"
        
        # --- Display Categorical Analysis ---
        categorical_data = eda_results.get('categorical_analysis', {})
        if categorical_data:
            md_content += "#### Categorical Data Insights\n"
            for col, values in categorical_data.items():
                if values:
                    md_content += f"**Top Values for `{col}`:**\n"
                    cat_df = pd.DataFrame(list(values.items()), columns=[col, 'Count'])
                    md_content += dataframe_to_markdown(cat_df) + "\n"
        
        # --- Display Missing Values ---
        missing_data = eda_results.get('missing_values', {})
        if missing_data:
            md_content += "#### Missing Values\n"
            missing_df = pd.DataFrame(list(missing_data.items()), columns=['Column', 'Missing Count'])
            md_content += dataframe_to_markdown(missing_df) + "\n"
        else:
            md_content += "**Missing Values:** No missing values were found in the key columns after initial cleaning.\n\n"

        md_content += "*A comprehensive, interactive data profile report is available for download from the sidebar. All generated charts and plots are displayed below this report.*\n\n"

        md_content += "### Feature Engineering\n"
        engineered_data_path = results.get("intermediate_files", {}).get("engineered_data")
        if engineered_data_path and cleaned_data_path and os.path.exists(engineered_data_path) and os.path.exists(cleaned_data_path):
            try:
                df_engineered = pd.read_pickle(engineered_data_path)
                df_cleaned = pd.read_pickle(cleaned_data_path)
                engineered_cols = set(df_engineered.columns)
                original_cols = set(df_cleaned.columns)
                new_features = engineered_cols - original_cols
                
                if new_features:
                    md_content += "New features were created to enhance model performance, including:\n"
                    for feature in sorted(list(new_features)):
                        md_content += f"- `{feature}`\n"
                    md_content += "\n"
                else:
                    md_content += "No new feature columns were added, but transformations like scaling or encoding may have been applied.\n\n"
            except Exception:
                md_content += "Feature engineering steps like creating time-based features, customer-level aggregations, and RFM scores were applied where appropriate.\n\n"
        else:
            md_content += "New features were created to enhance model performance, including time-based features, customer aggregations, and RFM scores where applicable.\n\n"

        # 4. Modeling & Validation (Core Analysis Results)
        md_content += "## 4. 🤖 Modeling & Validation\n"
        analysis_results_dict = results.get('analysis_results', {})
        md_content += self._generate_core_analysis_section(analysis_type, analysis_results_dict)

        # 5. Results & Interpretation from Strategist
        md_content += "## 5. 📈 Results & Interpretation\n"
        md_content += f"### Feature Importance\n{get_rec('modeling_interpretation.feature_importance')}\n\n"
        md_content += f"### Model Explainability (SHAP / LIME)\n{get_rec('modeling_interpretation.model_explainability')}\n\n"

        # 6. Conclusion & Recommendations from Strategist
        md_content += "## 6. ✅ Conclusion & Recommendations\n"
        md_content += f"### Conclusion\n{get_rec('conclusion_recommendations.conclusion')}\n\n"
        md_content += f"### Business Impact Analysis\n{get_rec('conclusion_recommendations.business_impact_analysis')}\n\n"
        md_content += f"### Limitations & Risks\n{get_rec('conclusion_recommendations.limitations_risks')}\n\n"
        md_content += f"### Deployment & Next Steps\n{get_rec('conclusion_recommendations.deployment_next_steps')}\n\n"
        
        md_content += self._generate_technical_details_section(results)
        return md_content
        
    def _generate_core_analysis_section(self, analysis_type, results_dict):
        """Generates the specific 'Modeling & Validation' content based on analysis type."""
        if not results_dict or isinstance(results_dict, str):
            return "Core analysis results were not properly generated.\n"

        content = ""
        if analysis_type == "RFM":
            content += "### RFM Segmentation\n"
            content += f"**Total Customers Analyzed:** {results_dict.get('total_customers', 'N/A')}\n"
            content += "The following table shows the profile of each customer segment based on average Recency, Frequency, and Monetary values.\n\n"
            segment_df = pd.DataFrame(results_dict.get('segment_analysis', []))
            content += dataframe_to_markdown(segment_df) if not segment_df.empty else "No segment data available.\n"
            # REMOVED hardcoded image link

        elif analysis_type == "Churn":
            content += "### Churn Prediction Modeling\n"
            content += "**Model Candidates & Validation:**\n"
            comparison_df = pd.DataFrame(results_dict.get('model_comparison', []))
            if not comparison_df.empty:
                content += "Multiple models were trained and evaluated. The best model was selected based on the F1-Score.\n\n"
                for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                    if col in comparison_df.columns:
                        comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.4f}")
                content += dataframe_to_markdown(comparison_df)
            content += f"\n\n**🏆 Best Performing Model:** **{results_dict.get('best_model', 'N/A')}**\n\n"
            content += "### Top Churn Drivers (Feature Importances)\nThis table shows the factors that most significantly influence churn prediction.\n\n"
            drivers_df = pd.DataFrame(results_dict.get('churn_drivers', []))
            content += dataframe_to_markdown(drivers_df) if not drivers_df.empty else "No churn driver data available.\n"

        elif analysis_type == "Segmentation":
            content += "### K-Means Clustering\n"
            content += f"- **Optimal Number of Clusters (K):** {results_dict.get('optimal_clusters', 'N/A')} (determined by Silhouette Score)\n"
            content += f"- **Best Silhouette Score:** {results_dict.get('best_silhouette_score', 0):.4f}\n\n"
            content += "### Segment Profiles (Cluster Centroids)\n"
            analysis_df = pd.DataFrame(results_dict.get('cluster_analysis', []))
            content += dataframe_to_markdown(analysis_df) if not analysis_df.empty else "No cluster profile data available.\n"
            # REMOVED hardcoded image link

        elif analysis_type == "Market_Basket":
            content += "### Apriori Algorithm for Association Rules\n"
            content += f"- **Total Transactions Analyzed:** {results_dict.get('total_transactions', 'N/A')}\n"
            content += f"- **Total Unique Items:** {results_dict.get('unique_items', 'N/A')}\n\n"
            content += "### Top 20 Product Association Rules (sorted by Lift)\n"
            assoc_df = pd.DataFrame(results_dict.get('associations', []))
            content += dataframe_to_markdown(assoc_df) if not assoc_df.empty else "No association rules were found.\n"

        return content + "\n"

    def _generate_technical_details_section(self, results: Dict[str, Any]) -> str:
        tech_details = results.get('technical_details', {})
        return f"""---
## Technical Details
- **Primary Database:** {tech_details.get('database_connection', 'N/A')}
- **Analysis Method:** {tech_details.get('analysis_method', 'N/A')}
- **Key Python Libraries:** {', '.join(tech_details.get('tools_used', []))}
*This report was generated automatically by the Marketing Analytics AI Assistant.*
"""

# --- RAG-Powered Post-Analysis Q&A Chain ---
def create_rag_chain(analysis_results: Dict[str, Any]):
    """
    Creates a RAG chain for Q&A about the analysis results.
    """
    # 1. Create the knowledge base
    report_context = json.dumps(make_json_serializable(analysis_results), indent=2)
    
    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([report_context])
    
    # 3. Create embeddings and a vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector = FAISS.from_documents(docs, embeddings)
        retriever = vector.as_retriever()
    except Exception as e:
        logging.error(f"Could not create vector store for RAG: {e}")
        return None

    # 4. Define the RAG chain
    # FIX: Instantiate a LangChain-compatible model for the chain
    rag_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                     temperature=0.1, # Use a low temperature for factual Q&A
                                     api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = PromptTemplate(
        template="""
        Answer the following question based only on the provided context.
        If the answer is not in the context, say "I cannot find this information in the analysis report."
        Be concise and helpful.

        <context>
        {context}
        </context>

        Question: {input}
        """,
        input_variables=["context", "input"],
    )

    # Use the new LangChain-native LLM instance here
    document_chain = create_stuff_documents_chain(rag_llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
