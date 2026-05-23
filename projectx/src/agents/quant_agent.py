import os
import re
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("QuantAnalystAgent")

class QuantAnalystAgent:
    """
    QuantAnalystAgent ingests structured financial metrics in CSV format,
    loads them into a pandas DataFrame, parses natural language queries for
    quantitative components (tickers, quarters, metrics, aggregates), and
    performs the corresponding numerical data aggregation or slicing.
    """

    def __init__(self, data_path: str = "data/financial_metrics.csv") -> None:
        self.data_path: str = data_path
        self.df: Optional[pd.DataFrame] = None
        self.tickers: List[str] = []
        self.quarters: List[str] = []
        
        # Load database on initialization
        self.load_data()

    def load_data(self) -> None:
        """
        Safely loads corporate financial data from the configured path
        handling missing files and format exceptions.
        """
        if not os.path.exists(self.data_path):
            logger.error(f"Structured metrics file not found at: {self.data_path}")
            self.df = pd.DataFrame()
            return

        try:
            # Load CSV into DataFrame
            self.df = pd.read_csv(self.data_path)
            # Normalize column names to lowercase and strip whitespaces
            self.df.columns = [col.strip().lower() for col in self.df.columns]
            
            # Extract unique tickers and quarters for mapping
            if "ticker" in self.df.columns:
                self.tickers = [t.upper() for t in self.df["ticker"].dropna().unique()]
            if "quarter" in self.df.columns:
                self.quarters = [q.upper() for q in self.df["quarter"].dropna().unique()]
                
            logger.info(f"Successfully loaded financial metrics with shape {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            self.df = pd.DataFrame()

    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Parses an incoming query, performs numerical filtering, slicing, 
        and aggregation, and returns structured result payload.

        Args:
            query (str): Natural language financial query.

        Returns:
            Dict[str, Any]: Execution state, metrics found, and results.
        """
        if self.df is None or self.df.empty:
            return {
                "status": "error",
                "message": "Financial data database is empty or not loaded.",
                "value": None
            }

        query_lower = query.lower()
        
        # Define mapping of vocabulary queries to DataFrame columns
        metric_map = {
            "revenue": "revenue_b",
            "sales": "revenue_b",
            "net income": "net_income_b",
            "net profit": "net_income_b",
            "ebitda": "ebitda_b",
            "eps": "eps",
            "earnings per share": "eps",
            "pe ratio": "pe_ratio",
            "p/e ratio": "pe_ratio",
            "p/e": "pe_ratio"
        }

        # Define mapping of aggregation keywords
        agg_map = {
            "average": ("mean", "average"),
            "mean": ("mean", "mean"),
            "total": ("sum", "total sum"),
            "sum": ("sum", "sum"),
            "maximum": ("max", "maximum"),
            "max": ("max", "maximum"),
            "highest": ("max", "highest"),
            "minimum": ("min", "minimum"),
            "min": ("min", "minimum"),
            "lowest": ("min", "lowest")
        }

        # 1. Parse Ticker(s) from Query (check both symbols and company names with word boundaries)
        matched_tickers = []
        for t in self.tickers:
            if re.search(rf"\b{re.escape(t)}\b", query.upper()):
                matched_tickers.append(t)
        
        # Fallback: check if query contains company name terms (e.g., "Apple" -> "AAPL")
        if self.df is not None and not self.df.empty and "company" in self.df.columns:
            for _, row in self.df.dropna(subset=["ticker", "company"]).iterrows():
                ticker = row["ticker"].upper()
                company_name = row["company"].lower()
                # Extract clean words, ignoring corporate suffixes
                core_words = [w for w in re.split(r'\W+', company_name) if w not in {"inc", "corp", "corporation", "co", "ltd", "limited", ""}]
                for word in core_words:
                    if len(word) > 2 and re.search(rf"\b{re.escape(word)}\b", query_lower):
                        matched_tickers.append(ticker)
                        break
        matched_tickers = list(set(matched_tickers))
        
        # 2. Parse Quarter(s) from Query (using word boundaries)
        matched_quarters = []
        for q in self.quarters:
            # E.g. match "Q1" or "Q1_2025" or "Q1 2025"
            q_clean = q.replace("_", " ").lower()
            q_parts = q_clean.split()
            # Check with word boundaries for robustness
            if re.search(rf"\b{re.escape(q.lower())}\b", query_lower) or all(re.search(rf"\b{re.escape(part)}\b", query_lower) for part in q_parts):
                matched_quarters.append(q)
        
        # Deduplicate quarters matched
        matched_quarters = list(set(matched_quarters))

        # 3. Parse Target Metric
        target_metric = None
        target_col = None
        for key, col in metric_map.items():
            if key in query_lower:
                target_metric = key
                target_col = col
                break  # Pick the first matched metric

        # 4. Parse Aggregation
        applied_agg = None
        agg_func = None
        for key, (func, label) in agg_map.items():
            if key in query_lower:
                applied_agg = label
                agg_func = func
                break

        try:
            # Start with complete dataframe
            temp_df = self.df.copy()

            # Apply ticker filter if specified
            if matched_tickers:
                temp_df = temp_df[temp_df["ticker"].str.upper().isin(matched_tickers)]

            # Apply quarter filter if specified
            if matched_quarters:
                # If quarter was parsed specifically like "Q1_2025", filter exactly
                temp_df = temp_df[temp_df["quarter"].str.upper().isin(matched_quarters)]
            elif "q1" in query_lower or "q2" in query_lower or "q3" in query_lower or "q4" in query_lower:
                # Fallback fuzzy matching for quarter abbreviations
                for q_abr in ["q1", "q2", "q3", "q4"]:
                    if q_abr in query_lower:
                        temp_df = temp_df[temp_df["quarter"].str.lower().str.contains(q_abr)]

            # Check if dataframe is empty after filtering
            if temp_df.empty:
                return {
                    "status": "success",
                    "message": f"No data found matching criteria (Tickers: {matched_tickers}, Quarters: {matched_quarters}).",
                    "value": None
                }

            # Handle Aggregations vs. Slices
            if target_col:
                if agg_func:
                    # Calculate aggregate value (e.g., mean, sum)
                    numerical_val = temp_df[target_col].agg(agg_func)
                    # Round value for readability
                    result_val = round(float(numerical_val), 4) if not pd.isna(numerical_val) else None
                    ticker_str = ", ".join(matched_tickers) if matched_tickers else "all companies"
                    quarter_str = f" for {', '.join(matched_quarters)}" if matched_quarters else ""
                    formatted_result = f"The {applied_agg} {target_metric} for {ticker_str}{quarter_str} is {result_val}."
                    
                    return {
                        "status": "success",
                        "query": query,
                        "tickers": matched_tickers,
                        "quarters": matched_quarters,
                        "metric": target_metric,
                        "aggregation": applied_agg,
                        "value": result_val,
                        "formatted_result": formatted_result
                    }
                else:
                    # Retrieve raw slice values
                    results = []
                    for _, row in temp_df.iterrows():
                        ticker = row.get("ticker", "UNKNOWN").upper()
                        quarter = row.get("quarter", "UNKNOWN")
                        val = row.get(target_col, None)
                        results.append({"ticker": ticker, "quarter": quarter, "value": val})
                    
                    # Create readable text output
                    lines = [f"{r['ticker']} ({r['quarter']}): {r['value']}" for r in results]
                    formatted_result = f"Found {len(results)} metrics for {target_metric}:\n" + "\n".join(lines)
                    
                    return {
                        "status": "success",
                        "query": query,
                        "tickers": matched_tickers,
                        "quarters": matched_quarters,
                        "metric": target_metric,
                        "aggregation": None,
                        "value": results,
                        "formatted_result": formatted_result
                    }
            else:
                # No specific metric was targeted, return the matched dataframe rows as a dict list
                records = temp_df.to_dict(orient="records")
                formatted_result = f"Found {len(records)} records matching query features. Sliced table:\n"
                formatted_result += temp_df.to_markdown(index=False) if hasattr(temp_df, 'to_markdown') else str(records)
                
                return {
                    "status": "success",
                    "query": query,
                    "tickers": matched_tickers,
                    "quarters": matched_quarters,
                    "metric": None,
                    "aggregation": None,
                    "value": records,
                    "formatted_result": formatted_result
                }

        except Exception as e:
            logger.error(f"Error performing data transformation or slicing: {str(e)}")
            return {
                "status": "error",
                "message": f"Transformation error: {str(e)}",
                "value": None
            }
