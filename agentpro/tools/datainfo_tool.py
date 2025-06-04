from .base_tool import Tool
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from collections import Counter
import json
class DataInfoTool(Tool):
    name: str = "Dataset Information Analyzer"
    description: str = "Takes a path to a csv file, Analyzes it as a pandas DataFrame and returns comprehensive information about the dataset structure, columns, and data distribution."
    action_type: str = "data_info_tool"
    input_format: str = "A string with path to a dataset, no lists , no quotes, raw string input."
    def _safe_convert_to_numeric(self, series: pd.Series) -> pd.Series:
        """Safely convert series to numeric, handling errors gracefully."""
        try:
            return pd.to_numeric(series, errors='coerce')
        except Exception:
            return series
    def _safe_convert_to_datetime(self, series: pd.Series) -> pd.Series:
        """Safely convert series to datetime, handling errors gracefully."""
        try:
            return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        except Exception:
            return series
    def _is_numeric_dtype(self, dtype) -> bool:
        """Check if dtype is numeric."""
        return pd.api.types.is_numeric_dtype(dtype)
    def _is_datetime_dtype(self, dtype) -> bool:
        """Check if dtype is datetime."""
        return pd.api.types.is_datetime64_any_dtype(dtype)
    def _get_percentage_distribution(self, value_counts: pd.Series) -> Dict[str, float]:
        """Calculate percentage distribution of values."""
        total = value_counts.sum()
        if total == 0:
            return {}
        return {str(k): round((v / total) * 100, 2) for k, v in value_counts.items()}
    def _infer_datetime_frequency(self, dt_series: pd.Series) -> str:
        """Infer the frequency of datetime series."""
        try:
            dt_series_clean = dt_series.dropna().sort_values()
            if len(dt_series_clean) < 2:
                return "insufficient_data"
            diffs = dt_series_clean.diff().dropna()
            if len(diffs) == 0:
                return "no_variation"
            mode_diff = diffs.mode()
            if len(mode_diff) == 0:
                return "irregular"
            common_diff = mode_diff.iloc[0]
            if common_diff <= pd.Timedelta(minutes=1):
                return "sub_minute"
            elif common_diff <= pd.Timedelta(hours=1):
                return "hourly_or_minutes"
            elif common_diff <= pd.Timedelta(days=1):
                return "daily"
            elif common_diff <= pd.Timedelta(days=7):
                return "weekly"
            elif common_diff <= pd.Timedelta(days=31):
                return "monthly"
            elif common_diff <= pd.Timedelta(days=366):
                return "yearly"
            else:
                return "multi_year"
        except Exception:
            return "unknown"
    def _get_datetime_row_counts(self, dt_series: pd.Series) -> Dict[str, int]:
        """Get count of rows per unique datetime value."""
        try:
            dt_clean = dt_series.dropna()
            if len(dt_clean) == 0:
                return {}
            value_counts = dt_clean.value_counts()
            unique_counts = Counter(value_counts.values)
            result = {}
            for count, frequency in unique_counts.items():
                if frequency == 1:
                    result[f"{count}_rows_per_datetime"] = f"{frequency} datetime value"
                else:
                    result[f"{count}_rows_per_datetime"] = f"{frequency} datetime values"
            return result
        except Exception:
            return {"error": "Could not analyze datetime row distribution"}
    def _analyze_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Analyze a single column and return comprehensive information."""
        try:
            info = {
                "column_name": col_name,
                "dtype": str(series.dtype),
                "total_values": len(series),
                "missing_values": series.isna().sum(),
                "non_missing_values": series.notna().sum()
            }
            try:
                unique_values = series.dropna().nunique()
                info["unique_values"] = unique_values
                if unique_values <= 20 and unique_values > 0:
                    unique_list = series.dropna().unique()
                    if len(unique_list) <= 20:  # Double check
                        info["unique_values_list"] = [str(x) for x in unique_list]
                    if unique_values <= 10:
                        value_counts = series.value_counts(dropna=False)
                        info["value_distribution"] = self._get_percentage_distribution(value_counts)
            except Exception:
                info["unique_values"] = "could_not_calculate"
            if self._is_numeric_dtype(series.dtype):
                try:
                    numeric_series = series.dropna()
                    if len(numeric_series) > 0:
                        info["numeric_stats"] = {"min": float(numeric_series.min()), "max": float(numeric_series.max()), "mean": float(numeric_series.mean()), "median": float(numeric_series.median()) }
                except Exception:
                    info["numeric_stats"] = "could_not_calculate"
            elif self._is_datetime_dtype(series.dtype):
                try:
                    dt_series = series.dropna()
                    if len(dt_series) > 0:
                        info["datetime_stats"] = {"start": str(dt_series.min()), "end": str(dt_series.max()), "frequency": self._infer_datetime_frequency(dt_series), "rows_per_datetime": self._get_datetime_row_counts(series) }
                except Exception:
                    info["datetime_stats"] = "could_not_calculate"
            elif series.dtype == 'object' and len(series.dropna()) > 0:
                try:
                    sample = series.dropna().head(5)
                    converted = self._safe_convert_to_datetime(sample)
                    if not converted.isna().all():
                        full_converted = self._safe_convert_to_datetime(series)
                        if full_converted.notna().sum() > len(series) * 0.5:  # More than 50% convertible
                            dt_clean = full_converted.dropna()
                            if len(dt_clean) > 0:
                                info["potential_datetime_stats"] = {
                                    "start": str(dt_clean.min()),
                                    "end": str(dt_clean.max()),
                                    "frequency": self._infer_datetime_frequency(full_converted),
                                    "convertible_values": len(dt_clean),
                                    "rows_per_datetime": self._get_datetime_row_counts(full_converted)
                                }
                except Exception:
                    pass 
            elif series.dtype == 'object' and len(series.dropna()) > 0:
                try:
                    sample = series.dropna().head(10)
                    converted = self._safe_convert_to_numeric(sample)
                    if not converted.isna().all():
                        full_converted = self._safe_convert_to_numeric(series)
                        if full_converted.notna().sum() > len(series) * 0.5: 
                            numeric_clean = full_converted.dropna()
                            if len(numeric_clean) > 0:
                                info["potential_numeric_stats"] = {"min": float(numeric_clean.min()), "max": float(numeric_clean.max()), "mean": float(numeric_clean.mean()), "median": float(numeric_clean.median()), "convertible_values": len(numeric_clean) }
                except Exception:
                    pass
            return info
        except Exception as e:
            return {"column_name": col_name, "error": f"Could not analyze column: {str(e)}", "dtype": str(series.dtype) if hasattr(series, 'dtype') else "unknown", "total_values": len(series) if hasattr(series, '__len__') else 0 }
    def _format_output(self, df_info: Dict[str, Any]) -> str:
        """Format the analysis results into a readable string."""
        try:
            output = []
            output.append("=" * 60)
            output.append("DATASET INFORMATION ANALYSIS")
            output.append("=" * 60)
            output.append(f"Total Rows: {df_info['total_rows']}")
            output.append(f"Total Columns: {df_info['total_columns']}")
            output.append("")
            for i, col_info in enumerate(df_info['columns'], 1):
                output.append(f"Column {i}: {col_info['column_name']}")
                output.append("-" * 40)
                output.append(f"  Data Type: {col_info['dtype']}")
                output.append(f"  Total Values: {col_info['total_values']}")
                output.append(f"  Missing Values: {col_info['missing_values']}")
                output.append(f"  Non-Missing Values: {col_info['non_missing_values']}")
                if 'unique_values' in col_info:
                    output.append(f"  Unique Values: {col_info['unique_values']}")
                    if 'unique_values_list' in col_info:
                        unique_str = ", ".join(col_info['unique_values_list'])
                        output.append(f"  Unique Values List: {unique_str}")
                    if 'value_distribution' in col_info:
                        output.append("  Value Distribution (%):")
                        for value, percentage in col_info['value_distribution'].items():
                            output.append(f"    {value}: {percentage}%")
                if 'numeric_stats' in col_info and col_info['numeric_stats'] != "could_not_calculate":
                    stats = col_info['numeric_stats']
                    output.append(f"  Numeric Statistics:")
                    output.append(f"    Min: {stats['min']}")
                    output.append(f"    Max: {stats['max']}")
                    output.append(f"    Mean: {stats['mean']}")
                    output.append(f"    Median: {stats['median']}")
                if 'datetime_stats' in col_info and col_info['datetime_stats'] != "could_not_calculate":
                    stats = col_info['datetime_stats']
                    output.append(f"  DateTime Statistics:")
                    output.append(f"    Start: {stats['start']}")
                    output.append(f"    End: {stats['end']}")
                    output.append(f"    Frequency: {stats['frequency']}")
                    if stats['rows_per_datetime']:
                        output.append(f"    Rows per DateTime:")
                        for key, value in stats['rows_per_datetime'].items():
                            output.append(f"      {key}: {value}")
                if 'potential_datetime_stats' in col_info:
                    stats = col_info['potential_datetime_stats']
                    output.append(f"  Potential DateTime (if converted):")
                    output.append(f"    Start: {stats['start']}")
                    output.append(f"    End: {stats['end']}")
                    output.append(f"    Frequency: {stats['frequency']}")
                    output.append(f"    Convertible Values: {stats['convertible_values']}")
                if 'potential_numeric_stats' in col_info:
                    stats = col_info['potential_numeric_stats']
                    output.append(f"  Potential Numeric (if converted):")
                    output.append(f"    Min: {stats['min']}")
                    output.append(f"    Max: {stats['max']}")
                    output.append(f"    Mean: {stats['mean']}")
                    output.append(f"    Median: {stats['median']}")
                    output.append(f"    Convertible Values: {stats['convertible_values']}")
                if 'error' in col_info:
                    output.append(f"  Error: {col_info['error']}")
                output.append("")
            return "\n".join(output)
        except Exception as e:
            return f"Error formatting output: {str(e)}\n\nRaw data: {str(df_info)}"
    def run(self, input_text: Any) -> str:
        """Main method to analyze DataFrame and return information."""
        try:
            df = pd.read_csv(input_text)
            #df = None
            if isinstance(input_text, pd.DataFrame):
                df = input_text
            elif isinstance(input_text, dict) and 'dataframe' in input_text:
                df = input_text['dataframe']
                if not isinstance(df, pd.DataFrame):
                    return "Error: 'dataframe' key must contain a pandas DataFrame object."
            elif isinstance(input_text, str):
                try:
                    data = json.loads(input_text)
                    if 'dataframe' in data:
                        df = data['dataframe']
                    else:
                        return "Error: JSON input must contain 'dataframe' key with DataFrame object."
                except json.JSONDecodeError:
                    return "Error: String input must be valid JSON with 'dataframe' key."
            else:
                return "Error: Input must be a pandas DataFrame, dict with 'dataframe' key, or JSON string."
            if df is None or not isinstance(df, pd.DataFrame):
                return "Error: No valid pandas DataFrame found in input."
            if df.empty:
                return "Error: DataFrame is empty (no rows or columns)."
            analysis_result = {"total_rows": len(df), "total_columns": len(df.columns), "columns": [] }
            for col_name in df.columns:
                try:
                    col_analysis = self._analyze_column(df[col_name], col_name)
                    analysis_result["columns"].append(col_analysis)
                except Exception as e:
                    analysis_result["columns"].append({"column_name": col_name, "error": f"Failed to analyze column: {str(e)}", "dtype": str(df[col_name].dtype) if col_name in df.columns else "unknown"})
            return self._format_output(analysis_result)
        except Exception as e:
            return f"Error analyzing DataFrame: {str(e)}"
