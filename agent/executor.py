"""
Sandboxed Python code execution engine.
Executes LLM-generated code in a restricted environment.
"""

import sys
import io
import signal
import traceback
import threading
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str = ""
    result_df: Any = None
    result_value: Any = None
    figures: list = field(default_factory=list)
    error: str = ""
    error_traceback: str = ""


# Builtins that are safe to expose to generated code
SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "hasattr": hasattr,
    "getattr": getattr,
    "repr": repr,
    "format": format,
    "hash": hash,
    "id": id,
    "any": any,
    "all": all,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "oct": oct,
    "bin": bin,
    "pow": pow,
    "divmod": divmod,
    "True": True,
    "False": False,
    "None": None,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "Exception": Exception,
    "StopIteration": StopIteration,
    "RuntimeError": RuntimeError,
    "ZeroDivisionError": ZeroDivisionError,
}

# Explicitly blocked attributes
BLOCKED_NAMES = {
    "os", "sys", "subprocess", "importlib", "shutil", "pathlib",
    "open", "exec", "eval", "compile", "__import__",
    "globals", "locals", "vars", "dir",
    "setattr", "delattr",
    "exit", "quit",
    "breakpoint", "input",
}


def _validate_code(code: str) -> str | None:
    """
    Validate generated code for unsafe patterns.
    Returns error message if unsafe, None if ok.
    """
    # Check for blocked names in the code
    dangerous_patterns = [
        "import os", "import sys", "import subprocess", "import shutil",
        "import importlib", "import pathlib",
        "os.system", "os.popen", "os.exec", "os.remove", "os.unlink",
        "subprocess.", "shutil.",
        "__import__", "importlib.",
        "open(", "exec(", "eval(",
        "compile(", "globals(", "locals(",
        "breakpoint(", "input(",
        "exit(", "quit(",
    ]

    for pattern in dangerous_patterns:
        if pattern in code:
            return f"Security: blocked pattern '{pattern}' detected in code."

    return None


def execute_code(code: str, dataframes: dict[str, pd.DataFrame], timeout: int = 30) -> ExecutionResult:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Python code string to execute
        dataframes: Dict mapping variable names to DataFrames (e.g., {"df": my_df})
        timeout: Maximum execution time in seconds

    Returns:
        ExecutionResult with outputs, figures, and any errors
    """
    # Validate code safety
    error = _validate_code(code)
    if error:
        return ExecutionResult(success=False, error=error)

    # Close any existing figures
    plt.close("all")

    # Capture stdout
    old_stdout = sys.stdout
    captured_stdout = io.StringIO()

    # Build the restricted namespace
    namespace: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        # Data libraries
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
        # Visualization
        "plt": plt,
        "matplotlib": matplotlib,
        "sns": sns,
        "seaborn": sns,
    }

    # Add ML libraries (lazy import to avoid startup cost)
    try:
        import sklearn
        from sklearn import (
            linear_model, ensemble, cluster, svm,
            metrics, preprocessing, model_selection,
            tree, neighbors, decomposition,
        )
        from sklearn.metrics import silhouette_score
        namespace.update({
            "sklearn": sklearn,
            "linear_model": linear_model,
            "ensemble": ensemble,
            "cluster": cluster,
            "svm": svm,
            "tree": tree,
            "neighbors": neighbors,
            "decomposition": decomposition,
            "sk_metrics": metrics,
            "sk_preprocessing": preprocessing,
            "model_selection": model_selection,
        })

        # Add commonly used classes directly
        namespace.update({
            "LinearRegression": linear_model.LinearRegression,
            "LogisticRegression": linear_model.LogisticRegression,
            "RandomForestRegressor": ensemble.RandomForestRegressor,
            "RandomForestClassifier": ensemble.RandomForestClassifier,
            "GradientBoostingRegressor": ensemble.GradientBoostingRegressor,
            "GradientBoostingClassifier": ensemble.GradientBoostingClassifier,
            "KMeans": cluster.KMeans,
            "IsolationForest": ensemble.IsolationForest,
            "StandardScaler": preprocessing.StandardScaler,
            "LabelEncoder": preprocessing.LabelEncoder,
            "train_test_split": model_selection.train_test_split,
            "cross_val_score": model_selection.cross_val_score,
            "PCA": decomposition.PCA,
            "silhouette_score": silhouette_score,
        })
    except ImportError:
        pass

    # Add ML utility functions
    try:
        from analysis.ml_engine import prepare_ml_data, evaluate_model, format_ml_report
        namespace.update({
            "prepare_ml_data": prepare_ml_data,
            "evaluate_model": evaluate_model,
            "format_ml_report": format_ml_report,
        })
    except ImportError:
        pass

    # Add scipy stats
    try:
        from scipy import stats as scipy_stats
        namespace["scipy_stats"] = scipy_stats
    except ImportError:
        pass

    # Add plotly
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        namespace["px"] = px
        namespace["go"] = go
    except ImportError:
        pass

    # Add datetime
    import datetime
    import math
    namespace["datetime"] = datetime
    namespace["math"] = math

    # Add the user's dataframes
    namespace.update(dataframes)

    # Container for results
    namespace["result_df"] = None
    namespace["result_value"] = None
    namespace["figures"] = []
    namespace["plotly_figures"] = []

    result = ExecutionResult(success=False)

    def _run():
        nonlocal result
        try:
            sys.stdout = captured_stdout
            exec(code, namespace)

            result.success = True
            result.stdout = captured_stdout.getvalue()

            # Collect result dataframe
            if namespace.get("result_df") is not None:
                result.result_df = namespace["result_df"]

            # Collect result value
            if namespace.get("result_value") is not None:
                result.result_value = namespace["result_value"]

            # Collect matplotlib figures
            fig_nums = plt.get_fignums()
            for num in fig_nums:
                fig = plt.figure(num)
                result.figures.append(fig)

            # Also collect any figures explicitly added to the figures list
            if namespace.get("figures"):
                for fig in namespace["figures"]:
                    if fig not in result.figures:
                        result.figures.append(fig)

            # Collect plotly figures
            if namespace.get("plotly_figures"):
                if not hasattr(result, "plotly_figures"):
                    result.plotly_figures = []
                result.plotly_figures = namespace["plotly_figures"]

        except Exception as e:
            result.success = False
            result.error = f"{type(e).__name__}: {str(e)}"
            result.error_traceback = traceback.format_exc()
            result.stdout = captured_stdout.getvalue()
        finally:
            sys.stdout = old_stdout

    # Run with timeout using a thread
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        result.success = False
        result.error = f"Execution timed out after {timeout} seconds."
        sys.stdout = old_stdout

    return result
