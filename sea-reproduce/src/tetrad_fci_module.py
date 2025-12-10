#!/usr/bin/env python3
"""
PyTetrad standard FCI (Fast Causal Inference) wrapper.

FCI is a constraint-based causal discovery algorithm that:
- Discovers PAG (Partial Ancestral Graph) structure
- Handles latent confounders and selection bias
- Uses conditional independence tests

Features:
- Robust JVM bootstrap (py-tetrad bundled jar / env / ./resources)
- Mixed/discrete/continuous data handling
- Returns FCI-compatible adjacency matrix {-1, 0, 1, 2}
"""

import os, glob, warnings
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import jpype, jpype.imports
from importlib.resources import files
from pandas.api.types import is_integer_dtype, is_categorical_dtype, is_float_dtype

warnings.filterwarnings("ignore")


class TetradFCI:
    """
    Standard FCI wrapper using PyTetrad.

    Parameters:
        alpha: float = 0.01       # CI test significance level
        depth: int = -1           # max conditioning set size (-1 = unlimited)
        max_path_length: int = -1 # max path length for discriminating path rule (-1 = unlimited)
        include_undirected: bool = True  # include undirected edges
    
    Output format (FCI-compatible):
        -1 = backward edge (<-)
         0 = no edge
         1 = undirected edge (-)
         2 = forward edge (->)
    """

    def __init__(self, **kwargs):
        self.alpha = kwargs.get("alpha", 0.01)
        self.depth = kwargs.get("depth", -1)
        self.max_path_length = kwargs.get("max_path_length", -1)
        self.include_undirected = kwargs.get("include_undirected", True)

        self._ensure_jvm()
        self._import_tetrad_modules()

    # ---------------- JVM + imports ----------------

    def _ensure_jvm(self):
        if jpype.isJVMStarted():
            return
        jars = []
        # 1) bundled jar from py-tetrad
        try:
            jars.append(str(files("pytetrad.resources") / "tetrad-current.jar"))
        except Exception:
            pass
        # 2) env override
        if os.getenv("TETRAD_JAR"):
            jars.append(os.getenv("TETRAD_JAR"))
        # 3) local resources
        jars += glob.glob(os.path.join("resources", "*tetrad*jar"))
        jars = [j for j in jars if j and os.path.exists(j)]
        if not jars:
            raise RuntimeError(
                "No Tetrad JAR found. Install py-tetrad or set TETRAD_JAR, or drop a jar in ./resources/"
            )
        jpype.startJVM(jpype.getDefaultJVMPath(), classpath=jars)

    def _import_tetrad_modules(self):
        try:
            import edu.cmu.tetrad.search as search
            import edu.cmu.tetrad.search.test as test
            import edu.cmu.tetrad.graph as graph
            import pytetrad.tools.translate as ptt
        except Exception as e:
            raise RuntimeError(f"Failed to import Tetrad modules: {e}")
        self.search = search
        self.test = test
        self.graph = graph
        self.ptt = ptt

    # ---------------- Type handling ----------------

    def _detect_data_types(self, df: pd.DataFrame) -> Tuple[list, list]:
        """Return (categorical_cols, continuous_cols) using dtype."""
        cats, cont = [], []
        for c in df.columns:
            if is_integer_dtype(df[c]) or is_categorical_dtype(df[c]):
                cats.append(c)
            elif is_float_dtype(df[c]):
                cont.append(c)
            else:
                try:
                    df[c] = df[c].astype("int64")
                    cats.append(c)
                except Exception:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
                    cont.append(c)
        return cats, cont

    def _convert_to_tetrad_format(self, df: pd.DataFrame):
        """Coerce dtypes explicitly, return (Tetrad DataSet, cats, cont)."""
        df = df.copy()
        cats, cont = self._detect_data_types(df)
        for c in df.columns:
            df[c] = df[c].astype("int64") if c in cats else df[c].astype("float64")
        tetrad_data = self.ptt.pandas_data_to_tetrad(df)
        if hasattr(tetrad_data, "getDataSet"):
            tetrad_data = tetrad_data.getDataSet()
        return tetrad_data, cats, cont

    # ---------------- CI tests ----------------

    def _create_independence_test(self, tetrad_data, cats, cont):
        """
        Choose test by data type:
          mixed      -> IndTestConditionalGaussianLrt
          discrete   -> IndTestChiSquare
          continuous -> IndTestFisherZ
        """
        if cats and cont:
            return self.test.IndTestConditionalGaussianLrt(tetrad_data, self.alpha, True)
        elif cats and not cont:
            return self.test.IndTestChiSquare(tetrad_data, self.alpha)
        else:
            return self.test.IndTestFisherZ(tetrad_data, self.alpha)

    # ---------------- PAG → adjacency ----------------

    def _pag_to_adjacency_matrix(self, pag, columns: list) -> np.ndarray:
        """
        Convert PAG to FCI-compatible adjacency with values {-1, 0, 1, 2}.

        Encoding (to match FCI downstream processing):
          -1 = backward edge (<-)
           0 = no edge
           1 = undirected edge (-)
           2 = forward edge (->)
        """
        n = len(columns)
        adj = np.zeros((n, n), dtype=int)
        Endpoint = self.graph.Endpoint

        for i, a in enumerate(columns):
            na = pag.getNode(a)
            if na is None:
                continue
            for j, b in enumerate(columns):
                if i == j:
                    continue
                nb = pag.getNode(b)
                if nb is None:
                    continue
                e = pag.getEdge(na, nb)
                if e is None:
                    continue

                # Map endpoints relative to (na, nb)
                if e.getNode1() == na:
                    ea = e.getEndpoint1()
                    eb = e.getEndpoint2()
                else:
                    ea = e.getEndpoint2()
                    eb = e.getEndpoint1()

                # Convert PAG endpoints to FCI-compatible values
                if ea == Endpoint.TAIL and eb == Endpoint.ARROW:
                    # a -> b (definite directed)
                    adj[i, j] = 2
                elif ea == Endpoint.ARROW and eb == Endpoint.TAIL:
                    # a <- b (definite backward)
                    adj[i, j] = -1
                elif ea == Endpoint.TAIL and eb == Endpoint.TAIL:
                    # a - b (undirected/skeleton)
                    adj[i, j] = 1
                elif ea == Endpoint.CIRCLE and eb == Endpoint.ARROW:
                    # a o-> b (partial forward)
                    adj[i, j] = 2
                elif ea == Endpoint.ARROW and eb == Endpoint.CIRCLE:
                    # a <-o b (partial backward)
                    adj[i, j] = -1
                elif ea == Endpoint.CIRCLE and eb == Endpoint.CIRCLE:
                    # a o-o b (fully uncertain, treat as undirected)
                    adj[i, j] = 1
                elif ea == Endpoint.ARROW and eb == Endpoint.ARROW:
                    # a <-> b (bidirected, treat as undirected for compatibility)
                    adj[i, j] = 1
                elif self.include_undirected:
                    # Fallback: any other edge type treated as undirected
                    adj[i, j] = 1

        return adj

    # ---------------- Public API ----------------

    def run(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[list] = None,
            prior: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Run standard FCI and return FCI-compatible adjacency matrix.

        Args:
            data: Input data as DataFrame or numpy array
            columns: Column names (required if data is numpy array)
            prior: Optional prior knowledge dictionary

        Returns:
            Adjacency matrix with FCI-compatible values {-1, 0, 1, 2}
        """
        if isinstance(data, np.ndarray):
            if columns is None:
                raise ValueError("Column names must be provided when input is a numpy array.")
            df = pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            columns = list(df.columns)
        else:
            raise ValueError("Input data must be a pandas DataFrame or numpy array.")
        if df.empty:
            raise ValueError("Input data cannot be empty.")

        # Build knowledge object if prior knowledge provided
        knowledge = None
        if prior is not None:
            try:
                from utils.tetrad_prior_knowledge import build_tetrad_knowledge
                knowledge = build_tetrad_knowledge(prior, columns)
            except Exception as e:
                print(f"[WARNING] Could not build knowledge for FCI: {e}")

        tetrad_data, cats, cont = self._convert_to_tetrad_format(df)
        indep = self._create_independence_test(tetrad_data, cats, cont)

        # Run standard FCI algorithm
        alg = self.search.Fci(indep)
        
        if hasattr(alg, "setDepth"):
            alg.setDepth(self.depth)
        
        if hasattr(alg, "setMaxPathLength"):
            alg.setMaxPathLength(self.max_path_length)

        if knowledge is not None and hasattr(alg, "setKnowledge"):
            alg.setKnowledge(knowledge)

        # Run with retry on failure
        try:
            pag = alg.search()
        except Exception as e:
            try:
                if hasattr(alg, "setDepth"):
                    alg.setDepth(2)
                pag = alg.search()
            except Exception:
                return np.zeros((len(columns), len(columns)), dtype=int)

        return self._pag_to_adjacency_matrix(pag, columns)


# -------------- Convenience function --------------

def run_tetrad_fci(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    alpha: float = 0.01,
    depth: int = -1,
    max_path_length: int = -1,
    include_undirected: bool = True,
    prior: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Convenience function to run standard Tetrad FCI algorithm.

    Args:
        data: Input data as DataFrame or numpy array
        columns: Column names (required if data is numpy array)
        alpha: Significance level for independence tests
        depth: Maximum conditioning set size (-1 for unlimited)
        max_path_length: Max path length for discriminating path rule (-1 for unlimited)
        include_undirected: Whether to include undirected edges
        prior: Optional prior knowledge dictionary

    Returns:
        Adjacency matrix (n_vars, n_vars) with FCI-compatible values {-1, 0, 1, 2}
    """
    fci = TetradFCI(
        alpha=alpha,
        depth=depth,
        max_path_length=max_path_length,
        include_undirected=include_undirected
    )
    return fci.run(data, columns, prior=prior)


# -------------- Quick sanity demo --------------

if __name__ == "__main__":
    print("Tetrad FCI Module (sanity test)")
    np.random.seed(42)
    n = 1000

    # Mixed toy structure
    cat = np.random.choice([0, 1, 2], size=n)
    x = (cat - cat.mean()) + np.random.normal(0, 1, size=n)
    y = 1.0 * x + 0.8 * (cat == 1) + 1.5 * (cat == 2) + np.random.normal(0, 1, size=n)
    noise = np.random.normal(0, 1, size=n)

    df = pd.DataFrame({
        "cat": cat.astype("int64"),
        "x": x.astype("float64"),
        "y": y.astype("float64"),
        "noise": noise.astype("float64"),
    })

    print("Running Tetrad FCI (alpha=0.05, depth=2)…")
    adj = run_tetrad_fci(df, alpha=0.05, depth=2)
    print("Adjacency matrix:")
    print(adj)

