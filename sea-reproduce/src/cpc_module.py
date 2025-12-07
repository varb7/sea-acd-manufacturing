#!/usr/bin/env python3
"""
PyTetrad CPC (Conservative PC) algorithm wrapper.

Features:
- Robust JVM bootstrap (py-tetrad bundled jar / env / ./resources)
- Mixed/discrete/continuous CI test selection
- Optional depth, prior knowledge support
- Outputs GES-compatible CPDAG format {-1, 0, 1} for GIES aggregator compatibility
- Conservative orientation rules (more robust to faithfulness violations)
"""

import os, glob
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import jpype, jpype.imports
from importlib.resources import files
from pandas.api.types import is_integer_dtype, is_categorical_dtype, is_float_dtype


class TetradCPC:
    """
    Conservative PC algorithm wrapper using PyTetrad.

    CPC is a variant of PC that uses more conservative orientation rules,
    making it more robust to violations of the faithfulness assumption.

    Params:
        alpha: float = 0.05   # CI test significance level
        depth: int = -1       # max conditioning set size (-1 = unlimited)
        include_undirected: bool = True  # include undirected edges
    
    Output format (GES-compatible):
        For edge i → j:  adj[i,j] = -1, adj[j,i] = 1
        For edge i — j:  adj[i,j] = -1, adj[j,i] = -1
        For no edge:     adj[i,j] = 0,  adj[j,i] = 0
    """

    def __init__(self, **kwargs):
        self.alpha = kwargs.get("alpha", 0.05)
        self.depth = kwargs.get("depth", -1)
        self.include_undirected = kwargs.get("include_undirected", True)

        self._ensure_jvm()
        self._import_tetrad_modules()

    # ---------------- JVM + imports ----------------

    def _ensure_jvm(self):
        if jpype.isJVMStarted():
            return
        jars = []
        try:
            jars.append(str(files("pytetrad.resources") / "tetrad-current.jar"))
        except Exception:
            pass
        if os.getenv("TETRAD_JAR"):
            jars.append(os.getenv("TETRAD_JAR"))
        jars += glob.glob(os.path.join("resources", "*tetrad*jar"))
        jars = [j for j in jars if j and os.path.exists(j)]
        if not jars:
            raise RuntimeError("No Tetrad JAR found. Install py-tetrad or set TETRAD_JAR, or drop a jar in ./resources/")
        jpype.startJVM(jpype.getDefaultJVMPath(), classpath=jars)

    def _import_tetrad_modules(self):
        import edu.cmu.tetrad.search as search
        import edu.cmu.tetrad.search.test as test
        import edu.cmu.tetrad.graph as graph
        import pytetrad.tools.translate as ptt
        self.search = search
        self.test = test
        self.graph = graph
        self.ptt = ptt

    # ---------------- Type handling ----------------

    def _detect_data_types(self, df: pd.DataFrame) -> Tuple[list, list]:
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
        df = df.copy()
        cats, cont = self._detect_data_types(df)
        for c in df.columns:
            df[c] = df[c].astype("int64") if c in cats else df[c].astype("float64")
        tetrad_data = self.ptt.pandas_data_to_tetrad(df)
        if hasattr(tetrad_data, "getDataSet"):
            tetrad_data = tetrad_data.getDataSet()
        return tetrad_data, cats, cont

    # ---------------- CI test ----------------

    def _create_independence_test(self, tetrad_data, cats, cont):
        """Select appropriate independence test based on data types."""
        if cats and cont:
            # Mixed data: use Conditional Gaussian LRT
            return self.test.IndTestConditionalGaussianLrt(tetrad_data, self.alpha, True)
        elif cats and not cont:
            # All discrete: use Chi-Square
            return self.test.IndTestChiSquare(tetrad_data, self.alpha)
        else:
            # All continuous: use Fisher's Z
            return self.test.IndTestFisherZ(tetrad_data, self.alpha)

    # ---------------- CPDAG → adjacency (GES format) ----------------

    def _cpdag_to_adjacency(self, cpdag, columns: list) -> np.ndarray:
        """
        Convert CPDAG to GES-compatible adjacency with values {-1, 0, 1}.

        GES format encoding:
            For directed edge i → j:
                adj[i,j] = -1 (outgoing from i)
                adj[j,i] = 1  (incoming to j)
            For undirected edge i — j:
                adj[i,j] = -1
                adj[j,i] = -1
            For no edge:
                adj[i,j] = 0
                adj[j,i] = 0
        """
        n = len(columns)
        adj = np.zeros((n, n), dtype=int)
        Endpoint = self.graph.Endpoint

        for i, a in enumerate(columns):
            na = cpdag.getNode(a)
            for j, b in enumerate(columns):
                if i == j:
                    continue
                nb = cpdag.getNode(b)
                e = cpdag.getEdge(na, nb)
                if e is None:
                    continue

                # Map endpoints relative to (na, nb)
                if e.getNode1() == na:
                    ea = e.getEndpoint1()
                    eb = e.getEndpoint2()
                else:
                    ea = e.getEndpoint2()
                    eb = e.getEndpoint1()

                # Convert CPDAG endpoints to GES-compatible values
                if ea == Endpoint.TAIL and eb == Endpoint.ARROW:
                    # i → j: directed edge from i to j
                    adj[i, j] = -1  # outgoing from i
                elif ea == Endpoint.ARROW and eb == Endpoint.TAIL:
                    # i ← j: directed edge from j to i
                    adj[i, j] = 1   # incoming to i
                elif ea == Endpoint.TAIL and eb == Endpoint.TAIL:
                    # i — j: undirected edge
                    adj[i, j] = -1

        return adj

    # ---------------- Public API ----------------

    def run(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[list] = None,
            prior: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Run CPC algorithm and return GES-compatible CPDAG adjacency.

        Args:
            data: Input data as DataFrame or numpy array
            columns: Column names (required if data is numpy array)
            prior: Optional prior knowledge dictionary

        Returns:
            Adjacency matrix with GES-compatible values {-1, 0, 1}
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
                if knowledge is not None:
                    print(f"[CPC] Applied prior knowledge with {len(prior.get('forbidden_edges', []))} forbidden edges, "
                          f"{len(prior.get('tier_ordering', []))} tiers")
            except Exception as e:
                print(f"[CPC] Warning: Could not build knowledge object: {e}")
                knowledge = None

        tetrad_data, cats, cont = self._convert_to_tetrad_format(df)
        indep = self._create_independence_test(tetrad_data, cats, cont)
        
        # Run CPC (Conservative PC) algorithm
        alg = self.search.Cpc(indep)
        if hasattr(alg, "setDepth"):
            alg.setDepth(self.depth)
        if knowledge is not None and hasattr(alg, "setKnowledge"):
            alg.setKnowledge(knowledge)
        
        cpdag = alg.search()
        return self._cpdag_to_adjacency(cpdag, columns)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "depth": self.depth,
            "include_undirected": self.include_undirected,
        }


def run_cpc(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    alpha: float = 0.05,
    depth: int = -1,
    include_undirected: bool = True,
    prior: Optional[Dict] = None,
) -> np.ndarray:
    """
    Convenience function to run CPC (Conservative PC) algorithm.

    Args:
        data: Input data as DataFrame or numpy array
        columns: Column names (required if data is numpy array)
        alpha: Significance level for independence tests
        depth: Maximum conditioning set size (-1 for unlimited)
        include_undirected: Whether to include undirected edges
        prior: Optional prior knowledge dictionary

    Returns:
        GES-compatible adjacency matrix with values {-1, 0, 1}
    """
    cpc = TetradCPC(alpha=alpha, depth=depth, include_undirected=include_undirected)
    return cpc.run(data, columns, prior=prior)


if __name__ == "__main__":
    # Quick test
    import numpy as np
    
    np.random.seed(42)
    n, k = 200, 5
    data = np.random.randn(n, k)
    # Add some dependencies
    data[:, 1] += 0.8 * data[:, 0]
    data[:, 2] += 0.6 * data[:, 1]
    data[:, 3] += 0.5 * data[:, 0] + 0.4 * data[:, 2]
    
    columns = [f"v{i}" for i in range(k)]
    
    print("Running CPC (alpha=0.05, depth=-1)...")
    adj = run_cpc(data, columns=columns, alpha=0.05, depth=-1)
    print("Adjacency matrix (GES format):")
    print(adj)
    print(f"Unique values: {set(adj.flatten())}")
    print(f"Non-zero edges: {np.sum(adj != 0)}")
