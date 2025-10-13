#!/usr/bin/env python3
"""
Modular RFCI (Really Fast Causal Inference) using PyTetrad.

- Robust JVM bootstrap (bundled jar / env / ./resources).
- Safe dtype handling: ints/categories -> discrete; floats -> continuous.
- Correct independence-test constructors (no stray args) via search.test.
- Depth handling: setDepth(-1) for unlimited.
- Orientation-aware PAG -> adjacency (directed a→b only).
"""

import os, glob, warnings
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import jpype, jpype.imports
from importlib.resources import files
from pandas.api.types import is_integer_dtype, is_categorical_dtype, is_float_dtype

warnings.filterwarnings("ignore")


class TetradRFCI:
    """
    Clean RFCI wrapper using PyTetrad.

    Params:
      alpha: float = 0.01   # CI test significance (higher -> less conservative)
      depth: int  = -1      # max conditioning set size (-1 = unlimited)
      count_partial: bool   # count a o-> b as directed (default False)
    """

    def __init__(self, **kwargs):
        self.alpha = kwargs.get("alpha", 0.01)
        self.depth = kwargs.get("depth", -1)
        self.count_partial = kwargs.get("count_partial", False)
        # When True, include any PAG adjacency (undirected/ambiguous) as symmetric edges
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
        """Return (categorical_cols, continuous_cols) using dtype (no nunique heuristics)."""
        cats, cont = [], []
        for c in df.columns:
            if is_integer_dtype(df[c]) or is_categorical_dtype(df[c]):
                cats.append(c)
            elif is_float_dtype(df[c]):
                cont.append(c)
            else:
                # Best-effort fallback: try int; otherwise numeric float
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

    # ---------------- CI tests + RFCI ----------------

    def _create_independence_test(self, tetrad_data, cats, cont):
        """
        Choose test by data type:
          mixed      -> IndTestConditionalGaussianLrt(DataSet, alpha)
          discrete   -> IndTestChiSquare(DataSet, alpha)
          continuous -> IndTestFisherZ(DataSet, alpha)
        """
        if cats and cont:
            return self.test.IndTestConditionalGaussianLrt(tetrad_data, self.alpha, True)
        elif cats and not cont:
            test = self.test.IndTestChiSquare(tetrad_data, self.alpha)
            return test
        else:
            return self.test.IndTestFisherZ(tetrad_data, self.alpha)

    def _run_rfci(self, indep_test):
        rfci = self.search.Rfci(indep_test)
        rfci.setDepth(self.depth)
        return rfci.search()

    # ---------------- PAG → adjacency (directed) ----------------

    def _pag_to_adjacency_matrix(self, pag, columns: list) -> np.ndarray:
        """
        Mark a→b iff endpoint at a is TAIL and at b is ARROW.
        Optionally also count a o-> b when count_partial=True.
        """
        n = len(columns)
        adj = np.zeros((n, n), dtype=int)
        Endpoint = self.graph.Endpoint

        for i, a in enumerate(columns):
            na = pag.getNode(a)
            for j, b in enumerate(columns):
                if i == j:
                    continue
                nb = pag.getNode(b)
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
                if ea == Endpoint.TAIL and eb == Endpoint.ARROW:
                    adj[i, j] = 1
                elif self.count_partial and ea == Endpoint.CIRCLE and eb == Endpoint.ARROW:
                    adj[i, j] = 1
                elif self.include_undirected:
                    # treat any connected-but-uncertain as undirected skeleton
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    # ---------------- Public API ----------------

    def run(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[list] = None) -> np.ndarray:
        """Run RFCI and return a directed adjacency (parents → children) from the PAG."""
        if isinstance(data, np.ndarray):
            if columns is None:
                raise ValueError("Column names must be provided when input is a numpy array.")
            df = pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            df = data.copy(); columns = list(df.columns)
        else:
            raise ValueError("Input data must be a pandas DataFrame or numpy array.")
        if df.empty:
            raise ValueError("Input data cannot be empty.")

        tetrad_data, cats, cont = self._convert_to_tetrad_format(df)
        indep = self._create_independence_test(tetrad_data, cats, cont)
        pag = self._run_rfci(indep)
        return self._pag_to_adjacency_matrix(pag, columns)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "depth": self.depth,
            "count_partial": self.count_partial,
        }

    def set_parameters(self, **kwargs):
        if "alpha" in kwargs:
            self.alpha = kwargs["alpha"]
        if "depth" in kwargs:
            self.depth = kwargs["depth"]
        if "count_partial" in kwargs:
            self.count_partial = kwargs["count_partial"]


# -------------- Convenience function --------------

def run_rfci(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    alpha: float = 0.01,
    depth: int = -1,
    count_partial: bool = False,
    include_undirected: bool = True,
) -> np.ndarray:
    rfci = TetradRFCI(alpha=alpha, depth=depth, count_partial=count_partial, include_undirected=include_undirected)
    return rfci.run(data, columns)


# -------------- Quick sanity demo --------------

if __name__ == "__main__":
    print("Tetrad RFCI Module (sanity test)")
    np.random.seed(42)
    n = 1000

    # Mixed toy structure:
    # cat -> x -> y, and cat -> y; noise unrelated
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

    print("Running RFCI (alpha=0.05, depth=2)…")
    adj = run_rfci(df, alpha=0.05, depth=2)
    print("Edges (directed count):", int(np.sum(adj)))
    print(adj)
