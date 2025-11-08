#!/usr/bin/env python3
"""
PyTetrad Conservative FCI (cFCI) wrapper.
Returns PAG; supports optional inclusion of undirected skeleton edges.
"""

import os, glob
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import jpype, jpype.imports
from importlib.resources import files
from pandas.api.types import is_integer_dtype, is_categorical_dtype, is_float_dtype


class TetradCFCI:
    def __init__(self, **kwargs):
        self.alpha = kwargs.get("alpha", 0.01)
        self.depth = kwargs.get("depth", -1)
        self.include_undirected = kwargs.get("include_undirected", True)
        self._ensure_jvm(); self._import_tetrad_modules()

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
        self.search, self.test, self.graph, self.ptt = search, test, graph, ptt

    def _detect_data_types(self, df: pd.DataFrame) -> Tuple[list, list]:
        cats, cont = [], []
        for c in df.columns:
            if is_integer_dtype(df[c]) or is_categorical_dtype(df[c]): cats.append(c)
            elif is_float_dtype(df[c]): cont.append(c)
            else:
                try: df[c] = df[c].astype("int64"); cats.append(c)
                except Exception: df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64"); cont.append(c)
        return cats, cont

    def _convert(self, df: pd.DataFrame):
        df = df.copy(); cats, cont = self._detect_data_types(df)
        for c in df.columns:
            df[c] = df[c].astype("int64") if c in cats else df[c].astype("float64")
        tetrad_data = self.ptt.pandas_data_to_tetrad(df)
        if hasattr(tetrad_data, "getDataSet"): tetrad_data = tetrad_data.getDataSet()
        return tetrad_data, cats, cont

    def _indep(self, tetrad_data, cats, cont):
        if cats and cont: return self.test.IndTestConditionalGaussianLrt(tetrad_data, self.alpha, True)
        elif cats and not cont: return self.test.IndTestChiSquare(tetrad_data, self.alpha)
        else: return self.test.IndTestFisherZ(tetrad_data, self.alpha)

    def _pag_to_adjacency(self, pag, columns):
        n = len(columns); adj = np.zeros((n, n), dtype=int); Endpoint = self.graph.Endpoint
        for i, a in enumerate(columns):
            na = pag.getNode(a)
            for j, b in enumerate(columns):
                if i == j: continue
                nb = pag.getNode(b); e = pag.getEdge(na, nb)
                if e is None: continue
                if e.getNode1() == na:
                    ea, eb = e.getEndpoint1(), e.getEndpoint2()
                else:
                    ea, eb = e.getEndpoint2(), e.getEndpoint1()
                if ea == Endpoint.TAIL and eb == Endpoint.ARROW:
                    adj[i, j] = 1
                elif self.include_undirected:
                    adj[i, j] = 1; adj[j, i] = 1
        return adj

    def run(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[list] = None,
            prior: Optional[dict] = None) -> np.ndarray:
        if isinstance(data, np.ndarray):
            if columns is None: raise ValueError("Column names must be provided when input is a numpy array.")
            df = pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.DataFrame):
            df = data.copy(); columns = list(df.columns)
        else:
            raise ValueError("Input data must be a pandas DataFrame or numpy array.")
        if df.empty: raise ValueError("Input data cannot be empty.")
        
        # Build knowledge object if prior knowledge provided
        knowledge = None
        if prior is not None:
            try:
                from utils.tetrad_prior_knowledge import build_tetrad_knowledge
                knowledge = build_tetrad_knowledge(prior, columns)
            except Exception as e:
                print(f"[WARNING] Could not build knowledge for CFCI: {e}")
        
        tetrad_data, cats, cont = self._convert(df)
        indep = self._indep(tetrad_data, cats, cont)
        alg = self.search.Cfci(indep)
        if hasattr(alg, "setDepth"): alg.setDepth(self.depth)
        if knowledge is not None: alg.setKnowledge(knowledge)
        pag = alg.search()
        return self._pag_to_adjacency(pag, columns)


def run_cfci(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    alpha: float = 0.01,
    depth: int = -1,
    include_undirected: bool = True,
    prior: Optional[dict] = None,
) -> np.ndarray:
    cfci = TetradCFCI(alpha=alpha, depth=depth, include_undirected=include_undirected)
    return cfci.run(data, columns, prior=prior)


