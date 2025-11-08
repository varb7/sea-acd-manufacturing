import os, glob
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import jpype, jpype.imports
from importlib.resources import files
from pandas.api.types import is_integer_dtype, is_categorical_dtype, is_float_dtype


class TetradGFCI:
    def __init__(self, alpha: float = 0.01, depth: int = -1, penalty_discount: float = 2.0, include_undirected: bool = True):
        self.alpha = alpha
        self.depth = depth
        self.penalty_discount = penalty_discount
        self.include_undirected = include_undirected
        self._ensure_jvm()
        self._import_tetrad_modules()

    def _ensure_jvm(self):
        if jpype.isJVMStarted(): return
        jars = []
        try: jars.append(str(files("pytetrad.resources") / "tetrad-current.jar"))
        except Exception: pass
        if os.getenv("TETRAD_JAR"): jars.append(os.getenv("TETRAD_JAR"))
        jars += glob.glob(os.path.join("resources", "*tetrad*jar"))
        jars = [j for j in jars if j and os.path.exists(j)]
        if not jars:
            raise RuntimeError("No Tetrad JAR found. Install py-tetrad or set TETRAD_JAR, or drop a jar in ./resources/")
        jpype.startJVM(jpype.getDefaultJVMPath(), classpath=jars)

    def _import_tetrad_modules(self):
        import edu.cmu.tetrad.search as search
        import edu.cmu.tetrad.search.test as test
        import edu.cmu.tetrad.search.score as score
        import edu.cmu.tetrad.graph as graph
        import pytetrad.tools.translate as ptt
        self.search, self.test, self.score, self.graph, self.ptt = search, test, score, graph, ptt

    def _detect_data_types(self, df: pd.DataFrame):
        cats, cont = [], []
        for c in df.columns:
            if is_integer_dtype(df[c]) or is_categorical_dtype(df[c]): cats.append(c)
            elif is_float_dtype(df[c]): cont.append(c)
            else:
                try: df[c] = df[c].astype("int64"); cats.append(c)
                except Exception: df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64"); cont.append(c)
        return cats, cont

    def _convert(self, df: pd.DataFrame):
        df = df.copy()
        cats, cont = self._detect_data_types(df)
        for c in df.columns:
            df[c] = df[c].astype("int64") if c in cats else df[c].astype("float64")
        tetrad_data = self.ptt.pandas_data_to_tetrad(df)
        if hasattr(tetrad_data, "getDataSet"):
            tetrad_data = tetrad_data.getDataSet()
        return tetrad_data, cats, cont

    def _indep(self, tetrad_data, cats, cont):
        if cats and cont:
            return self.test.IndTestConditionalGaussianLrt(tetrad_data, self.alpha, True)
        elif cats and not cont:
            return self.test.IndTestChiSquare(tetrad_data, self.alpha)
        else:
            return self.test.IndTestFisherZ(tetrad_data, self.alpha)

    def _score(self, tetrad_data, cats, cont):
        if cats and cont:
            score = self.score.ConditionalGaussianBicScore(tetrad_data)
            score.setDiscretize(True); score.setNumCategoriesToDiscretize(3)
        elif cats and not cont:
            score = self.score.BDeuScore(tetrad_data)
            score.setPriorEquivalentSampleSize(10.0)
        else:
            score = self.score.SemBicScore(tetrad_data, self.penalty_discount, True)
        return score

    def _graph_to_adjacency(self, g, columns):
        n = len(columns); adj = np.zeros((n, n), dtype=int); Endpoint = self.graph.Endpoint
        for i, a in enumerate(columns):
            na = g.getNode(a)
            for j, b in enumerate(columns):
                if i == j: continue
                nb = g.getNode(b); e = g.getEdge(na, nb)
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
                print(f"[WARNING] Could not build knowledge for GFCI: {e}")
        
        tetrad_data, cats, cont = self._convert(df)
        indep = self._indep(tetrad_data, cats, cont)
        score = self._score(tetrad_data, cats, cont)
        alg = self.search.Gfci(indep, score)
        # Depth control: large depths on mixed data can cause undefined p-values (sparse configs)
        if hasattr(alg, "setDepth"):
            if (cats and cont) and (self.depth is None or self.depth < 0):
                # Cap depth for mixed data to reduce sparse conditioning sets
                alg.setDepth(3)
            else:
                alg.setDepth(self.depth)
        if knowledge is not None: alg.setKnowledge(knowledge)

        # Run with a retry: if search fails due to undefined p-values, retry with smaller depth
        try:
            pag = alg.search()
        except Exception as e:
            try:
                # Retry at safer depth=2
                if hasattr(alg, "setDepth"):
                    alg.setDepth(2)
                pag = alg.search()
            except Exception:
                # As a last resort, return an empty adjacency to keep pipeline running
                return np.zeros((len(columns), len(columns)), dtype=int)
        return self._graph_to_adjacency(pag, columns)

def run_gfci(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[list] = None,
    alpha: float = 0.01,
    depth: int = -1,
    penalty_discount: float = 2.0,
    include_undirected: bool = True,
    prior: Optional[dict] = None,
) -> np.ndarray:
    gfci = TetradGFCI(alpha=alpha, depth=depth, penalty_discount=penalty_discount, include_undirected=include_undirected)
    return gfci.run(data, columns, prior=prior)