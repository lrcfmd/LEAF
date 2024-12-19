"""
This module provides a class for handling compositional embeddings.

Typical usage example:
    Fe2O3_magpie = CompositionalEmbedding("Fe2O3", "magpie")
"""
import collections
import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import energy_distance, wasserstein_distance
from sklearn.metrics import DistanceMetric
from tqdm import tqdm

from .core import Embedding
from .utils.math import cosine_distance

tqdm.pandas()
# Modified from pymatgen.core.Compositions


def formula_parser(formula: str) -> Dict[str, float]:
    # TO-DO: Add validation to check composition contains real elements.
    """
    Parse a string formula.

    Returns a dictionary of the composition with key:value pairs
    of element symbol:amount.

    Args:
        formula (str): A string formula e.g. CsPbI3, Li7La3Zr2O12

    Returns:
        (dict): A dictionary of the composition

    """
    # For Metallofullerene
    formula = formula.replace("@", "")

    regex = r"\(([^\(\)]+)\)\s*([\.e\d]*)"
    r = re.compile(regex)
    m = re.search(r, formula)
    if m:
        factor = 1.0
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = _get_sym_dict(m.group(1), factor)
        expanded_sym = "".join([f"{el}{amt}" for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return formula_parser(expanded_formula)
    return _get_sym_dict(formula, 1)


# Parses formula and returns a dictionary of ele symbol: amount
# From pymatgen.core.composition
def _get_sym_dict(formula: str, factor: Union[int, float]) -> Dict[str, float]:
    sym_dict: Dict[str, float] = collections.defaultdict(float)
    regex = r"([A-Z][a-z]*)\s*([-*\.e\d]*)"
    r = re.compile(regex)
    for m in re.finditer(r, formula):
        el = m.group(1)
        amt = 1.0
        if m.group(2).strip() != "":
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        formula = formula.replace(m.group(), "", 1)
    if formula.strip():
        raise ValueError(f"{formula} is an invalid formula")

    return sym_dict


# Function for fractional compositions
def _get_fractional_composition(formula: str) -> Dict[str, float]:
    el_dict = formula_parser(formula)
    elamt = {}
    natoms = 0
    for el, v in el_dict.items():
        elamt[el] = v
        natoms += abs(v)
    return {el: elamt[el] / natoms for el in elamt}


# Class to handle compositional embeddings
class CompositionalEmbedding:
    """
    Class to handle compositional embeddings.

    Args:
        formula (str): A string formula e.g. CsPbI3, Li7La3Zr2O12
        embedding (Union[str, Embedding]): Either a string name of the embedding
        or an Embedding instance
        x (int, optional): The non-stoichiometric amount.
    """

    def __init__(self, formula: str, embedding: Union[str, Embedding], x=1):
        """Initialise a CompositionalEmbedding instance."""
        self.embedding = embedding

        # If a string has been passed for embedding, create an Embedding instance
        if isinstance(embedding, str):
            self.embedding = Embedding.load_data(embedding)

        self.embedding_name: str = self.embedding.embedding_name
        # Set an attribute for the formula
        self.formula = formula

        # Set an attribute for the comp dict
        comp_dict = formula_parser(self.formula)
        self._natoms = 0
        for el, v in comp_dict.items():
            if v < 0:
                raise ValueError("Formula cannot contain negative amounts of elements")
            self._natoms += abs(v)

        self.composition = comp_dict

        # Set an attribute for the element list
        self.element_list = list(self.composition.keys())
        # Set an attribute for the element matrix
        self.el_matrix = np.zeros(
            shape=(len(self.composition), len(self.embedding.embeddings["H"]))
        )
        for i, k in enumerate(self.composition.keys()):
            self.el_matrix[i] = self.embedding.embeddings[k]
        self.el_matrix = np.nan_to_num(self.el_matrix)

        # Set an attribute for the stoichiometric vector
        self.stoich_vector = np.array(list(self.composition.values()))

        # Set an attribute for the normalised stoichiometric vector
        self.norm_stoich_vector = self.stoich_vector / self._natoms

    @property
    def fractional_composition(self):
        """Fractional composition of the Composition."""
        return _get_fractional_composition(self.formula)

    @property
    def num_atoms(self) -> float:
        """Total number of atoms in Composition."""
        return self._natoms

    @property
    def embedding_dim(self) -> int:
        """Dimension of the embedding."""
        return self.embedding.dim

    def as_dict(self) -> dict:
        # TO-DO: Need to create a dict representation for the embedding class
        """Return the CompositionalEmbedding class as a dict."""
        return {
            "formula": self.formula,
            "composition": self.composition,
            "fractional_composition": self.fractional_composition,
            # 'embedding':self.embedding.as_
        }
        # Se

        # Set an attribute
        pass

    def _mean_feature_vector(self) -> np.ndarray:
        """
        Compute a weighted mean feature vector based of the embedding.

        The dimension of the feature vector is the same as the embedding.

        """
        return np.dot(self.norm_stoich_vector, self.el_matrix)

    def _variance_feature_vector(self) -> np.ndarray:
        """Compute a weighted variance feature vector."""
        diff_matrix = self.el_matrix - self._mean_feature_vector()

        diff_matrix = diff_matrix**2
        return np.dot(self.norm_stoich_vector, diff_matrix)

    def _minpool_feature_vector(self) -> np.ndarray:
        """Compute a min pooled feature vector."""
        return np.min(self.el_matrix, axis=0)

    def _maxpool_feature_vector(self) -> np.ndarray:
        """Compute a max pooled feature vector."""
        return np.max(self.el_matrix, axis=0)

    def _range_feature_vector(self) -> np.ndarray:
        """Compute a range feature vector."""
        return np.ptp(self.el_matrix, axis=0)

    def _sum_feature_vector(self) -> np.ndarray:
        """Compute the weighted sum feature vector."""
        return np.dot(self.stoich_vector, self.el_matrix)

    def _geometric_mean_feature_vector(self) -> np.ndarray:
        """Compute the geometric mean feature vector."""
        return np.exp(np.dot(self.norm_stoich_vector, np.log(self.el_matrix)))

    def _harmonic_mean_feature_vector(self) -> np.ndarray:
        """Compute the harmonic mean feature vector."""
        return np.reciprocal(
            np.dot(self.norm_stoich_vector, np.reciprocal(self.el_matrix))
        )

    _stats_functions_dict = {
        "mean": "_mean_feature_vector",
        "variance": "_variance_feature_vector",
        "minpool": "_minpool_feature_vector",
        "maxpool": "_maxpool_feature_vector",
        "range": "_range_feature_vector",
        "sum": "_sum_feature_vector",
        "geometric_mean": "_geometric_mean_feature_vector",
        "harmonic_mean": "_harmonic_mean_feature_vector",
    }

    def feature_vector(self, stats: Union[str, list] = "mean"):
        """
        Compute a feature vector.

        The feature vector is a concatenation of
        the statistics specified in the stats argument.

        Args:
            stats (list): A list of strings specifying the statistics to be computed.
            The default is ['mean'].

        Returns:
            np.ndarray: A feature vector of dimension (len(stats) * embedding_dim).
        """
        implemented_stats = [
            "mean",
            "variance",
            "minpool",
            "maxpool",
            "range",
            "sum",
            "geometric_mean",
            "harmonic_mean",
        ]
        if isinstance(stats, str):
            stats = [stats]
        if not isinstance(stats, list):
            raise ValueError("Stats argument must be a list of strings")
        if not all([isinstance(s, str) for s in stats]):
            raise ValueError("Stats argument must be a list of strings")
        if not all([s in implemented_stats for s in stats]):
            raise ValueError(
                f" {[stat for stat in stats if stat not in implemented_stats]} "
                "are not valid statistics."
            )
        feature_vector = []
        for s in stats:
            feature_vector.append(getattr(self, self._stats_functions_dict[s])())
        return np.concatenate(feature_vector)

    def distance(
        self,
        comp_other,
        distance_metric: str = "euclidean",
        stats: Union[str, List[str]] = "mean",
    ):
        """
        Compute the distance between two compositions.

        Args:
            comp_other (Union[str, CompositionalEmbedding]): The other composition.
            distance_metric (str): The metric to be used. The default is 'euclidean'.

        Returns:
            float: The distance between the two CompositionalEmbedding objects.
        """
        if isinstance(comp_other, str):
            comp_other = CompositionalEmbedding(comp_other, self.embedding)
        if not isinstance(comp_other, CompositionalEmbedding):
            raise ValueError(
                "comp_other must be a string or a CompositionalEmbedding object."
            )
        if self.embedding_name != comp_other.embedding_name:
            raise ValueError(
                "The two CompositionalEmbedding objects must have the same embedding."
            )
        return _composition_distance(
            self, comp_other, self.embedding, distance_metric, stats
        )

    def __repr__(self):
        return (
            f"CompositionalEmbedding(formula={self.formula}, "
            f"embedding={self.embedding_name})"
        )

    def __str__(self):
        return (
            f"CompositionalEmbedding(formula={self.formula}, "
            f"embedding={self.embedding_name})"
        )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.formula == other.formula
                and self.embedding_name == other.embedding_name
            )
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.formula, self.embedding))


def _composition_distance(
    comp1: Union[str, CompositionalEmbedding],
    comp2: Union[str, CompositionalEmbedding],
    embedding: Optional[Union[str, Embedding]] = None,
    distance_metric: str = "euclidean",
    stats: Union[str, List[str]] = "mean",
) -> float:
    """Compute the distance between two compositions."""
    if not isinstance(comp1, CompositionalEmbedding):
        comp1 = CompositionalEmbedding(comp1, embedding=embedding)
    if not isinstance(comp2, CompositionalEmbedding):
        comp2 = CompositionalEmbedding(comp2, embedding=embedding)
    assert comp1.embedding_name == comp2.embedding_name

    comp1_vec = comp1.feature_vector(stats=stats)
    comp2_vec = comp2.feature_vector(stats=stats)

    scikit_metrics = ["euclidean", "manhattan", "chebyshev"]

    scipy_metrics = {"wasserstein": wasserstein_distance, "energy": energy_distance}

    # valid_metrics = scikit_metrics + list(scipy_metrics.keys()) + ["cosine"]

    if distance_metric in scikit_metrics:
        distance = DistanceMetric.get_metric(distance_metric)

        return distance.pairwise(
            comp1_vec.reshape(1, -1),
            comp2_vec.reshape(1, -1),
        )[
            0
        ][0]

    elif distance_metric in scipy_metrics.keys():
        return scipy_metrics[distance_metric](comp1_vec, comp2_vec)
    elif distance_metric == "cosine_distance":
        return cosine_distance(comp1_vec, comp2_vec)


def composition_featuriser(
    data: Union[pd.DataFrame, pd.Series, CompositionalEmbedding, list],
    formula_column: str = "formula",
    embedding: Union[Embedding, str] = "magpie",
    stats: Union[str, list] = "mean",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Compute a feature vector for a composition.

    The feature vector is based on the statistics specified
    in the stats argument.

    Args:
        data (Union[pd.DataFrame, pd.Series, list, CompositionalEmbedding]):
        A pandas DataFrame or Series containing a column named 'formula',
        a list of formula, or a CompositionalEmbedding class
        embedding (Union[Embedding, str], optional): A Embedding class or a string
        stats (Union[str, list], optional): A list of statistics to be computed.
        The default is ['mean'].
        inplace (bool, optional): Whether to perform the operation in place on the data.
        The default is False.

    Returns:
        Union[pd.DataFrame,list]: A pandas DataFrame containing the feature vector,
        or a list of feature vectors is returned
    """
    if isinstance(stats, str):
        stats = [stats]
    if isinstance(data, pd.Series):
        data = data.to_frame(name="formula")
    if isinstance(data, pd.DataFrame):
        if not inplace:
            data = data.copy()
        if formula_column not in data.columns:
            raise ValueError(
                f"The data must contain a column named {formula_column}  to featurise."
            )
        print("Featurising compositions...")
        comps = [
            CompositionalEmbedding(x, embedding)
            for x in tqdm(data[formula_column].tolist())
        ]
        print("Computing feature vectors...")
        fvs = [x.feature_vector(stats) for x in tqdm(comps)]
        feature_names = comps[0].embedding.feature_labels
        feature_names = [
            f"{stat}_{feature}" for stat in stats for feature in feature_names
        ]
        data_new = pd.concat([data, pd.DataFrame(fvs, columns=feature_names)], axis=1)
        # data.columns = []
        return data_new
    elif isinstance(data, list):
        comps = [CompositionalEmbedding(x, embedding) for x in data]
        return [x.feature_vector(stats) for x in tqdm(comps)]

    elif isinstance(data, CompositionalEmbedding):
        return data.feature_vector(stats)
    else:
        raise ValueError(
            "The data must be a pandas DataFrame, Series, "
            "list or CompositionalEmbedding class."
        )
