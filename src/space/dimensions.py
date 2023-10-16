""" Hyper parametrization space dimension definition"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from skopt.space import Categorical, Dimension, Integer, Real


class BiDimension(Dimension, ABC):
    """
    Abstract dimension class. Bi variant of Scikit Optimize Dimension mother class.

    We add to skopt Dimension class a type attribute to differentiate features param from fit params and model params.
    A Dimension is here defined as a named hyperparameter axis range;
    an hyperparameter as a value drawn from this axis;
    and a search space as a list of space over which we want to optimize the model.
    """
    dim_type_lst = ['features', 'fit', 'model', 'sampling', 'text_variable_selection', 'numerical_variable_selection']

    @abstractmethod
    def __init__(self, dim_type: str) -> None:
        """
        Instantiate BiDimension

        :param dim_type: parameter type (features, fit, model, sampling,
            text_variable_selection, or numerical_variable_selection)
        """
        assert dim_type in self.dim_type_lst, f'Param type must be chosen from {self.dim_type_lst}'
        self.dim_type = dim_type


class IntegerDim(Integer, BiDimension):
    """
    Bi variant of Scikit Optimize Integer class.

    We add to skopt Integer class a type attribute to differentiate features params from fit params and model params.
    A Dimension is here defined as a named hyperparameter axis range;
    an hyperparameter as a value drawn from this axis;
    and a search space as a list of space over which we want to optimize the model.
    """

    def __init__(self, low: int, high: int, name: str, dim_type: str) -> None:
        """
        Instantiate IntegerDim

        :param low: parameter lower bound (included)
        :param high: parameter upper bound (included)
        :param name: string parameter name
        :param dim_type: parameter type (features, fit, model, or sampling)
        """
        if dim_type in ['text_variable_selection', 'numerical_variable_selection']:
            raise AssertionError(
                (
                    'The variable_selection dim_types are used only within the HyperparamTuner.' +
                    ' text_cols and numerical_cols are feature_params and should be provided as such'
                )
            )
        Integer.__init__(self, low=low, high=high, name=name)
        BiDimension.__init__(self, dim_type=dim_type)


class RealDim(Real, BiDimension):
    """
    Bi variant of Scikit Optimize Real class.

    We add to skopt Real class a type attribute to differentiate features params from fit params and model params.
    A Dimension is here defined as a named hyperparameter axis range;
    an hyperparameter as a value drawn from this axis;
    and a search space as a list of space over which we want to optimize the model.
    """

    def __init__(self, low: float, high: float, name: str, dim_type: str) -> None:
        """
        Instantiate RealDim

        :param low: parameter lower bound (included)
        :param high: parameter upper bound (included)
        :param name: string parameter name
        :param dim_type: parameter type (features, fit, model, or sampling)
        """
        if dim_type in ['text_variable_selection', 'numerical_variable_selection']:
            raise AssertionError(
                (
                    'The variable_selection dim_types are used only within the HyperparamTuner.' +
                    ' text_cols and numerical_cols are feature_params and should be provided as such'
                )
            )
        Real.__init__(self, low=low, high=high, name=name)
        BiDimension.__init__(self, dim_type=dim_type)


class CategoricalDim(Categorical, BiDimension):
    """
    Bi variant of Scikit Optimize Categorical class.

    We add to skopt Categorical class a type attribute to differentiate features params from fit params and model
    params.
    A Dimension is here defined as a named hyperparameter axis range;
    an hyperparameter as a value drawn from this axis;
    and a search space as a list of space over which we want to optimize the model.
    """

    def __init__(
        self, categories: List, name: str, dim_type: str, transform: str = 'onehot',
    ) -> None:
        """
        Instantiate CategoricalDim

        :param categories: list of categorical choices. Element of the list can be of any type.
        :param name: string parameter name
        :param dim_type: parameter type (features, fit, model, or sampling)
        """
        if dim_type in ['text_variable_selection', 'numerical_variable_selection']:
            raise AssertionError(
                (
                    'The variable_selection dim_types are used only within the HyperparamTuner.' +
                    ' text_cols and numerical_cols are feature_params and should be provided as such'
                )
            )
        Categorical.__init__(self, categories=categories, name=name, transform=transform)
        BiDimension.__init__(self, dim_type=dim_type)


class VariableSelectionDim(Categorical, BiDimension):
    """
    Bi sub variant of Scikit Optimize Categorical class for text variable selection.

    This class is used only within hyperparam tuner - numerical_cols and text_cols are feature_params and should be
    provided as such. The TextVariableDim is generated to break down the numerical_cols and text_cols params in
    a series of booleans over which the hyperparamtuner may optimize.
    """

    def __init__(self, name: str, variable_type: str, transform: str = 'onehot'):
        """
        Instantiate VariableSelectionDim

        :param name: variable string name
        :param variable_type: variable type; either text or numerical
        :param transform: type of encoding; by default onehot encoding
        """
        assert variable_type in ['text', 'numerical'], 'Only text or numerical variables are supported for selection'
        Categorical.__init__(self, categories=[True, False], name=name, transform=transform)
        BiDimension.__init__(self, dim_type=variable_type + '_variable_selection')
