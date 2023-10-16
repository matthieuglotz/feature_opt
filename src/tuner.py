"""
Hyper parameters Tuner over features parameters, fit parameters and model parameters, and data
sampling parameters.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
from sklearn.metrics import f1_score
from skopt import gp_minimize
from skopt.callbacks import DeltaXStopper
from skopt.space import Categorical
from woob.tools.log import getLogger

from space.dimensions import BiDimension

if TYPE_CHECKING:
    import logging

    from budgea.plugins.categorization_ds.features_generator import FeaturesGenerator
    from budgea.plugins.categorization_ds.text_vectorizer import BiVectorizer
    from model_training.ml_models import BiMLModel


class HyperParamTuner:
    """
    Bayesian hyper parameter tuner.

    (i) Initialise by randomly drawing n_initial_points (parameter from optimizer_params) from the search space and
    evaluate bi_model scores at these points using the evaluate_model method.

    (ii) Then use points scores to fit a surrogate model proxying the posterior distribution of scores to
    hyperparameters.

    (iii) Acquire the optimal point of the surrogate model as the next set of hyperparameter values to evaluate.

    (iv) Evaluate the score of the optimal point using the evaluate_model method (back to step (i)). The loop ends when
    the number of points evaluated reaches n_samples.

    See link below for more information:
    https://machinelearningmastery.com/what-is-bayesian-optimization/
    """
    DEFAULT_SCORER_PARAMS = {f1_score: {'average': 'macro'}}
    DEFAULT_MINIMIZER_PARAMS = {
        'callback': DeltaXStopper(1e-8), 'n_initial_points': 10, 'verbose': True, 'random_state': 13, 'n_calls': 100,
    }

    def __init__(
        self, logger: logging.Logger, category_lst: List, keywords_df: pd.DataFrame,
        train_set: pd.DataFrame, train_features: FeaturesGenerator, score_set: pd.DataFrame,
        score_features: FeaturesGenerator, search_space: List[BiDimension], bi_model: Type[BiMLModel],
        vectorizer: Type[BiVectorizer], tune_scorer: Callable = f1_score,
        tune_minimizer: Callable = gp_minimize, tune_scorer_kwargs: Optional[dict] = None,
        minimizer_params: Optional[dict] = None, outfile: str = None, max_train_time: int = 7200,
    ) -> None:
        """
        Instantiate the HyperParamTuner.

        :param logger: logger object
        :param category_lst: config provided categories names.
        :param keywords_df: Pandas dataframe with keywords scores
        :param train_set: cleaned Pandas dataframe with training labelled transaction data.
        :param train_features: Instantiated CategoFeatures object with the training set features. May not be run.
        :param score_set: cleaned Pandas dataframe with scoring labelled transaction data.
        :param score_features: Instantiated CategoFeatures object with the scoring set features. May not be run.
        :param search_space: list of Scikit-Optimize style search parameters for the Feature Generator. e.g.
        [Integer(1, 6, name='freq_months', type='features'), Real(0, 1, name='lambda', type='model')
        :param bi_model: BiModel object whose hyperparameters we want to optimize.
        :param tune_scorer: sklearn metrics scorer. Default f1_score
        :param tune_scorer_kwargs: scorer keyword arguments. Default None. If None, uses average='macro' for f1_score.
        :param tune_minimizer: Optional Scikit minimizer. Default Gaussian Process Minimizer.
        :param minimizer_params: dictionnary including minimizer params such as n_calls for the number of iterations
        :param max_train_time: max model train time in seconds
        """
        # General run, config, and logger attributes
        self.logger = getLogger(self.__class__.__name__, parent=logger)
        self.category_lst = category_lst
        if outfile and os.path.exists(outfile):
            raise FileExistsError('Results must be saved in a new file.')
        self.outfile = outfile
        self.results_df = pd.DataFrame()

        # Feature and keyword attributes
        self.keywords_df = keywords_df
        self.train_set = train_set
        self.train_base_feature = train_features
        self.train_label = np.array([])
        self.train_feature_dict = {}
        self.sampled_feature_dict = {}
        self.score_set = score_set
        self.score_base_feature = score_features
        self.score_label = np.array([])
        self.score_feature_dict = {}
        self.default_vectorizer = vectorizer
        self.max_train_time = max_train_time

        # Hyper parametrization attributes

        # Set search space
        self.search_space = self.process_search_space(search_space)

        # Insert non kwarg hyper parametrization attributes
        self.bi_model = bi_model
        self.scorer = tune_scorer
        self.minimizer = tune_minimizer
        self.counter = 1

        # Handle kwarg arguments; start from default, update with provided if provided.
        self.scorer_params = self.DEFAULT_SCORER_PARAMS.get(tune_scorer, {})
        if tune_scorer_kwargs is not None:
            self.scorer_params.update(tune_scorer_kwargs)

        self.minimizer_params = self.DEFAULT_MINIMIZER_PARAMS
        x0, y0 = self.get_default_result()
        self.minimizer_params['x0'] = x0
        self.minimizer_params['y0'] = y0
        if minimizer_params is not None:
            self.minimizer_params.update(minimizer_params)

    def optimize(self) -> Tuple[dict, dict, dict, dict]:
        """
        Return the optimal set of features, fit, and model parameters.

        The Scikit Optimize optimizer runs the operation of (i) sampling n_initial_points, (ii) updating the surrogate
        model, (iii) acquiring the optimal hyperparameter point, (iv) returning the optimal once n_samples have been
        evaluated.

        :return feature_params:dict, fit_params:dict, model_params:dict
        each of these dictionaries is structured as: {"dimension1_name": value, "dimension2_name": value, ...}
        Note that each of the values provided is the optimal parameter value estimated by the optimizer
        """
        optimal_params = self.minimizer(self.evaluate_model, self.search_space, **self.minimizer_params)
        return self.process_params(hyperparams=optimal_params, search_space=self.search_space)

    def evaluate_model(self, hyperparams: List) -> float:
        """
        Evaluate model score with the given hyper parameters, using the providing train and score sets.

        The function (i) names hyperparams and sort them by type;
        then (ii) if some feature_params were specified it recalculates train and score features;
        then (iii) it retrains the model using the specified fit and model params;
        then (iv) evaluates the trained model score using the score labels.

        :param hyperparams: list of hyperparameter values, provided by the Scikit Optimize optimizer

        :return score: float 1 - score. We aim to maximize the score but the optimizers are minimizers. We therefore
            provide 1 - score because minimizing 1 - score is equivalent to maximizing the score.
        """
        # (i) Process hyperparams
        feature_params, fit_params, model_params, sampling_params = self.process_params(hyperparams, self.search_space)
        self.logger.info(
            "Evaluating point %d using features params %s, fit_params %s, model_params %s, sampling_params %s",
            self.counter,
            feature_params,
            fit_params,
            model_params,
            sampling_params,
        )

        # (ii) Recompute features if features params were provided
        try:
            feature_key = str(feature_params)
            train_features = self.train_feature_dict.get(feature_key, None)  # Check if pre calculated
            if not train_features:
                self.logger.info('Recalculating train features')

                train_features = self.run_features(
                    features=self.train_base_feature,
                    clean_df=self.train_set,
                    keywords_df=self.keywords_df,
                    feature_params=feature_params,
                )
                self.train_feature_dict[feature_key] = train_features

                # Label must be calculated from features to respect transaction order.
                # However they do not need to be recalculated everytime we recalculate features.
                if np.any(self.train_label):
                    self.train_label = map_category_name_to_id(
                        cat_series=train_features.freq_features[DataColNames.MOTHER_CAT],
                        categories=train_features.categories,
                    )

            score_features = self.score_feature_dict.get(feature_key, None)  # Check if pre calculated
            if not score_features:
                self.logger.info('Recalculating score features')
                score_features = self.run_features(
                    features=self.score_base_feature,
                    clean_df=self.score_set,
                    keywords_df=self.keywords_df,
                    feature_params=feature_params,
                )
                self.score_feature_dict[feature_key] = score_features

                # Label must be calculated from features to respect transaction order.
                # However they do not need to be recalculated everytime we recalculate features.
                if np.any(self.score_label):
                    self.score_label = map_category_name_to_id(
                        cat_series=score_features.freq_features[DataColNames.MOTHER_CAT],
                        categories=score_features.categories,
                    )

            # (iib) Resample if required
            if sampling_params:
                sampling_params_key = str(sampling_params)
                combined_key = feature_key + sampling_params_key
                train_features_final, train_labels_final = self.sampled_feature_dict.get(
                    combined_key, (None, None),
                )
                if train_features_final is None and train_labels_final is None:
                    model_trainer = ModelTrainer(logger=self.logger, categories=train_features.categories)
                    train_features_final, train_labels_final = model_trainer.resample(
                        train_features=self.train_feature_dict[feature_key],
                        train_labels=self.train_label,
                        sampling_pipeline=sampling_params['sampling_pipeline'],
                    )
                    self.sampled_feature_dict[combined_key] = train_features_final, train_labels_final
            else:
                train_features_final, train_labels_final = self.train_feature_dict[feature_key], self.train_label

            # (iii) Fit model.
            self.logger.info('Training model %s', self.bi_model.__name__)
            clock_start = datetime.now()
            model = self.bi_model(logger=self.logger)
            with time_limit(self.max_train_time):
                model.train(
                    train_features=train_features_final,
                    train_label=train_labels_final,
                    valid_features=self.score_feature_dict[feature_key],
                    valid_label=self.score_label,
                    category_lst=self.category_lst,
                    model_params=model_params,
                    fit_params=fit_params,
                )
            train_time = datetime.now() - clock_start
            clock_start = datetime.now()

            # (iv) Predict results on validation set. Evaluate model score.
            predicted_label = model.predict(features=self.score_feature_dict[feature_key])
            predict_time = datetime.now() - clock_start
            if len(np.shape(self.score_label)) > 1:  # LSTM uses label which has 1 col per category
                score_label = np.argmax(self.score_label, axis=1)
            else:
                score_label = self.score_label
            score = self.scorer(y_true=score_label, y_pred=predicted_label, **self.scorer_params)
            self.logger.info('Point %d evaluation ended. Score: %.2f', self.counter, score)

        except Exception as error:
            self.logger.error(
                'Error %s at evaluation point %d. registering 0 score.',
                error,
                self.counter,
                exc_info=True,
            )
            score = 0
            train_time = 0
            predict_time = 0

        self.counter += 1

        # Save results to outfile, return 1 - score, as the optimizer is a minimizer (better must be lower)
        if self.outfile:
            iter_df = pd.concat(
                [
                    self.get_param_df(feature_params, 'features'),
                    self.get_param_df(fit_params, 'fit'),
                    self.get_param_df(model_params, 'model'),
                    self.get_param_df(sampling_params, 'sampling'),
                ],
                axis=1
            )
            iter_df['score'] = score
            iter_df['train_time'] = train_time
            iter_df['predict_time'] = predict_time
            self.results_df = pd.concat([self.results_df, iter_df], axis=0)

            if self.counter % 10 == 0 or self.counter == self.minimizer_params['n_calls']:
                header_bool = ~os.path.exists(self.outfile)  # Add headers only on first run, not after
                try:
                    self.results_df.to_csv(self.outfile, mode='a', header=header_bool)
                except Exception as error:
                    self.logger.error(
                        'Outfile export failed: %s',
                        error,
                        exc_info=True,
                    )

        return 1 - score

    @classmethod
    def process_params(cls, hyperparams, search_space):
        """
        Name and sort hyperparams and returns them by type.

        BI version of the Scikit Optimize wrapper use_named_args. We use a function instead of a decorator to be able
        to use it inside a class (a decorator would not take an instance attribute as argument). Further, we do not use
        Scikit Optimize decorator because it provides hyperparameters as a flat dictionary, whilst we want to separate
        parameters according to their type.

        :param hyperparams: list of hyperparam values (e.g. [1, 0.91, 'linear'])
        :param search_space: list of dimensions over which we want to optimize the model.
        (e.g. [
            Integer(1, 3, 'past_months', 'features),
            Real(0, 1, 'sub_sampling', 'model'),
            Categorical(['linear', 'quadratic'], 'kernel', 'onehot', 'model'),
        ]

        :return: tuple with feature_params:dict, fit_params:dict, model_params:dict
        each of these dictionaries is structured as: {"dimension1_name": value, "dimension2_name": value, ...}
        """
        # Check type - after random initialization optimizer returns OptimizerResult object instead of list
        if isinstance(hyperparams, OptimizeResult):
            hyperparams = hyperparams.x

        # Check that the number of hyper parameters matches the search space's length
        cls.check_search_space_len(hyperparams=hyperparams, search_space=search_space)

        # Create a dict where the keys are the names of the search space dimensions
        # and the values are taken from the list of hyperparameters hyperparams.
        feature_param, fit_params, model_params, sampling_params, text_variables, numerical_variables = (
            {param.name: value for param, value in zip(search_space, hyperparams) if param.dim_type == dim_type}
            for dim_type in BiDimension.dim_type_lst
        )
        if text_variables:
            text_cols = [variable for variable, keep in dict(text_variables).items() if keep]
            feature_param['text_cols'] = text_cols
        if numerical_variables:
            numerical_cols = [variable for variable, keep in dict(numerical_variables).items() if keep]
            feature_param['numerical_cols'] = numerical_cols

        return feature_param, fit_params, model_params, sampling_params

    def run_features(self, features, clean_df, keywords_df, feature_params=None):
        """
        Calculate features from a clean transaction df (train set or score set)

        :param features: instantiated CategoFeature object
        :param clean_df: cleaned transaction dataset with labels
        :param keywords_df: Pandas Keyword DataFrame
        :param feature_params: optional params for FeatureGenerator. Default None, no params passed.

        :returns features: FeatureGenerator object with features calculated
        """
        # If feature_kwargs are none return empty dict. Dict not default value because dict are mutable.
        if feature_params is None:
            feature_params = {}

        # If feature_params do not include a vectorizer, use vectorizer passed at init by default.
        if 'vectorizer' not in feature_params:
            feature_params['vectorizer'] = self.default_vectorizer

        # Calculate features
        features.run(
            bi_model=self.bi_model,
            transac_df=clean_df,
            keywords_df=keywords_df,
            **feature_params,
        )

        return features

    def process_search_space(self, search_space):
        """
        Process search space argument

        Check whether search space name and type are correct. Also reconstruct, as a helper, some search space
        dimensions, such as variable_selection feature_params.
        :param search_space: list of dimensions over which we want to optimize the model.
        """
        self.check_search_space_types(search_space)
        self.check_search_space_names(search_space)

        # Rework variable_selection dimensions
        processed_search_space = []
        for dimension in search_space:
            # We search for dim names to allow these to be passed as feature_params.
            if dimension.name in ['text_cols', 'numerical_cols']:
                for var in dimension.categories:
                    processed_search_space.append(
                        VariableSelectionDim(
                            name=var,
                            variable_type=dimension.name.split('_')[0]
                        )
                    )
            else:
                processed_search_space.append(dimension)

        return processed_search_space

    def get_default_result(self):
        """
        Get accuracy score using default param values for the optimized dimensions

        :return x0, y0: the default hyperparameter values and their score
        """
        # Get default features, fit, model, and sampling values
        default_dict = {
            'features': self.train_base_feature.DEFAULT_PARAMS,
            'model': self.bi_model.DEFAULT_PARAMS,
            'fit': self.bi_model.DEFAULT_FIT_PARAMS,
            'sampling': ModelTrainer.DEFAULT_SAMPLING,
        }

        # Get x0, the default dimension values for the specified search_space
        x0 = []
        for dim in self.search_space:
            if dim.dim_type not in ('numerical_variable_selection', 'text_variable_selection'):
                default_value = default_dict[dim.dim_type].get(dim.name)
                if default_value is None:
                    self.logger.info(
                        'No default for %s hyperparam %s, initializing hyperparam with random values',
                        dim.dim_type,
                        dim.name,
                    )
                    return None, None
                if Categorical in dim.__class__.mro():
                    x0.append(default_value)

                # Confirm that default value is allowed
                else:
                    if dim.low <= default_value <= dim.high:
                        x0.append(default_value)
                    else:
                        self.logger.info(
                            'Dimension %s default value %.4f is not between boundaries %.4f and %.4f',
                            dim.name,
                            default_value,
                            dim.low,
                            dim.high,
                        )
                        self.logger.info('Initializing hyperparam with random values')
                        return None, None
            else:
                x0.append(dim.name in default_dict['features'][dim.dim_type.split('_')[0] + '_cols'])

        # Compute y0
        self.logger.info('Fitting current default as first model')
        y0 = self.evaluate_model(x0)
        self.logger.info('Default param values yield a score of %d', y0)
        return x0, y0

    @staticmethod
    def check_search_space_types(search_space: List[BiDimension]) -> None:
        """
        Check whether all elements of a list are of BiDimension class and raise a ValueError if they are not.

        Raises `TypeError`, If one or more element in the list hyperparams is not a BiDimension. Implementation
        inspired from Scikit Optimize utils check_list_type.

        :param search_space: list of dimensions over which we want to optimize the model.
        """

        # List of the elements in the list that are incorrectly typed.
        err = list(filter(lambda a: not isinstance(a, BiDimension), search_space))

        # If the list is non-empty then raise an exception.
        if len(err) > 0:
            msg = "All elements in list must be instances of {}, but found: {}"
            msg = msg.format(BiDimension, err)
            raise TypeError(msg)

    @staticmethod
    def check_search_space_names(search_space: List[BiDimension]) -> None:
        """
        Check whether all elements of a list have a non null name attribute.

        Raises `AttributeError`, If one or more element in the list hyperparams do not have a name attribute.
        Implementation inspired from Scikit Optimize utils check_dimension_names.

        :param search_space: list of dimensions over which we want to optimize the model.
        """

        # List of the dimensions that have no names.
        try:
            err_dims = list(filter(lambda dim: dim.name is None, search_space))

        except AttributeError:
            raise AttributeError("All dimensions must have names")

        # If the list is non-empty then raise an exception.
        if len(err_dims) > 0:
            msg = "All dimensions must have names, but found: {}"
            msg = msg.format(err_dims)
            raise AttributeError(msg)

    @staticmethod
    def check_search_space_len(hyperparams: List, search_space: List[BiDimension]) -> None:
        """
        Check that the number of dimensions matches the number of hyperparams.

        Raises ValueError if lengths don't match.

        :param hyperparams: list of hyperparam values (e.g. [1, 0.91, 'linear'])
        :param search_space: list of dimensions over which we want to optimize the model.
        (e.g. [
            Integer(1, 3, 'past_months', 'features),
            Real(0, 1, 'sub_sampling', 'model'),
            Categorical(['linear', 'quadratic'], 'kernel', 'onehot', 'model'),
        ])
        """
        if len(search_space) != len(hyperparams):
            msg = "Mismatch in number of hyperparams: len(hyperparams)=={} and len(x)=={}"
            msg = msg.format(len(search_space), len(hyperparams))
            raise ValueError(msg)

    def get_param_df(self, param_dict: dict, dim_type: str) -> pd.DataFrame:
        """
        Create a dataframe from a param dictionary.

        Type is added to col names to disambiguate fit, features, and model parameters whose have the same name.
        :param param_dict: dictionary of hyperparam values (e.g. {'freq_months': 1, 'min_proba': 0.91})
        :param dim_type: string describing the type of the param (i.e. features, fit, or model)

        :return param_df: Single row DataFrame with dim_type + '_' + param names as columns, and param values as values
        """
        str_dict = {key: str(value) for key, value in param_dict.items()}
        param_df = pd.DataFrame(str_dict, index=[self.counter], columns=str_dict.keys())
        param_df.columns = [dim_type + '_' + col for col in param_df.columns]
        return param_df
