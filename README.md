# feature_opt
Scikit Opt based hyperparametrisation framework optimising feature parameters jointly with model and fit parameters

## Problem statement

Existing hyperparameter tuning framework, such as scikit-optimise, provide a wide range of technique for model parameter optimisation, ranging from brute-force grid search to more advanced Bayesian Optimization search techniques. However, the existing libraries focus solely on model and fit parameters - no support is provided to optimize feature selection and features parameters. This gap is particularly relevant as feature selection and parameters have been shown by research to both have a significant impact on model performance but also on the selection of model hyperparameters themselves (see Jie, Jiawei 2018); with the litterature describing how this can be particularly relevant for NLP (see Daelemans, Hoste, et al 2010).

This project aims to answer this by providing a framework to jointly optimize feature extraction parameters, feature selection, model fitting and model parameters, and even data sampling, using the latest Bayesian Optimization search techniques.

## Framework

This project builds on Scikit Opt optimization framework, by which parameters are associated with a dimension in the optimization space, feature parameters being just another dimensions amongst the other parameters being optimized. The entire prediction pipeline is rerun when using new feature parameters combinations, but for optimization the algorithm caches past feature combinations not to recalculate features if using the same features parameters with different model parameters. Optimization is operated by default using Scikit-Optimizer Gaussian Process minimizer, but can handle any Scikit-Opt style optimizer.

The overall process for an hyperparametrization_run is as follows: 
    
    (i) Initialise by randomly drawing n_initial_points (parameter from optimizer_params) from the search space and
    evaluate bi_model scores at these points using the evaluate_model method.
    
    (ii) Then use points scores to fit a surrogate model proxying the posterior distribution of scores to
    hyperparameters.
    
    (iii) Acquire the optimal point of the surrogate model as the next set of hyperparameter values to evaluate.
    
    (iv) Evaluate the score of the optimal point using the evaluate_model method (back to step (i)). The loop ends when
    the number of points evaluated reaches n_samples.

## Practical use

First define a hyperparametrisation space, i.e. a list of dimensions (e.g. feature or model parameters) over which the model pipeline should be optimized:
```
lstm_params = [
    RealDim(low=0.000001, high=0.01, name='learning_rate', dim_type='model'),
    RealDim(low=0, high=0.3, name='dropout_lstm', dim_type='model'),
    RealDim(low=0, high=0.3, name='dropout_numerical', dim_type='model'),
    RealDim(low=0, high=0.3, name='dropout_final', dim_type='model'),
    RealDim(low=0, high=0.8, name='shrink_rate', dim_type='model'),
    IntegerDim(low=8, high=256, name='num_unit_lstm', dim_type='model'),
    IntegerDim(low=8, high=256, name='num_unit_numerical', dim_type='model'),
    IntegerDim(low=1, high=2, name='num_layer_numerical', dim_type='model'),
    IntegerDim(low=8, high=256, name='num_unit_final', dim_type='model'),
    IntegerDim(low=1, high=2, name='num_layer_final', dim_type='model'),
    IntegerDim(low=8, high=256, name='batch_size', dim_type='fit'),
    IntegerDim(low=5, high=50, name='epochs', dim_type='fit'),
    CategoricalDim(
        categories=[
            'value', 'frequency', 'Logement', 'Prêt', 'Revenu', 'Autres', 'max',
            'Logement_dummy', 'Prêt_dummy', 'Revenu_dummy', 'Autres_dummy', 'top1', 'has_company',
            'has_person',
        ],
        dim_type='features',
        name='numerical_cols',
    ),
]
```

Second call the HyperparamTuner object, pass it the pipeline parameters, and call it's optimize method to get back the optimal feature, fit, model, and sampling parameters.
```
tuner = HyperParamTuner(
    logger=logger, category_lst=[category.name for category in categories],
    keywords_df=keywords_df, train_set=train_features.freq_features,
    train_features=train_features,
    score_set=valid_features.freq_features, score_features=valid_features,
    search_space=hyperparam_search_space, bi_model=model_class, vectorizer=vectorizer,
    minimizer_params=kwargs.get('minimizer_params', {}),
    outfile=kwargs.get('hyper_param_outfile', None),
    max_train_time=kwargs.get('max_train_time', 7200),
)
feature_params, fit_params, model_params, sampling_params = tuner.optimize()
```
