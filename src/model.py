#!/usr/bin/env python
# coding: utf-8

# In[337]:


import pandas as pd
import numpy as np
import category_encoders
import yaml
import os
import importlib
import xgboost
import lightgbm
import warnings
import sklearn
import imblearn
from sklearn.experimental import enable_iterative_imputer

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

from joblib import dump, load
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import get_scorer
from imblearn.pipeline import Pipeline

from dataprep import load_data, prep_data, load_config, file_path

def split_training_test_sets(df, test_size, random_state = 42):
    X = df.drop(columns=['repeat_customer'])
    y = df['repeat_customer']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state = random_state)
    return X_train, X_test, y_train, y_test

def get_func(import_module: str, function_name: str, function_params: dict = {}):
    function_class = getattr(importlib.import_module(import_module), function_name)
    function = function_class(**function_params)  # Instantiates the model
    return function

def pipeline(X_train, y_train):
    
    print('Starting pipeline...')
    def do_cross_validation(pipe, X, y, model, scoring_methods='accuracy', cv=10):
        cv = cross_validate(pipe, X, y, scoring=scoring_methods, cv=cv)
        d = {}
        d['model'] = model
        for key in cv.keys():
            d[key] = np.mean(cv[key])
        return d
    
    print('Loading models...')
    models = {}
    for module, function in config['models'].items():
        models[function] = get_func(module, function)
        
    print('Loading model parameters...')
    params = config['params']

    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    
    sampler = get_func(config['sampling_method_lib'],
                   config['sampling_method_func'],
                   config['sampling_method_params'])
    
    print('Setting up Pipeline...')
    
    numerical_transformer = Pipeline(
    steps=[('imputer', get_func(config['num_imputer_lib'],
                                config['num_imputer_func'],
                                config['num_imputer_params'])),
           ('scaler', get_func(config['num_scaler_lib'],
                                config['num_scaler_func'],
                                config['num_scaler_params']))])
    
    categorical_transformer = Pipeline(
    steps=[('imputer', get_func(config['cat_imputer_lib'],
                                config['cat_imputer_func'],
                                config['cat_imputer_params'])),
           ('encoder', get_func(config['cat_encoder_lib'],
                                config['cat_encoder_func'],
                                config['cat_encoder_params']))])
    
    
    processor = ColumnTransformer(transformers=[('numerical', numerical_transformer, numeric_features),
                                                ('categorical', categorical_transformer, categorical_features)])
    
    
    results = []
    results_cols = None
    cv_scoring = config['scoring']
    pipes = {}
    
    for model in models.keys():
        print(f"Finding optimal hyperparameters for {model} model...")
        pipe = Pipeline(steps=[('processor', processor),
                               ('sampling', sampler),
                               ('classifier', models[model])])

        random_cv = RandomizedSearchCV(
                                       estimator = pipe,
                                       param_distributions = params[model],
                                       cv = config['RandomizedSearchCV_cv'],
                                       n_iter = config['RandomizedSearchCV_n_iter'],
                                       scoring = config['RandomizedSearchCV_scoring'],
                                       n_jobs = config['RandomizedSearchCV_n_jobs'],
                                       verbose = config['RandomizedSearchCV_scoring_verbose'], 
                                       return_train_score = config['RandomizedSearchCV_return_train_score'],
                                       random_state = config['RandomizedSearchCV_random_state']
                                      )
        
        random_cv.fit(X_train, y_train)
        best_pipe = random_cv.best_estimator_
        pipes[model] = best_pipe
        result_row = do_cross_validation(best_pipe, X_train, y_train, model, cv_scoring, cv=config['cv'])
        results.append(result_row)
        results_cols = list(result_row.keys())
    
    results = pd.DataFrame(results, columns=results_cols)
    return results, pipes, models

def evaluate_models(pipes, X_train, X_test, y_train, y_test):
    
    print('Evaluating models...')
    scoring = config['scoring']
    results = []
    cols = ['model'] + scoring 
    
    if type(pipes) is not dict:
        pipes.fit(X_train, y_train)
        preds = pipes.predict(X_test)
        result_row = {}
        result_row['model'] = pipes['classifier'].__class__.__name__
        for score in scoring:
            result_row[score] = get_scorer(score)._score_func(y_test, preds) 
        results.append(result_row)
        
        final_results = pd.DataFrame(results, columns=cols)
        return final_results
    else:
        for pipe in pipes.values():
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            result_row = {}
            result_row['model'] = pipe['classifier'].__class__.__name__
            for score in scoring:
                result_row[score] = get_scorer(score)._score_func(y_test, preds) 
            results.append(result_row)
        final_results = pd.DataFrame(results, columns=cols)
        return final_results

if __name__ == "__main__":
    config = load_config("config.yaml")
    data = load_data()
    df = prep_data(data)
    
    # Perform train_test_split
    X_train, X_test, y_train, y_test = split_training_test_sets(df, 
                                                                test_size=config['train_test_split_test_size'], 
                                                                random_state = config['train_test_split_random_state']
                                                               )
    
    if config['run'] == True:
        # Find best hyperparameters on training data for each model using k-fold cross validation
        cv_results, pipes, models = pipeline(X_train, y_train)
        print("Cross Validation Results on Training Data:")
        print(cv_results.sort_values(by='test_roc_auc', ascending=False))
        # Save results
        cv_results.to_csv(file_path('../results/cv_results.csv'))
        # Predict on test dataset
        print("Final Results on Test Data:")
        final_results = evaluate_models(pipes, X_train, X_test, y_train, y_test).sort_values(by='roc_auc', ascending=False)
        print(final_results)
        # Save results
        final_results.to_csv(file_path('../results/final_results.csv'))

    if config['save'] == True:
        # Save model(s) to joblib file
        print('Model Saved!')
        dump(pipes, file_path('../saved_model/saved_model.joblib'))
        

    if config['load'] == True:
        # Load model from joblib file
        print('Model Loaded!')
        pipe = load(file_path('../saved_model/saved_model.joblib'))
        print("Results on Test Data:")
        saved_model_results = evaluate_models(pipe, X_train, X_test, y_train, y_test)
        print(saved_model_results)
        saved_model_results.to_csv(file_path('../saved_model/saved_model_results.csv'))









