import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, homogeneity_score
from sklearn.preprocessing import MinMaxScaler, Normalizer
from functools import partial
from tqdm import tqdm
import random
from opendataval import dataval
from opendataval.dataloader import DataFetcher
from opendataval.dataval import (
    KNNShapley, LavaEvaluator
)
from opendataval.model import ClassifierSkLearnWrapper
from tqdm import tqdm


class Trainer:
    def __init__(self, model='softmax', num_classes=2):
        self.model = self.get_model(model)
        self.num_classes = num_classes
        
    def get_model(self, model): 
        if model == 'softmax':
            return LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
        elif model == 'rf':
            return RandomForestClassifier(n_estimators=10, max_depth=5)
        elif model == 'knn':
            return KNeighborsClassifier(n_neighbors=self.num_classes)
        else:
            return LinearRegression()

    def compute_utility(x_train, y_train, x_test, y_test):
        """
        Train a softmax regression model on seller data and evaluate on buyer's test data.
        
        Args:
            x_train: Training features
            y_train: Training labels
            x_test: Test features
            y_test: Test labels
        
        Returns:
            Utility score (accuracy)
        """
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return accuracy_score(y_test, y_pred)



def get_model_valuation(
    x_train, y_train, 
    x_val, y_val, 
    value_method="KNNShapley", 
    num_classes=2, 
    return_data_values=False,
):
    """
    Get data valuation using OpenDataVal framework
    
    Args:
        x_train: Training features
        y_train: Training labels
        x_val: Validation features
        y_val: Validation labels
        value_method: Valuation method
        num_classes: Number of classes
        return_data_values: Whether to return individual data values or average
        
    Returns:
        Data values or average value
    """

    wrap = ClassifierSkLearnWrapper
    if task == "bin":
        pred_model = wrap(LogisticRegression, num_classes=2, max_iter=100)
    elif task == "multi":
        pred_model = wrap(RandomForestClassifier, num_classes=num_classes, n_estimators=10, max_depth=5)
    else:
        pred_model = wrap(KMeans, num_classes=num_classes, n_clusters=num_classes, n_init='auto')

    fetcher = DataFetcher.from_data_splits(
        x_train, y_train, x_val, y_val, x_val, y_val, one_hot=False
    )

    valuer = getattr(dataval, value_method)()
    valuer.train(fetcher=fetcher, pred_model=pred_model)
    data_values = valuer.data_values
    seller_value = np.mean(data_values)
    
    if return_data_values:
        return data_values
    else:
        return seller_value


