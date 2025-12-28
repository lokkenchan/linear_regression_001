import os
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2,
                             root_mean_squared_error)

from mlflow.models import infer_signature

from sklearn.model_selection train_test_split, GridSearchCV
from urllib.parse import urlparse # get schema of remote repo url

