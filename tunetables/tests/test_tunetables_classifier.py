import unittest

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tunetables.scripts.transformer_prediction_interface import TuneTablesClassifier


class TestTuneTablesClassifierSingle(unittest.TestCase):
    def test_main(self):
        x, y = load_breast_cancer(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        clf = TuneTablesClassifier()    
        clf.fit(x_train, y_train)
        y_eval = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_eval)
        print("Accuracy:", np.round(accuracy, 2))
