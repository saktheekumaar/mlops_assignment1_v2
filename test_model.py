
import unittest
from sklearn.metrics import accuracy_score
from wine_qlty import lr, rf, svm, X_test, y_test  # Assuming models and data are in wine_quality_model.py

class TestWineQualityModel(unittest.TestCase):

    def test_logistic_regression_accuracy(self):
        predictions = lr.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        self.assertGreater(accuracy, 0.5, "Logistic Regression accuracy is not greater than 0.5")

    def test_random_forest_accuracy(self):
        predictions = rf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        self.assertGreater(accuracy, 0.5, "Random Forest accuracy is not greater than 0.5")

    def test_svm_accuracy(self):
        predictions = svm.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        self.assertGreater(accuracy, 0.5, "SVM accuracy is not greater than 0.5")

if __name__ == '__main__':
    unittest.main()
