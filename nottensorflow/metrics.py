# import matplotlib.pyplot as plt
import numpy as np

class ConfusionMatrix:
    def __init__(self, true_labels, pred_labels, num_classes):
        """
        Expects the labels to already be encoded as class ID's
        """
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.num_classes = num_classes

        assert np.max(true_labels) <= num_classes
        assert np.max(pred_labels) <= num_classes

        self.matrix = np.zeros(shape=(num_classes, num_classes), dtype=int) # Row = pred, Col = true
        for i, j in zip(pred_labels, true_labels):
            self.matrix[i, j] += 1

    def display(self):
        # TODO: Return a matplotlib figure of `self.matrix`
        pass

    def print_statistics(self):
        # Accuracy
        # Precision
        # Recall
        # F1 Score
        pass

    def accuracy(self):
        correct = np.trace(self.matrix)
        total = np.sum(self.matrix)
        return correct / total
    
    def precision(self, C):
        """
        Precision = TP / (TP + FP) = True_Positive / All_Positive_Predictions

        Treats class `C` as the 'positive' class, and all other classes as 'negative'.
        """
        total_pos_pred = np.sum(self.matrix[:, C])
        true_pos = self.matrix[C, C]
        return true_pos / total_pos_pred
    
    def recall(self, C):
        """
        Recall = TP / (TP + FN) = True_Positive / All_Real_Positives

        Treats class `c` as the 'positive' class, and all other classes as 'negative'.
        """
        total_pos_real = np.sum(self.matrix[C, :])
        true_pos = self.matrix[C, C]
        return true_pos / total_pos_real

    def f1_score(self, C):
        """
        F1 = 2 * ((Precision * Recall) / (Precision + Recall))

        Treats class `c` as the 'positive' class, and all other classes as 'negative'.
        """
        precision = self.precision(C)
        recall = self.recall(C)
        return 2 * precision * recall / (precision + recall)


