# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

class ConfusionMatrix:
    def __init__(self, true_labels, pred_labels, num_classes: int):
        """
        Expects `true_label` and `pred_label` to already be enoded as class ID's, starting from 0.
        """
        self.num_classes = num_classes
        assert np.max(true_labels) <= num_classes
        assert np.max(pred_labels) <= num_classes

        self.matrix = np.zeros(shape=(num_classes, num_classes), dtype=int) # Row = pred, Col = true
        for i, j in zip(pred_labels, true_labels):
            self.matrix[i, j] += 1

    def display(self):
        """
        Returns a matplotlib `Table` representing the confusion matrix. 
        Column labels are the integer encodings of the table. 
        
        If you want the labels to be something different, you will have to manually 
        edit them before calling `plt.show()`.
        """
        _, ax = plt.subplots()
        ax.axis('off')  # turn off the axis

        # Set rows & cols
        classes = [str(c + 1) for c in range(self.num_classes)]
        rows = cols = classes
        # Set Data
        table_data: list[list[str]] = self.matrix.astype(str).tolist()

        # Instantiate table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        return table
    
    def print_statistics(self):
        print(f'Accuracy: {self.accuracy():.4f}')
        for cls in range(self.num_classes):
            print(f'Precision {cls}: {self.precision(cls):.4f}')
            print(f'Recall {cls}: {self.precision(cls):.4f}')
            print(f'F1 Score {cls}: {self.precision(cls):.4f}')

    def accuracy(self):
        correct = np.trace(self.matrix)
        total = np.sum(self.matrix)
        return correct / total
    
    def precision(self, C: int):
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


