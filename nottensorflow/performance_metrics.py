from matplotlib import pyplot as plt, cm, colors, colormaps as cmaps
import numpy as np

class ConfusionMatrix:
    def __init__(self, true_labels: np.ndarray, pred_labels: np.ndarray, num_classes: int):
        """
        Expects `true_label` and `pred_label` to already be enoded as class ID's, starting from 0.
        """
        self.num_classes = num_classes
        pred_labels = pred_labels.squeeze().astype(int)
        true_labels = true_labels.squeeze().astype(int)
        try:
            assert np.max(true_labels) <= num_classes
            assert np.max(pred_labels) <= num_classes
        except AssertionError as e:
            print('True: ', np.max(true_labels), 'Pred: ', np.max(pred_labels))
            raise e

        self.matrix = np.zeros(shape=(num_classes, num_classes), dtype=int,) # Row = pred, Col = true
        for i, j in zip(pred_labels, true_labels):
            self.matrix[i, j] += 1
        
    
    def display(self, ax, row_labels=None, col_labels=None, color_scheme='Blues', side_length=0.075,):
        """
        Returns a matplotlib `Table` representing the confusion matrix. 
        Column labels are the integer encodings of the table. 
        
        If you want the labels to be something different, you will have to manually 
        edit them before calling `plt.show()`.
        """
        # Set Data
        table_data: list[list[str]] = self.matrix.tolist()
        if row_labels is None:
            row_labels = [str(i) for i in range(self.num_classes)]
        if col_labels is None:
            col_labels = [str(i) for i in range(self.num_classes)]
        
        norm = colors.Normalize(vmin=self.matrix.min(), vmax=self.matrix.max())
        cmap = cmaps[color_scheme]

        # Instantiate table
        ax.axis('off')
        table = ax.table(
            cellText=table_data, 
            rowLabels=row_labels,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            )

        # Style table
        for (row, col), cell in table.get_celld().items():
            # Make cell square
            cell.set_height(side_length)
            cell.set_width(side_length)

            if row == 0 or col == 0: # Header cells
                # cell.set_facecolor('#ccc')
                pass
            else:
                val = self.matrix[row-1, col-1] # Requires move to CPU
                cell.set_facecolor(cmap(norm(val)))
                
        return table
    
    def print_statistics(self):
        print(f'Accuracy: {self.accuracy():.4f}')
        for cls in range(self.num_classes):
            print(f'Precision ({cls}): {self.precision(cls):.4f}')
            print(f'Recall ({cls}): {self.precision(cls):.4f}')
            print(f'F1 Score ({cls}): {self.precision(cls):.4f}')

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