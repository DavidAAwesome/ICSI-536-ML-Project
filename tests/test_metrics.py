from nottensorflow.metrics import ConfusionMatrix
from nottensorflow.cross_validation import count_classes
import random
import matplotlib.pyplot as plt

random.seed(42)

classes = ['bird', 'cat', 'dog', 'monkey', 'a', 'b', 'c']
labels = [classes[random.randint(0, len(classes)-1)] for _ in range(100)]
num_classes = len(classes)
true = [random.randint(0, num_classes-1) for _ in range(100)]
pred = [random.randint(0, num_classes-1) for _ in range(100)]

def test_count_classes():
    assert count_classes(labels) == len(classes)

def test_display_matrix_no_colLabel_and_no_rowLabel():
    cmat = ConfusionMatrix(true, pred, num_classes)
    # Test all overloads of method call
    axes = [
        cmat.display(),
        cmat.display(row_labels=classes),
        cmat.display(col_labels=classes.reverse()),
        cmat.display(classes, classes.reverse())
    ]
    plt.show()
    
    ax = axes[0]
    for i in range(num_classes):
        for j in range(num_classes):
            count = int(ax[i,j].get_text().get_text())
            assert count == cmat.matrix[i,j]