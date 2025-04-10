from nottensorflow.metrics import ConfusionMatrix
from nottensorflow.cross_validation import count_classes
import random
import matplotlib.pyplot as plt

random.seed(42)

def test_count_classes():
    classes = ['bird', 'cat', 'dog', 'monkey', 'a', 'b', 'c']
    labels = [classes[random.randint(0, len(classes)-1)] for _ in range(100)]
    assert count_classes(labels) == len(classes)

def test_display_matrix():
    num_classes = 7
    true = [random.randint(0, num_classes-1) for _ in range(100)]
    pred = [random.randint(0, num_classes-1) for _ in range(100)]
    confuddled = ConfusionMatrix(true, pred, num_classes)
    ax = confuddled.display()
    plt.show()

    for i in range(num_classes):
        for j in range(num_classes):
            count = int(ax[i,j].get_text().get_text())
            assert count == confuddled.matrix[i,j]
