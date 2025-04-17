import numpy as np
from nottensorflow.neural_net import Model, Layer
from nottensorflow.metrics import ConfusionMatrix

    
def count_classes(labels):
    classes = set()
    for label in labels:
        classes.add(label)
    return len(classes)

def cross_validation(model_layers: list[Layer], x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, loss_fn, passes: int, use_SGD=True):
    x_split = np.hsplit(x, passes)
    y_split = np.hsplit(y, passes)
    models = []
   
    for i in range(passes):
        x_valid = x_split[i]
        x_train = np.concatenate(np.delete(x_split, i))
        
        y_valid = y_split[i]
        y_train = np.concatenate(np.delete(y_split, i)) 
        
        model = Model(SGD=use_SGD, layers=model_layers)
        model.train(x_train, y_train, epochs, learning_rate, loss_fn=loss_fn)
        
        y_valid_pred = model.predict(x_valid)
        y_train_pred = model.predict(x_train)
        
        confusionMatrix_valid = ConfusionMatrix(true_labels=y_valid, pred_labels=y_valid_pred, num_classes=count_classes(y))
        confusionMatrix_train = ConfusionMatrix(true_labels=y_train, pred_labels=y_train_pred, num_classes=count_classes(y))
        models.append((model, confusionMatrix_train, confusionMatrix_valid))
    return models