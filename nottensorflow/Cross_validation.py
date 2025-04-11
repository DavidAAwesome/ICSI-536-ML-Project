import numpy as np

import neural_network
import preformance_metrics

def count_classes(labels):
    classes = set()
    for label in labels:
        classes.add(label)
    return len(classes)


def cross_validation(model_layers, x, y, epochs, learning_rate, loss_fn, passes):
    x_split = np.hsplit(x, passes)
    y_split = np.hsplit(y, passes)
    models = []

    for i in range(passes):
        x_test = x_split[i]
        x_train = np.concatenate(np.delete(x_split, i))

        y_test = y_split[i]
        y_train = np.concatenate(np.delete(y_split, i))

        model = neural_network.Model().add(model_layers)
        model.train_SGD(x_train, y_train, epochs, learning_rate, loss_fn=loss_fn,batch_size=32)

        y_valid_pred = model.predict(x_test)
        y_train_pred = model.predict(x_train)

        confusionMatrix_valid = preformance_metrics.ConfusionMatrix(true_labels=y_test, pred_labels=y_valid_pred,
                                                num_classes=count_classes(y))
        confusionMatrix_train = preformance_metrics.ConfusionMatrix(true_labels=y_train, pred_labels=y_train_pred,
                                                num_classes=count_classes(y))
        models.add([model, confusionMatrix_train, confusionMatrix_valid])
        print('Confusion Matrix Test acc', confusionMatrix_valid.accuracy())
    return models