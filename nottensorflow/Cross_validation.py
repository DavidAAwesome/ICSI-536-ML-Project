import numpy as np
from nottensorflow import neural_net
from nottensorflow import preformance_metrics

def count_classes(labels):
    return labels.shape[1]

def cross_validation(model, x, y, epochs, learning_rate, loss_fn, passes):
    fold_size = len(x) // passes
    models = []

    for i in range(passes):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < passes - 1 else len(x)
        
        x_test = x[start_idx:end_idx]
        y_test = y[start_idx:end_idx]
        
        x_train = np.concatenate([x[:start_idx], x[end_idx:]])
        y_train = np.concatenate([y[:start_idx], y[end_idx:]])

        fold_model = neural_net.Model()
        for layer in model.layers:
            fold_model.add(layer)

        fold_model.train_SGD(x_train, y_train, epochs, learning_rate, loss_fn=loss_fn, batch_size=32)

        y_valid_pred = fold_model.predict(x_test)
        y_train_pred = fold_model.predict(x_train)

        y_test_indices = np.argmax(y_test, axis=1)
        y_valid_pred_indices = np.argmax(y_valid_pred, axis=1)
        y_train_indices = np.argmax(y_train, axis=1)
        y_train_pred_indices = np.argmax(y_train_pred, axis=1)

        confusionMatrix_valid = preformance_metrics.ConfusionMatrix(true_labels=y_test_indices, 
                                                                  pred_labels=y_valid_pred_indices,
                                                                  num_classes=count_classes(y))
        confusionMatrix_train = preformance_metrics.ConfusionMatrix(true_labels=y_train_indices, 
                                                                  pred_labels=y_train_pred_indices,
                                                                  num_classes=count_classes(y))
        models.append([fold_model, confusionMatrix_train, confusionMatrix_valid])
        print('Confusion Matrix Test acc', confusionMatrix_valid.accuracy())
    return models
