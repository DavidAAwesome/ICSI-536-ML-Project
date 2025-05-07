import time
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


from nottensorflow.image_processing import process_image
from nottensorflow.neural_net import Model
from nottensorflow.Layer import Dense
from nottensorflow.activation_fns import ReLU, Softmax, Sigmoid
from nottensorflow.loss_functions import MeanSquaredLoss, CrossEntropyLoss
from nottensorflow.Cross_validation import cross_validation
from nottensorflow.performance_metrics import ConfusionMatrix

NUM_CLASSES = 7

## Read in data from CSV (*.npz)
base_dir = os.path.dirname(__file__)
bone_yolo_dir = os.path.join(base_dir, 'data')

with np.load(os.path.join(bone_yolo_dir, 'v8_train_extracted.npz')) as f:
    train_data = np.concatenate([f['extracted'], f['extracted_augmented']], axis=0)
    np.random.shuffle(train_data)

# One-hot encode label vector
def onehot(y, num_classes):
    y = y.squeeze().astype(int)
    out = np.zeros((y.size, num_classes), dtype=int)
    out[np.arange(y.size), y] = 1
    return out

print("Train data shape:", train_data.shape)
X_train = train_data[:, :-1] # type: ignore
y_train = onehot(train_data[:, -1:], NUM_CLASSES)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

img_size = 224 * 224

## Hyperparameter configurations
configurations = [
    {
        'name': 'Optimized Deeper Network',
        'learning_rate': 0.1,
        'epochs': 100,
        'batch_size': 32,
        'layers': [
            (Dense(img_size, 128), ReLU()),
            (Dense(128, 64), ReLU()),
            (Dense(64, 32), ReLU()),
            (Dense(32, NUM_CLASSES), Softmax())
        ],
        'loss_fn': CrossEntropyLoss()
    },
    {
        'name': 'Balanced Network',
        'learning_rate': 0.05,
        'epochs': 100,
        'batch_size': 32,
        'layers': [
            (Dense(img_size, 64), ReLU()),
            (Dense(64, 32), ReLU()),
            (Dense(32, NUM_CLASSES), Softmax())
        ],
        'loss_fn': CrossEntropyLoss()
    },
    {
        'name': 'Wide Network',
        'learning_rate': 0.1,
        'epochs': 100,
        'batch_size': 32,
        'layers': [
            (Dense(img_size, 256), ReLU()),
            (Dense(256, NUM_CLASSES), Softmax())
        ],
        'loss_fn': CrossEntropyLoss()
    },
    {
        'name': 'Optimized Batch Size',
        'learning_rate': 0.1,
        'epochs': 100,
        'batch_size': 48,
        'layers': [
            (Dense(img_size, 64), ReLU()),
            (Dense(64, 32), ReLU()),
            (Dense(32, NUM_CLASSES), Softmax())
        ],
        'loss_fn': CrossEntropyLoss()
    },
    {
        'name': 'Hybrid Architecture',
        'learning_rate': 0.1,
        'epochs': 100,
        'batch_size': 32,
        'layers': [
            (Dense(img_size, 128), ReLU()),
            (Dense(128, 64), ReLU()),
            (Dense(64, NUM_CLASSES), Softmax())
        ],
        'loss_fn': CrossEntropyLoss()
    }
]

def plot_accuracy_per_configuration(trained_models, results):
    config_to_models = defaultdict(list)
    for model, result in zip(trained_models, results):
        config_to_models[result['config']].append((model, result['fold']))

    for config_name, models_folds in config_to_models.items():
        plt.figure(figsize=(10, 5))
        for model, fold in models_folds:
            plt.plot(model.accuracy_history, label=f"Fold {fold+1}")
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy')
        plt.title(f"{config_name} - Training Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"accuracy_{config_name.replace(' ', '_').lower()}.png")
        plt.show()


def plot_combined_accuracy(trained_models, results):
    config_to_best_model = {}
    config_to_best_acc = {}
    for model, result in zip(trained_models, results):
        config = result['config']
        acc = result['valid_acc']
        if config not in config_to_best_acc or acc > config_to_best_acc[config]:
            config_to_best_acc[config] = acc
            config_to_best_model[config] = model

    plt.figure(figsize=(12, 6))
    for config, model in config_to_best_model.items():
        plt.plot(model.accuracy_history, label=config)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Comparison of Training Accuracies Across Configurations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_comparison_all_configurations.png')
    plt.show()

results = []
trained_models = []

for config in configurations:
    print(f"\nTesting configuration: {config['name']}")
    start_time = time.time()
    
    my_model = Model().set_name(config['name'])
    for layer, activation in config['layers']:
        my_model.add(layer)
        my_model.add(activation)
    
    models = cross_validation(my_model, X_train, y_train, 
                            epochs=config['epochs'], 
                            learning_rate=config['learning_rate'], 
                            loss_fn=config['loss_fn'], 
                            passes=2)    # CHANGE FOLDS HERE
    
    for i, (model, train_cm, valid_cm) in enumerate(models):
        results.append({
            'config': config['name'],
            'fold': i,
            'train_acc': train_cm.accuracy(),
            'valid_acc': valid_cm.accuracy(),
            'time': time.time() - start_time,
        })
        trained_models.append(model)
        print(f"\nFold {i+1}:")
        print(f"Training accuracy: {train_cm.accuracy():.3f}")
        print(f"Validation accuracy: {valid_cm.accuracy():.3f}")
        print(f"Training time: {time.time() - start_time:.2f} seconds")

plot_accuracy_per_configuration(trained_models, results)
plot_combined_accuracy(trained_models, results)

print("\nSummary of Results:")
print("Configuration\tFold\tTrain Acc\tValid Acc\tTime (s)")
print("-" * 60)
for result in results:
    print(f"{result['config']}\t{result['fold']+1}\t{result['train_acc']:.3f}\t{result['valid_acc']:.3f}\t{result['time']:.2f}")

config_valid_accs = defaultdict(list)
for result in results:
    config_valid_accs[result['config']].append(result['valid_acc'])

print("\nAverage Validation Accuracy by Configuration:")
best_avg_acc = -1
for config, accs in config_valid_accs.items():
    avg_acc = sum(accs) / len(accs)
    print(f"{config}: {avg_acc:.4f}")
    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        best_config_name = config

print(f"\nBest configuration: {best_config_name} (Avg. Validation Accuracy: {best_avg_acc:.4f})")

# best_config: dict
best_config = max(results, key=lambda x: x['valid_acc'])
print(f"\nEvaluating best configuration ({best_config['config']}) on Test Set:")

for i, result in enumerate(results):
    if result['config'] == best_config_name:
        best_model = trained_models[i]

# Test set performance
with np.load(os.path.join(bone_yolo_dir, 'v8_test_extracted.npz')) as file:
    test_data = file['extracted']
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

predictions = np.argmax(best_model.predict(X_test), axis=1)
test_acc = ConfusionMatrix(true_labels=y_test, pred_labels=predictions, num_classes=NUM_CLASSES).accuracy()
print(f'**Test Accuracy: {test_acc:.2%}**')