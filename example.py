import time
import matplotlib.pyplot as plt
from nottensorflow.image_processing import process_image
from nottensorflow.neural_net import Model
from nottensorflow.Layer import Dense
from nottensorflow.activation_fns import ReLU, Softmax, Sigmoid
from nottensorflow.loss_functions import MeanSquaredLoss, CrossEntropyLoss
import os
import numpy as np
from nottensorflow.Cross_validation import cross_validation
from collections import defaultdict

NUM_CLASSES = 7

## Read in data from CSV
base_dir = os.path.dirname(__file__)
bone_yolo_dir = os.path.join(base_dir, 'data', 'BoneFractureYolo8')
train_path = os.path.join(bone_yolo_dir, 'train_extracted.csv')
train_data = np.loadtxt(train_path, dtype=float, delimiter=',')

print("Train data shape:", train_data.shape)
X_train = train_data[:, :-NUM_CLASSES]
y_train = train_data[:, -NUM_CLASSES:]
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
            'time': time.time() - start_time
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
best_config = None
best_avg_acc = -1
for config, accs in config_valid_accs.items():
    avg_acc = sum(accs) / len(accs)
    print(f"{config}: {avg_acc:.4f}")
    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        best_config = config
print(f"\nBest configuration: {best_config} (Avg. Validation Accuracy: {best_avg_acc:.4f})")

best_config = max(results, key=lambda x: x['valid_acc'])
print(f"\nTesting best configuration ({best_config['config']}) on a random image:")

best_model = Model()
for layer, activation in next(c['layers'] for c in configurations if c['name'] == best_config['config']):
    best_model.add(layer)
    best_model.add(activation)

# train best model on full dataset
best_model.train_SGD(X_train, y_train, 
                    epochs=best_config['epochs'], 
                    learning_rate=best_config['learning_rate'], 
                    loss_fn=best_config['loss_fn'], 
                    batch_size=best_config['batch_size'])

n = 43
test_images_path = os.path.join(bone_yolo_dir, 'test', 'images')
random_image_path = os.listdir(test_images_path)[n]
random_image_path = os.path.join(test_images_path, random_image_path)
print(f'Image: {random_image_path}')
random_image = process_image(random_image_path)

test_labels_path = os.path.join(bone_yolo_dir, 'test', 'labels')
random_label_path = os.listdir(test_labels_path)[n]
random_label_path = os.path.join(test_labels_path, random_label_path)
print(f'Label: {random_label_path}')

with open(random_label_path) as file:
    contents = file.read()
    true_label = int(contents[0]) if contents else -1  # File empty == No fracture
predict_label = np.argmax(best_model.predict(random_image))

print(f"Prediction: {predict_label}, Truth: {true_label}")
