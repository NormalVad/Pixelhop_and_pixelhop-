import numpy as np
import matplotlib.pyplot as plt
from pixelhop import PixelHopPP, load_mnist, load_fashion_mnist
import argparse
import time
import os
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix

def visualize_filters(model, layer_idx=0, num_filters=16, save_path=None):
    if layer_idx >= len(model.units):
        print(f"Layer index {layer_idx} out of range")
        return
    unit = model.units[layer_idx]
    if not hasattr(unit.saab_transformer, 'kernels') or unit.saab_transformer.kernels is None:
        print("No kernels found in this unit")
        return     
    kernels = unit.saab_transformer.kernels
    if layer_idx == 0:
        window_size = unit.window_size
        if isinstance(window_size, tuple):
            h, w = window_size
        else:
            h = w = window_size
            
        # Get first channel for visualization
        n_display = min(num_filters, kernels.shape[1])
        filters = kernels.reshape(h, w, -1, kernels.shape[1])[:, :, 0, :n_display]
    else:
        # For other layers, shown the first component reshaped into a grid
        n_display = min(num_filters, kernels.shape[1])
        size = int(np.sqrt(kernels.shape[0]))
        filters = kernels[:size*size, :n_display].reshape(size, size, n_display)
    
    grid_size = int(np.ceil(np.sqrt(n_display)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            ax = axes[i, j]
            if idx < n_display:
                ax.imshow(filters[:, :, idx], cmap='viridis')
                ax.set_title(f"Filter {idx+1}")
            ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_experiment_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def run_mnist_experiment(num_samples=10000, th1=0.005, th2=0.001, exp_dir="results"):
    """PixelHop++ experiment on MNIST dataset"""
    print(f"\n{'='*60}")
    print(f"Running MNIST experiment with TH1={th1}, TH2={th2}")
    print(f"{'='*60}")
    mnist_dir = os.path.join(exp_dir, f"mnist_th1_{th1}_th2_{th2}")
    os.makedirs(mnist_dir, exist_ok=True)
    X_train, y_train, X_test, y_test = load_mnist(
        num_samples=num_samples, 
        balanced=True
    )
    print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
    model = PixelHopPP(TH1=th1, TH2=th2)
    model.fit(X_train, y_train)
    test_accuracy = model.evaluate(X_test, y_test)
    summary = model.get_model_summary()
    summary["dataset"] = "mnist"
    summary["samples"] = num_samples
    summary["th1"] = th1
    summary["th2"] = th2
    summary["test_accuracy"] = test_accuracy
    with open(os.path.join(mnist_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title(f"MNIST Confusion Matrix (TH1={th1})")
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), 
                   ha="center", va="center", 
                   color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.savefig(os.path.join(mnist_dir, "confusion_matrix.png"))
    plt.close()
    
    return model, test_accuracy, mnist_dir

def run_fashion_mnist_experiment(num_samples=10000, th1=0.005, th2=0.001, exp_dir="results"):
    """PixelHop++ experiment on Fashion-MNIST dataset"""
    print(f"\n{'='*60}")
    print(f"Running Fashion-MNIST experiment with TH1={th1}, TH2={th2}")
    print(f"{'='*60}")
    fashion_dir = os.path.join(exp_dir, f"fashion_mnist_th1_{th1}_th2_{th2}")
    os.makedirs(fashion_dir, exist_ok=True)
    X_train, y_train, X_test, y_test = load_fashion_mnist(
        num_samples=num_samples, 
        balanced=True
    )
    print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
    model = PixelHopPP(TH1=th1, TH2=th2)
    model.fit(X_train, y_train)
    test_accuracy = model.evaluate(X_test, y_test)
    summary = model.get_model_summary()
    summary["dataset"] = "fashion_mnist"
    summary["samples"] = num_samples
    summary["th1"] = th1
    summary["th2"] = th2
    summary["test_accuracy"] = test_accuracy
    with open(os.path.join(fashion_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap='Blues')
    plt.title(f"Fashion-MNIST Confusion Matrix (TH1={th1})")
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(np.arange(10), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(10), class_names)
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), 
                   ha="center", va="center", 
                   color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(fashion_dir, "confusion_matrix.png"))
    plt.close()
    
    return model, test_accuracy, fashion_dir

def explore_th1_values(dataset='mnist', num_samples=10000, th2=0.001, 
                     th1_values=None, exp_dir="results"):
    print(f"\n{'='*60}")
    print(f"Exploring TH1 values for {dataset.upper()} dataset")
    print(f"{'='*60}")
    explore_dir = os.path.join(exp_dir, f"{dataset}_th1_exploration")
    os.makedirs(explore_dir, exist_ok=True)
    if th1_values is None:
        th1_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    results = []
    for th1 in th1_values:
        if dataset.lower() == 'mnist':
            model, accuracy, result_dir = run_mnist_experiment(
                num_samples=num_samples, 
                th1=th1, 
                th2=th2,
                exp_dir=explore_dir
            )
        else:
            model, accuracy, result_dir = run_fashion_mnist_experiment(
                num_samples=num_samples, 
                th1=th1, 
                th2=th2,
                exp_dir=explore_dir
            )
        results.append({
            "th1": th1,
            "accuracy": accuracy,
            "model_size": model.get_model_size(),
            "training_time": model.training_time,
            "intermediate_nodes": [unit.get_intermediate_nodes() for unit in model.units],
            "discarded_nodes": [unit.get_discarded_nodes() for unit in model.units]
        })
        print(f"TH1={th1}: Test accuracy={accuracy:.4f}, Model size={model.get_model_size()}")
    with open(os.path.join(explore_dir, "th1_exploration_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(th1_values, [r["accuracy"] for r in results], 'o-')
    plt.xlabel('TH1 Value')
    plt.ylabel('Test Accuracy')
    plt.title(f'{dataset.upper()}: TH1 vs Test Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(th1_values, [r["model_size"] for r in results], 'o-')
    plt.xlabel('TH1 Value')
    plt.ylabel('Model Size (# parameters)')
    plt.title(f'{dataset.upper()}: TH1 vs Model Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(explore_dir, "th1_exploration_plots.png"))
    plt.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='PixelHop++ experiments on MNIST and Fashion-MNIST')
    parser.add_argument('--dataset', type=str, default='both', 
                     choices=['mnist', 'fashion_mnist', 'both'],
                     help='Dataset to use (default: both)')
    parser.add_argument('--samples', type=int, default=10000,
                     help='Number of training samples (default: 10000)')
    parser.add_argument('--th1', type=float, default=0.005,
                     help='Energy threshold for intermediate nodes (default: 0.005)')
    parser.add_argument('--th2', type=float, default=0.001,
                     help='Energy threshold for discarded nodes (default: 0.001)')
    parser.add_argument('--explore_th1', action='store_true',
                     help='Explore different TH1 values')
    
    args = parser.parse_args()
    exp_dir = create_experiment_dir()
    print(f"Results will be saved to: {exp_dir}")
    
    if args.explore_th1:
        if args.dataset in ['mnist', 'both']:
            explore_th1_values(
                dataset='mnist', 
                num_samples=args.samples, 
                th2=args.th2,
                exp_dir=exp_dir
            )
        if args.dataset in ['fashion_mnist', 'both']:
            explore_th1_values(
                dataset='fashion_mnist', 
                num_samples=args.samples, 
                th2=args.th2,
                exp_dir=exp_dir
            )
    else:
        if args.dataset in ['mnist', 'both']:
            run_mnist_experiment(
                num_samples=args.samples, 
                th1=args.th1, 
                th2=args.th2,
                exp_dir=exp_dir
            )
        if args.dataset in ['fashion_mnist', 'both']:
            run_fashion_mnist_experiment(
                num_samples=args.samples, 
                th1=args.th1, 
                th2=args.th2,
                exp_dir=exp_dir
            )

if __name__ == '__main__':
    main() 