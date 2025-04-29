import numpy as np
import matplotlib.pyplot as plt
from pixelhop import PixelHop, PixelHopPP, load_mnist, load_fashion_mnist
import argparse
import time
import os
import json
from datetime import datetime

def compare_models(dataset='mnist', num_samples=10000, th2=0.001):
    """Compare PixelHop and PixelHop++ models on the same dataset"""
    print(f"\n{'='*60}")
    print(f"Comparing PixelHop and PixelHop++ on {dataset.upper()} dataset")
    print(f"{'='*60}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"comparison_{dataset}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
    if dataset.lower() == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist(num_samples=num_samples, balanced=True)
    else:
        X_train, y_train, X_test, y_test = load_fashion_mnist(num_samples=num_samples, balanced=True)
    
    print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
    
    # Train PixelHop model
    print("\nTraining PixelHop model...")
    pixelhop_model = PixelHop(energy_threshold=th2)
    pixelhop_model.fit(X_train, y_train)
    pixelhop_test_acc = pixelhop_model.evaluate(X_test, y_test)
    
    # Train PixelHop++ model
    print("\nTraining PixelHop++ model...")
    pixelhop_pp_model = PixelHopPP(TH1=0.005, TH2=th2)
    pixelhop_pp_model.fit(X_train, y_train)
    pixelhop_pp_test_acc = pixelhop_pp_model.evaluate(X_test, y_test)
    
    # Compare results
    comparison = {
        "dataset": dataset,
        "num_samples": num_samples,
        "pixelhop": {
            "train_accuracy": pixelhop_model.train_accuracy,
            "test_accuracy": pixelhop_test_acc,
            "training_time": pixelhop_model.training_time,
            "model_size": pixelhop_model.get_model_size()
        },
        "pixelhop_pp": {
            "train_accuracy": pixelhop_pp_model.train_accuracy,
            "test_accuracy": pixelhop_pp_test_acc,
            "training_time": pixelhop_pp_model.training_time,
            "model_size": pixelhop_pp_model.get_model_size()
        }
    }
    
    # Save comparison results
    with open(os.path.join(results_dir, "comparison_results.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Create comparison charts
    create_comparison_charts(comparison, results_dir)
    
    return comparison

def create_comparison_charts(comparison, results_dir):
    """Create comparison visualizations"""
    # Extract data
    models = ["PixelHop", "PixelHop++"]
    train_acc = [comparison["pixelhop"]["train_accuracy"], comparison["pixelhop_pp"]["train_accuracy"]]
    test_acc = [comparison["pixelhop"]["test_accuracy"], comparison["pixelhop_pp"]["test_accuracy"]]
    training_time = [comparison["pixelhop"]["training_time"], comparison["pixelhop_pp"]["training_time"]]
    model_size = [comparison["pixelhop"]["model_size"], comparison["pixelhop_pp"]["model_size"]]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy comparison
    axs[0, 0].bar(models, train_acc, color=['blue', 'orange'])
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_title('Training Accuracy')
    axs[0, 0].set_ylim(0, 1)
    
    axs[0, 1].bar(models, test_acc, color=['blue', 'orange'])
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_title('Test Accuracy')
    axs[0, 1].set_ylim(0, 1)
    
    # Training time comparison
    axs[1, 0].bar(models, training_time, color=['blue', 'orange'])
    axs[1, 0].set_ylabel('Time (seconds)')
    axs[1, 0].set_title('Training Time')
    
    # Model size comparison
    axs[1, 1].bar(models, model_size, color=['blue', 'orange'])
    axs[1, 1].set_ylabel('Number of Parameters')
    axs[1, 1].set_title('Model Size')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison_charts.png"))
    plt.close()
    
    # Print summary
    print("\nComparison Summary:")
    print(f"{'Model':<12} {'Train Acc':<10} {'Test Acc':<10} {'Time (s)':<10} {'Parameters':<12}")
    print("-" * 60)
    print(f"PixelHop     {train_acc[0]:.4f}     {test_acc[0]:.4f}     {training_time[0]:.2f}      {model_size[0]}")
    print(f"PixelHop++   {train_acc[1]:.4f}     {test_acc[1]:.4f}     {training_time[1]:.2f}      {model_size[1]}")

def main():
    parser = argparse.ArgumentParser(description='Compare PixelHop and PixelHop++ models')
    parser.add_argument('--dataset', type=str, default='both', 
                      choices=['mnist', 'fashion_mnist', 'both'],
                      help='Dataset to use (default: both)')
    parser.add_argument('--samples', type=int, default=10000,
                      help='Number of training samples (default: 10000)')
    parser.add_argument('--th2', type=float, default=0.001,
                      help='Energy threshold for discarded nodes / PixelHop threshold (default: 0.001)')
    
    args = parser.parse_args()
    
    if args.dataset in ['mnist', 'both']:
        compare_models(dataset='mnist', num_samples=args.samples, th2=args.th2)
    
    if args.dataset in ['fashion_mnist', 'both']:
        compare_models(dataset='fashion_mnist', num_samples=args.samples, th2=args.th2)

if __name__ == '__main__':
    main() 