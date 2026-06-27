"""
Generate Confusion Matrix Heatmap for SE-CDT Drift Classification.

Creates a clear visualization showing classification accuracy per drift type.

Output: results/plots/confusion_matrix_se_cdt.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import BENCHMARK_PROPER_OUTPUTS, PLOTS_DIR


def load_benchmark_results():
    """Load benchmark results from pickle file."""
    results_path = BENCHMARK_PROPER_OUTPUTS["results_pkl"]
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    df = pd.read_pickle(results_path)
    return df


def compute_confusion_matrix(results_df):
    """
    Compute confusion matrix from SE-CDT classifications.
    
    Returns:
        confusion: Dict mapping (ground_truth, predicted) -> count
        labels: List of unique drift types
    """
    confusion = {}
    
    # All possible drift types
    labels = ["Sudden", "Gradual", "Incremental", "Recurrent", "Blip"]
    
    # Initialize confusion matrix
    for gt in labels:
        for pred in labels:
            confusion[(gt, pred)] = 0
    
    # Count classifications
    for _, row in results_df.iterrows():
        classifications = row.get("SE_Classifications", [])
        
        for item in classifications:
            gt = item.get("gt_type", "Unknown")
            pred = item.get("pred", "Unknown")
            
            if gt in labels and pred in labels:
                confusion[(gt, pred)] += 1
    
    return confusion, labels


def create_confusion_matrix_figure(confusion, labels, output_path):
    """
    Create and save confusion matrix heatmap.
    
    Args:
        confusion: Dict mapping (gt, pred) -> count
        labels: List of drift type labels
        output_path: Path to save figure
    """
    # Create matrix
    n = len(labels)
    matrix = np.zeros((n, n))
    
    for i, gt in enumerate(labels):
        for j, pred in enumerate(labels):
            matrix[i, j] = confusion.get((gt, pred), 0)
    
    # Compute percentages (row-wise normalization)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix_pct = matrix / row_sums * 100
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # === Left: Count Matrix ===
    ax1 = axes[0]
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax1,
        cbar_kws={'label': 'Count'}
    )
    ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    ax1.set_title('SE-CDT Classification: Counts', fontsize=14, fontweight='bold')
    
    # Rotate labels for readability
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # === Right: Percentage Matrix ===
    ax2 = axes[1]
    
    # Create annotation with both percentage and count
    annot = np.empty_like(matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            count = int(matrix[i, j])
            pct = matrix_pct[i, j]
            annot[i, j] = f'{pct:.1f}%\n({count})'
    
    sns.heatmap(
        matrix_pct,
        annot=annot,
        fmt='',
        cmap='RdYlGn',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax2,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Percentage (%)'}
    )
    ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    ax2.set_title('SE-CDT Classification: Row-Normalized (%)', fontsize=14, fontweight='bold')
    
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    # Compute overall metrics
    total = matrix.sum()
    correct = np.diag(matrix).sum()
    accuracy = correct / total * 100 if total > 0 else 0
    
    # Add summary text
    summary_text = (
        f"Overall Accuracy: {accuracy:.1f}%\n"
        f"Total Classifications: {int(total)}\n"
        f"Correct: {int(correct)}"
    )
    
    fig.text(
        0.5, 0.02,
        summary_text,
        ha='center',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {output_path}")
    print(f"  Overall Accuracy: {accuracy:.1f}%")
    
    return accuracy


def create_category_confusion_matrix(confusion, labels, output_path):
    """
    Create simplified TCD vs PCD confusion matrix.
    
    TCD (Transient Concept Drift): Sudden, Blip, Recurrent
    PCD (Persistent Concept Drift): Gradual, Incremental
    """
    TCD_TYPES = ["Sudden", "Blip", "Recurrent"]
    PCD_TYPES = ["Gradual", "Incremental"]
    
    # Aggregate into 2x2 matrix
    cat_matrix = np.zeros((2, 2))  # [TCD, PCD] x [TCD, PCD]
    
    for gt in labels:
        for pred in labels:
            count = confusion.get((gt, pred), 0)
            
            gt_is_tcd = gt in TCD_TYPES
            pred_is_tcd = pred in TCD_TYPES
            
            if gt_is_tcd and pred_is_tcd:
                cat_matrix[0, 0] += count  # TCD -> TCD
            elif gt_is_tcd and not pred_is_tcd:
                cat_matrix[0, 1] += count  # TCD -> PCD
            elif not gt_is_tcd and pred_is_tcd:
                cat_matrix[1, 0] += count  # PCD -> TCD
            else:
                cat_matrix[1, 1] += count  # PCD -> PCD
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cat_labels = ['TCD\n(Sudden/Blip/Recurrent)', 'PCD\n(Gradual/Incremental)']
    
    # Compute percentages
    row_sums = cat_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cat_pct = cat_matrix / row_sums * 100
    
    # Annotation
    annot = np.empty_like(cat_matrix, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f'{cat_pct[i, j]:.1f}%\n({int(cat_matrix[i, j])})'
    
    sns.heatmap(
        cat_pct,
        annot=annot,
        fmt='',
        cmap='RdYlGn',
        xticklabels=cat_labels,
        yticklabels=cat_labels,
        ax=ax,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Percentage (%)'}
    )
    
    ax.set_xlabel('Predicted Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Truth Category', fontsize=12, fontweight='bold')
    ax.set_title('SE-CDT Category Classification (TCD vs PCD)', fontsize=14, fontweight='bold')
    
    # Compute category accuracy
    total = cat_matrix.sum()
    correct = np.diag(cat_matrix).sum()
    cat_accuracy = correct / total * 100 if total > 0 else 0
    
    ax.text(
        0.5, -0.15,
        f'Category Accuracy: {cat_accuracy:.1f}%',
        transform=ax.transAxes,
        ha='center',
        fontsize=12,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Category confusion matrix saved to: {output_path}")
    print(f"  Category Accuracy: {cat_accuracy:.1f}%")
    
    return cat_accuracy


def main():
    """Generate all confusion matrix visualizations."""
    print("="*60)
    print("Generating SE-CDT Confusion Matrix Visualizations")
    print("="*60)
    
    # Load results
    try:
        df = load_benchmark_results()
        print(f"Loaded {len(df)} benchmark results")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python main.py compare' first to generate results.")
        return
    
    # Compute confusion matrix
    confusion, labels = compute_confusion_matrix(df)
    
    # Create output directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Generate visualizations
    output_path_main = PLOTS_DIR / "confusion_matrix_se_cdt.png"
    create_confusion_matrix_figure(confusion, labels, output_path_main)
    
    output_path_cat = PLOTS_DIR / "confusion_matrix_category.png"
    create_category_confusion_matrix(confusion, labels, output_path_cat)
    
    print("\n✓ All confusion matrix visualizations generated!")


if __name__ == "__main__":
    main()
