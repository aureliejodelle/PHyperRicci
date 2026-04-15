#!/usr/bin/env python3
"""
Single Protein Curvature Visualization - 6UEV_A
----------------------------------------------
Creates publication-quality visualizations of per-residue curvature data.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Set style for publication-quality figures
plt.style.use('default')
sns.set_style("white")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.linewidth': 1.2,
    'axes.edgecolor': 'black',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Colors for curvature
NEGATIVE_COLOR = '#E74C3C'  # Red for negative curvature
POSITIVE_COLOR = '#2ECC71'  # Green for positive curvature
NEUTRAL_COLOR = '#7F8C8D'   # Gray for neutral
LINE_COLOR = '#2C3E50'      # Dark blue for line

def load_curvature(json_path):
    """Load curvature data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    residues = []
    curvatures = []
    for idx, curv in data.items():
        residues.append(int(idx))
        curvatures.append(float(curv))
    
    return residues, curvatures

def create_main_curvature_plot(residues, curvatures, output_path):
    """
    Main curvature visualization - Line plot with colored fill.
    This is the primary figure for the pipeline.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Convert to numpy arrays
    residues = np.array(residues)
    curvatures = np.array(curvatures)
    
    # Plot the curvature line
    ax.plot(residues, curvatures, color=LINE_COLOR, linewidth=1.5, alpha=0.8, label='Curvature')
    
    # Fill negative curvature areas (below zero)
    ax.fill_between(residues, 0, curvatures, 
                    where=(curvatures < 0), 
                    color=NEGATIVE_COLOR, alpha=0.3, 
                    label='Negative curvature (concave)')
    
    # Fill positive curvature areas (above zero)
    ax.fill_between(residues, 0, curvatures, 
                    where=(curvatures > 0), 
                    color=POSITIVE_COLOR, alpha=0.3, 
                    label='Positive curvature (convex)')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Add rolling average for smoothing
    window = 10
    rolling_avg = pd.Series(curvatures).rolling(window=window, center=True).mean()
    ax.plot(residues, rolling_avg, color='#3498DB', linewidth=2, 
            alpha=0.8, label=f'Rolling average (window={window})')
    
    # Customize the plot
    ax.set_xlabel('Residue Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Curvature Value', fontsize=14, fontweight='bold')
    ax.set_title('7NH8_A - Per-Residue Curvature Profile', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
              edgecolor='black', fontsize=10)
    
    # Add statistics box
    stats_text = f"""Statistics:
    Mean: {np.mean(curvatures):.2f}
    Std: {np.std(curvatures):.2f}
    Min: {np.min(curvatures):.1f}
    Max: {np.max(curvatures):.1f}
    Negative: {(curvatures < 0).sum()}/{len(curvatures)} ({((curvatures < 0).mean()*100):.1f}%)
    Positive: {(curvatures > 0).sum()}/{len(curvatures)} ({((curvatures > 0).mean()*100):.1f}%)"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Main plot saved: {output_path}")

def create_heatmap_visualization(residues, curvatures, output_path):
    """
    Heatmap visualization - Shows curvature intensity along the sequence.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Create heatmap data (reshape into rows for better visualization)
    n_cols = 50  # Number of residues per row in heatmap
    n_rows = int(np.ceil(len(curvatures) / n_cols))
    
    # Pad the array to make it rectangular
    padded_curvatures = np.pad(curvatures, (0, n_rows * n_cols - len(curvatures)), 
                                constant_values=np.nan)
    heatmap_data = padded_curvatures.reshape(n_rows, n_cols)
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', 
                   interpolation='nearest', vmin=-50, vmax=50)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                        pad=0.05, aspect=40, shrink=0.8)
    cbar.set_label('Curvature Value', fontsize=11)
    
    # Customize
    ax.set_xlabel('Residue Position (within segment)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Segment Number', fontsize=12, fontweight='bold')
    ax.set_title('7NH8_A - Curvature Heatmap', fontsize=14, fontweight='bold')
    
    # Add residue number annotations on x-axis
    x_ticks = np.arange(0, n_cols, 10)
    x_labels = [str(i+1) for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Heatmap saved: {output_path}")

def create_distribution_plot(residues, curvatures, output_path):
    """
    Distribution visualization - Histogram and boxplot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram with KDE
    ax1 = axes[0]
    n, bins, patches = ax1.hist(curvatures, bins=30, color='#3498DB', 
                                 edgecolor='black', alpha=0.7, density=False)
    
    # Color bars by sign
    for patch, bin_edge in zip(patches, bins[:-1]):
        if bin_edge < 0:
            patch.set_facecolor(NEGATIVE_COLOR)
        else:
            patch.set_facecolor(POSITIVE_COLOR)
        patch.set_alpha(0.6)
    
    # Add KDE line
    from scipy import stats
    kde = stats.gaussian_kde(curvatures)
    x_grid = np.linspace(min(curvatures), max(curvatures), 200)
    ax1.plot(x_grid, kde(x_grid) * len(curvatures) * (bins[1]-bins[0]), 
             color='black', linewidth=2, label='KDE')
    
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero')
    ax1.axvline(x=np.mean(curvatures), color='green', linestyle='-', 
                linewidth=1.5, label=f'Mean: {np.mean(curvatures):.2f}')
    ax1.set_xlabel('Curvature Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Curvature Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    
    # Boxplot
    ax2 = axes[1]
    bp = ax2.boxplot(curvatures, vert=True, patch_artist=True,
                      boxprops=dict(facecolor='#3498DB', alpha=0.7, linewidth=1.5),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(color='black', linewidth=1),
                      capprops=dict(color='black', linewidth=1),
                      flierprops=dict(marker='o', markerfacecolor='gray', 
                                     markersize=4, alpha=0.5))
    ax2.set_ylabel('Curvature Value', fontsize=12)
    ax2.set_title('Curvature Summary', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(['7NH8_A'])
    ax2.grid(True, alpha=0.2, axis='y')
    
    # Add statistical annotations
    stats = {
        'Q1': np.percentile(curvatures, 25),
        'Median': np.median(curvatures),
        'Q3': np.percentile(curvatures, 75),
        'IQR': np.percentile(curvatures, 75) - np.percentile(curvatures, 25)
    }
    
    stats_text = f"Q1: {stats['Q1']:.2f}\nMedian: {stats['Median']:.2f}\nQ3: {stats['Q3']:.2f}\nIQR: {stats['IQR']:.2f}"
    ax2.text(1.05, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.suptitle('7NH8_A - Curvature Distribution Analysis', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Distribution plot saved: {output_path}")

def create_peak_analysis_plot(residues, curvatures, output_path, top_n=10):
    """
    Peak analysis - Highlights extreme curvature residues.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    residues = np.array(residues)
    curvatures = np.array(curvatures)
    
    # Plot all curvature
    ax.plot(residues, curvatures, color='gray', linewidth=1, alpha=0.5)
    ax.fill_between(residues, 0, curvatures, 
                    where=(curvatures < 0), 
                    color=NEGATIVE_COLOR, alpha=0.2)
    ax.fill_between(residues, 0, curvatures, 
                    where=(curvatures > 0), 
                    color=POSITIVE_COLOR, alpha=0.2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Find top N positive and negative peaks
    # Use scipy for peak detection
    from scipy.signal import find_peaks
    
    # Find positive peaks
    positive_peaks, _ = find_peaks(curvatures, height=np.percentile(curvatures[curvatures > 0], 75) if any(curvatures > 0) else None)
    # Find negative peaks (invert for detection)
    negative_peaks, _ = find_peaks(-curvatures, height=-np.percentile(curvatures[curvatures < 0], 25) if any(curvatures < 0) else None)
    
    # Get top N by magnitude
    positive_vals = [(r, c) for r, c in zip(residues[positive_peaks], curvatures[positive_peaks])]
    negative_vals = [(r, c) for r, c in zip(residues[negative_peaks], curvatures[negative_peaks])]
    
    positive_vals.sort(key=lambda x: x[1], reverse=True)
    negative_vals.sort(key=lambda x: x[1])
    
    # Highlight top positive peaks
    for i, (r, c) in enumerate(positive_vals[:top_n]):
        ax.scatter(r, c, color=POSITIVE_COLOR, s=100, zorder=5, 
                   edgecolor='black', linewidth=1.5)
        ax.annotate(f"{r}", (r, c), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold', color=POSITIVE_COLOR,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Highlight top negative peaks
    for i, (r, c) in enumerate(negative_vals[:top_n]):
        ax.scatter(r, c, color=NEGATIVE_COLOR, s=100, zorder=5,
                   edgecolor='black', linewidth=1.5)
        ax.annotate(f"{r}", (r, c), xytext=(5, -15), textcoords='offset points',
                   fontsize=9, fontweight='bold', color=NEGATIVE_COLOR,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Residue Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Curvature Value', fontsize=14, fontweight='bold')
    ax.set_title(f'7NH8_A - Curvature Peaks (Top {top_n} Positive/Negative)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=POSITIVE_COLOR, alpha=0.3, label='Positive curvature'),
        Patch(facecolor=NEGATIVE_COLOR, alpha=0.3, label='Negative curvature'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=POSITIVE_COLOR,
                   markersize=10, markeredgecolor='black', label=f'Top {top_n} positive peaks'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=NEGATIVE_COLOR,
                   markersize=10, markeredgecolor='black', label=f'Top {top_n} negative peaks')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Peak analysis saved: {output_path}")

def create_simple_pipeline_figure(residues, curvatures, output_path):
    """
    Simple, clean figure suitable for direct insertion into pipeline.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    residues = np.array(residues)
    curvatures = np.array(curvatures)
    
    # Main line
    ax.plot(residues, curvatures, color='#2C3E50', linewidth=1.8, alpha=0.9)
    
    # Colored fill
    ax.fill_between(residues, 0, curvatures, 
                    where=(curvatures < 0), 
                    color='#E74C3C', alpha=0.4, label='Negative (concave)')
    ax.fill_between(residues, 0, curvatures, 
                    where=(curvatures > 0), 
                    color='#2ECC71', alpha=0.4, label='Positive (convex)')
    
    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.6)
    
    # Styling
    ax.set_xlabel('Residue Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Curvature', fontsize=13, fontweight='bold')
    ax.set_title('7NH8_A: Protein Curvature Profile', fontsize=15, fontweight='bold', pad=15)
    
    # Subtle grid
    ax.grid(True, alpha=0.15, linestyle='--')
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
              edgecolor='black', fontsize=10)
    
    # Add minimal statistics
    stats = f"Mean: {np.mean(curvatures):.2f} | Std: {np.std(curvatures):.2f}"
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Pipeline figure saved: {output_path}")

def export_summary_data(residues, curvatures, output_dir):
    """Export summary data as CSV and JSON."""
    
    # Create DataFrame
    df = pd.DataFrame({
        'residue_index': residues,
        'curvature': curvatures
    })
    
    # Save CSV
    csv_path = output_dir / '7NH8_A_curvature_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV saved: {csv_path}")
    
    # Save summary statistics JSON
    summary = {
        'protein': '7NH8_A',
        'n_residues': len(residues),
        'mean_curvature': float(np.mean(curvatures)),
        'std_curvature': float(np.std(curvatures)),
        'min_curvature': float(np.min(curvatures)),
        'max_curvature': float(np.max(curvatures)),
        'median_curvature': float(np.median(curvatures)),
        'negative_percentage': float((np.array(curvatures) < 0).mean() * 100),
        'positive_percentage': float((np.array(curvatures) > 0).mean() * 100),
        'most_negative_residues': [
            {'residue': int(r), 'curvature': float(c)} 
            for r, c in zip(residues, curvatures) 
            if c == np.min(curvatures)
        ][:5],
        'most_positive_residues': [
            {'residue': int(r), 'curvature': float(c)} 
            for r, c in zip(residues, curvatures) 
            if c == np.max(curvatures)
        ][:5]
    }
    
    json_path = output_dir / '7NH8_A_curvature_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary JSON saved: {json_path}")

def main():
    # Setup paths
    json_path = Path("7NH8_A.json")
    output_dir = Path("curvature_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("7NH8_A - CURVATURE VISUALIZATION")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {json_path}")
    residues, curvatures = load_curvature(json_path)
    print(f"  ✓ Loaded {len(residues)} residues")
    print(f"  ✓ Curvature range: [{min(curvatures):.1f}, {max(curvatures):.1f}]")
    
    # Export data
    print("\nExporting data...")
    export_summary_data(residues, curvatures, output_dir)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Main plot (recommended for pipeline)
    create_main_curvature_plot(residues, curvatures, 
                                output_dir / "7NH8_A_curvature_main.pdf")
    
    # 2. Heatmap
    create_heatmap_visualization(residues, curvatures,
                                  output_dir / "7NH8_A_curvature_heatmap.pdf")
    
    # 3. Distribution plot
    create_distribution_plot(residues, curvatures,
                             output_dir / "7NH8_A_curvature_distribution.pdf")
    
    # 4. Peak analysis
    create_peak_analysis_plot(residues, curvatures,
                              output_dir / "7NH8_A_curvature_peaks.pdf", top_n=8)
    
    # 5. Simple pipeline figure (clean, minimal)
    create_simple_pipeline_figure(residues, curvatures,
                                   output_dir / "7NH8_A_pipeline_ready.pdf")
    
    print("\n" + "=" * 60)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("=" * 60)
    
    # Print summary
    print("\nQUICK SUMMARY:")
    print(f"  Total residues: {len(residues)}")
    print(f"  Mean curvature: {np.mean(curvatures):.3f}")
    print(f"  Standard deviation: {np.std(curvatures):.3f}")
    print(f"  Most negative residue: {residues[np.argmin(curvatures)]} ({min(curvatures):.1f})")
    print(f"  Most positive residue: {residues[np.argmax(curvatures)]} ({max(curvatures):.1f})")
    print(f"  % Negative curvature: {(np.array(curvatures) < 0).mean() * 100:.1f}%")
    print(f"  % Positive curvature: {(np.array(curvatures) > 0).mean() * 100:.1f}%")

if __name__ == "__main__":
    main()
