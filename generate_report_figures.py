#!/usr/bin/env python3
"""
Generate publication-quality figures for NuScenes-QA BTP report.

Produces:
  1. Training loss curves (all key models)
  2. Overall accuracy progression (epoch-by-epoch)
  3. Category-wise accuracy comparison (bar chart)
  4. Category-wise accuracy radar/spider chart
  5. Model comparison table (best results)
  6. Architecture evolution timeline
  7. Question type distribution (pie chart)

Usage:
    python generate_report_figures.py
"""

import re
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

# ========== OUTPUT DIR ==========
OUT_DIR = './report_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ========== STYLE ==========
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ========== PARSE LOG FILES ==========
def parse_log(filepath):
    """Parse a training log file. Returns dict with epochs, losses, and eval results."""
    data = {
        'epochs': [],
        'losses': [],
        'lrs': [],
        'evals': [],  # list of dicts {epoch, overall, comparison, count, exist, object, status}
    }
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    current_epoch = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Parse epoch + loss
        m = re.match(r'Epoch:\s*(\d+),\s*Loss:\s*([\d.]+),\s*Lr:\s*([\d.e+-]+)', line)
        if m:
            current_epoch = int(m.group(1))
            loss = float(m.group(2))
            lr = float(m.group(3))
            data['epochs'].append(current_epoch)
            data['losses'].append(loss)
            data['lrs'].append(lr)
            continue
        
        # Parse evaluation results
        m = re.match(r'Overall\s*:?\s*(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)', line)
        if m:
            eval_data = {'epoch': current_epoch, 'overall': float(m.group(3))}
            # Read the next lines for category results
            for j in range(i+1, min(i+20, len(lines))):
                cat_line = lines[j].strip()
                for cat in ['comparison', 'count', 'exist', 'object', 'status']:
                    cm = re.match(rf'^{cat}\s*:?\s*(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)', cat_line)
                    if cm:
                        eval_data[cat] = float(cm.group(3))
            data['evals'].append(eval_data)
    
    return data


# ========== LOAD ALL KEY LOGS ==========
LOG_DIR = './log'

# Define models to plot (name -> log file)
MODELS = {
    'BEV (Baseline)': 'log_run_111272.txt',
    'YOLO': 'log_run_yolo_v1.txt',
    'BEV-YOLO Fusion': 'log_run_fusion_v1.txt',
    'RadarXFormer': 'log_run_radarxf_v1.txt',
    'RadarXF-BEV Fusion': 'log_run_radarfx_fusion_final.txt',
    'Trimodal V1': 'log_run_trimodal_v1.txt',
    'Trimodal V3 (QType)': 'log_run_trimodal_v3_qtype.txt',
    'Trimodal V4 (Best)': 'log_run_trimodal_v4_finetune.txt',
}

model_data = {}
for name, logfile in MODELS.items():
    path = os.path.join(LOG_DIR, logfile)
    if os.path.exists(path):
        model_data[name] = parse_log(path)
        print(f"  Loaded {name}: {len(model_data[name]['epochs'])} epochs, "
              f"{len(model_data[name]['evals'])} evals")
    else:
        print(f"  [SKIP] {logfile} not found")


# ========== BEST RESULTS TABLE ==========
# Hardcoded best results from all experiments
BEST_RESULTS = {
    'BEV (Baseline)':     {'overall': 55.80, 'comparison': 68.50, 'count': 19.90, 'exist': 81.50, 'object': 45.50, 'status': 53.40, 'params': '35.8M'},
    'YOLO':               {'overall': 55.53, 'comparison': 68.11, 'count': 19.25, 'exist': 81.70, 'object': 45.34, 'status': 52.98, 'params': '35.8M'},
    'BEV-YOLO Fusion':    {'overall': 55.37, 'comparison': 67.89, 'count': 19.34, 'exist': 81.22, 'object': 45.57, 'status': 52.01, 'params': '36.2M'},
    'RadarXFormer':       {'overall': 53.90, 'comparison': 66.80, 'count': 18.50, 'exist': 80.10, 'object': 43.70, 'status': 50.90, 'params': '18.0M'},
    'RadarXF-BEV Fusion': {'overall': 55.62, 'comparison': 67.90, 'count': 20.00, 'exist': 81.80, 'object': 45.30, 'status': 52.80, 'params': '18.7M'},
    'Trimodal V1':        {'overall': 56.10, 'comparison': 67.71, 'count': 20.64, 'exist': 82.02, 'object': 46.89, 'status': 52.55, 'params': '18.7M'},
    'Trimodal V3 (QType)':{'overall': 56.06, 'comparison': 68.00, 'count': 21.33, 'exist': 82.26, 'object': 45.30, 'status': 53.46, 'params': '18.7M'},
    'Trimodal V4 (Best)': {'overall': 56.52, 'comparison': 68.30, 'count': 21.33, 'exist': 82.57, 'object': 46.30, 'status': 54.84, 'params': '18.7M'},
}


# ========== FIGURE 1: Training Loss Curves ==========
def plot_loss_curves():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#E91E63']
    
    for idx, (name, data) in enumerate(model_data.items()):
        if data['losses']:
            ax.plot(data['epochs'], data['losses'],
                    color=colors[idx % len(colors)],
                    linewidth=2, alpha=0.85,
                    label=name, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Convergence Across Models')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig1_loss_curves.png'))
    plt.close()
    print("  ✓ Figure 1: Loss curves")


# ========== FIGURE 2: Overall Accuracy Progression ==========
def plot_accuracy_progression():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#E91E63']
    
    for idx, (name, data) in enumerate(model_data.items()):
        if data['evals']:
            epochs = [e['epoch'] for e in data['evals']]
            accs = [e['overall'] for e in data['evals']]
            ax.plot(epochs, accs,
                    color=colors[idx % len(colors)],
                    linewidth=2.5, alpha=0.85,
                    label=name, marker='s', markersize=5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Overall Accuracy (%)')
    ax.set_title('Validation Accuracy Progression')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(50, 60)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig2_accuracy_progression.png'))
    plt.close()
    print("  ✓ Figure 2: Accuracy progression")


# ========== FIGURE 3: Category-wise Bar Chart ==========
def plot_category_comparison():
    categories = ['overall', 'comparison', 'count', 'exist', 'object', 'status']
    cat_labels = ['Overall', 'Comparison', 'Count', 'Exist', 'Object', 'Status']
    
    # Select key models for comparison
    key_models = ['BEV (Baseline)', 'YOLO', 'RadarXFormer', 'Trimodal V1', 'Trimodal V4 (Best)']
    key_models = [m for m in key_models if m in BEST_RESULTS]
    
    x = np.arange(len(categories))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#2196F3', '#FF9800', '#9C27B0', '#00BCD4', '#E91E63']
    
    for idx, model in enumerate(key_models):
        vals = [BEST_RESULTS[model].get(cat, 0) for cat in categories]
        offset = (idx - len(key_models)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=colors[idx % len(colors)], alpha=0.85, edgecolor='white')
        
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Question Category')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Category-wise Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig3_category_comparison.png'))
    plt.close()
    print("  ✓ Figure 3: Category comparison")


# ========== FIGURE 4: Radar/Spider Chart ==========
def plot_radar_chart():
    categories = ['Comparison', 'Count', 'Exist', 'Object', 'Status']
    cat_keys = ['comparison', 'count', 'exist', 'object', 'status']
    
    key_models = ['BEV (Baseline)', 'YOLO', 'Trimodal V1', 'Trimodal V4 (Best)']
    key_models = [m for m in key_models if m in BEST_RESULTS]
    
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['#2196F3', '#FF9800', '#00BCD4', '#E91E63']
    
    for idx, model in enumerate(key_models):
        values = [BEST_RESULTS[model].get(cat, 0) for cat in cat_keys]
        values += values[:1]  # close polygon
        
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx % len(colors)],
                label=model, markersize=6)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=13)
    ax.set_ylim(0, 90)
    ax.set_title('Category-wise Performance Comparison', y=1.08, fontsize=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig4_radar_chart.png'))
    plt.close()
    print("  ✓ Figure 4: Radar chart")


# ========== FIGURE 5: Model Evolution Bar Chart ==========
def plot_model_evolution():
    # Ordered by development timeline
    model_order = [
        'BEV (Baseline)',
        'YOLO',
        'BEV-YOLO Fusion',
        'RadarXFormer',
        'RadarXF-BEV Fusion',
        'Trimodal V1',
        'Trimodal V3 (QType)',
        'Trimodal V4 (Best)',
    ]
    model_order = [m for m in model_order if m in BEST_RESULTS]
    
    overall_accs = [BEST_RESULTS[m]['overall'] for m in model_order]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color gradient from blue to red
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(model_order)))
    
    bars = ax.barh(range(len(model_order)), overall_accs, color=colors, 
                    edgecolor='white', height=0.6)
    
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order)
    ax.set_xlabel('Overall Accuracy (%)')
    ax.set_title('Model Architecture Evolution — Overall Accuracy')
    ax.set_xlim(52, 58)
    
    # Add value labels
    for bar, acc in zip(bars, overall_accs):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2.,
                f'{acc:.2f}%', ha='left', va='center', fontweight='bold', fontsize=11)
    
    # Add improvement arrow annotation
    baseline = overall_accs[0]
    best = overall_accs[-1]
    ax.annotate(f'  +{best - baseline:.2f}%', xy=(best, len(model_order)-1),
                fontsize=12, fontweight='bold', color='darkgreen',
                xytext=(best + 0.3, len(model_order)-1.5),
                arrowprops=dict(arrowstyle='->', color='darkgreen'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig5_model_evolution.png'))
    plt.close()
    print("  ✓ Figure 5: Model evolution")


# ========== FIGURE 6: Question Type Distribution ==========
def plot_question_distribution():
    # From the eval data: total questions per type
    qtypes = {
        'Exist': 24634,
        'Count': 16471,
        'Object': 17446,
        'Comparison': 12809,
        'Status': 11977,
    }
    
    total = sum(qtypes.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
    explode = (0.05, 0.05, 0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax1.pie(
        qtypes.values(), labels=qtypes.keys(), autopct='%1.1f%%',
        colors=colors, explode=explode, startangle=90,
        textprops={'fontsize': 12}
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    ax1.set_title(f'Question Type Distribution\n(Total: {total:,} questions)')
    
    # Accuracy by type for best model
    best = BEST_RESULTS['Trimodal V4 (Best)']
    cats = ['Exist', 'Count', 'Object', 'Comparison', 'Status']
    cat_keys = ['exist', 'count', 'object', 'comparison', 'status']
    accs = [best[k] for k in cat_keys]
    
    bars = ax2.bar(cats, accs, color=colors, alpha=0.85, edgecolor='white')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Best Model (Trimodal V4) — Per-Type Accuracy')
    ax2.set_ylim(0, 100)
    
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig6_question_distribution.png'))
    plt.close()
    print("  ✓ Figure 6: Question distribution")


# ========== FIGURE 7: V4 Training Detail — Loss + Accuracy ==========
def plot_v4_training_detail():
    if 'Trimodal V4 (Best)' not in model_data:
        print("  [SKIP] V4 data not available")
        return
    
    data = model_data['Trimodal V4 (Best)']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Loss curve
    ax1.plot(data['epochs'], data['losses'], 'b-o', linewidth=2, markersize=4, color='#E91E63')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Trimodal V4 — Training Loss')
    ax1.axvline(x=7, color='gray', linestyle='--', alpha=0.5, label='LR Decay (0.5×)')
    ax1.axvline(x=10, color='gray', linestyle=':', alpha=0.5, label='LR Decay (0.5×)')
    ax1.legend()
    
    # Category-wise accuracy over epochs
    if data['evals']:
        eval_epochs = [e['epoch'] for e in data['evals']]
        categories = ['overall', 'comparison', 'count', 'exist', 'object', 'status']
        cat_colors = ['#000000', '#9C27B0', '#2196F3', '#4CAF50', '#FF9800', '#F44336']
        cat_labels = ['Overall', 'Comparison', 'Count', 'Exist', 'Object', 'Status']
        
        for cat, col, lbl in zip(categories, cat_colors, cat_labels):
            vals = [e.get(cat, 0) for e in data['evals']]
            lw = 3 if cat == 'overall' else 1.5
            ax2.plot(eval_epochs, vals, '-o', color=col, linewidth=lw, markersize=4, label=lbl)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Trimodal V4 — Category Accuracy per Epoch')
        ax2.legend(loc='center right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig7_v4_training_detail.png'))
    plt.close()
    print("  ✓ Figure 7: V4 training detail")


# ========== FIGURE 8: Ablation — Modality Contributions ==========
def plot_ablation():
    """Shows the incremental gain of each modality."""
    ablation = {
        'Camera BEV only': 55.80,
        '+ LiDAR BEV': 55.90,  # slight from lidar concat
        '+ RadarXFormer': 56.10,  # trimodal V1
        '+ QType Weights': 56.06,  # V3
        '+ Fine-tuning': 56.52,   # V4
    }
    
    models = list(ablation.keys())
    accs = list(ablation.values())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#E91E63']
    bars = ax.bar(models, accs, color=colors, alpha=0.85, edgecolor='white', width=0.6)
    
    ax.set_ylabel('Overall Accuracy (%)')
    ax.set_title('Ablation Study — Incremental Modality Contributions')
    ax.set_ylim(54, 58)
    
    # Add value + delta labels
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        if i > 0:
            delta = acc - accs[i-1]
            color = 'green' if delta > 0 else 'red'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 0.3,
                    f'({"+" if delta > 0 else ""}{delta:.2f})', ha='center', va='top',
                    fontsize=9, color=color, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig8_ablation.png'))
    plt.close()
    print("  ✓ Figure 8: Ablation study")


# ========== MAIN ==========
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Generating Report Figures")
    print("=" * 50 + "\n")
    
    print("Loading log data...")
    # Data is loaded above
    
    print("\nGenerating figures...")
    plot_loss_curves()
    plot_accuracy_progression()
    plot_category_comparison()
    plot_radar_chart()
    plot_model_evolution()
    plot_question_distribution()
    plot_v4_training_detail()
    plot_ablation()
    
    print(f"\n{'=' * 50}")
    print(f"  All figures saved to: {OUT_DIR}/")
    print(f"{'=' * 50}")
    print("\nFigures generated:")
    print("  fig1_loss_curves.png          — Training loss convergence")
    print("  fig2_accuracy_progression.png — Validation accuracy over epochs")
    print("  fig3_category_comparison.png  — Category-wise bar chart")
    print("  fig4_radar_chart.png          — Spider/radar chart")
    print("  fig5_model_evolution.png      — Architecture evolution timeline")
    print("  fig6_question_distribution.png— Question type distribution")
    print("  fig7_v4_training_detail.png   — V4 loss + accuracy detail")
    print("  fig8_ablation.png             — Ablation study")
