"""
Create a visual diagram showing the normalization flow before and after the fix
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_normalization_diagram():
    """Create a diagram showing the normalization flow"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # ============ BEFORE (Left) ============
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('BEFORE FIX (Inconsistent)', fontsize=16, fontweight='bold', color='red')
    
    # Step 1: Load GT
    box1 = FancyBboxPatch((1, 10), 8, 1, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax1.add_patch(box1)
    ax1.text(5, 10.5, 'Load GT Image', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrow down
    arrow1 = FancyArrowPatch((5, 10), (5, 9), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax1.add_artist(arrow1)
    
    # Step 2: Normalize GT
    box2 = FancyBboxPatch((1, 8), 8, 1, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax1.add_patch(box2)
    ax1.text(5, 8.5, 'Normalize GT (99th percentile)', ha='center', va='center', fontsize=10)
    ax1.text(5, 8.2, 'max = 1.0', ha='center', va='center', fontsize=9, style='italic')
    
    # Split into two paths
    arrow2a = FancyArrowPatch((3, 8), (2, 7), arrowstyle='->', mutation_scale=20, linewidth=2, color='blue')
    ax1.add_artist(arrow2a)
    arrow2b = FancyArrowPatch((7, 8), (8, 7), arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax1.add_artist(arrow2b)
    
    # Left path: Target (stays as is)
    box3a = FancyBboxPatch((0.5, 5.5), 3, 1.5, boxstyle="round,pad=0.1", 
                           edgecolor='blue', facecolor='lightcyan', linewidth=2)
    ax1.add_patch(box3a)
    ax1.text(2, 6.5, 'TARGET', ha='center', va='center', fontsize=11, fontweight='bold', color='blue')
    ax1.text(2, 6.1, '(Ground Truth)', ha='center', va='center', fontsize=9)
    ax1.text(2, 5.8, 'Normalized: max=1.0', ha='center', va='center', fontsize=8, style='italic')
    
    # Right path: Input (gets degraded and re-normalized)
    box3b = FancyBboxPatch((6.5, 6), 3, 1, boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax1.add_patch(box3b)
    ax1.text(8, 6.5, 'Apply Degradation', ha='center', va='center', fontsize=10)
    
    arrow3 = FancyArrowPatch((8, 6), (8, 5), arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax1.add_artist(arrow3)
    
    box4 = FancyBboxPatch((6.5, 3.5), 3, 1.5, boxstyle="round,pad=0.1", 
                          edgecolor='red', facecolor='#ffcccc', linewidth=3)
    ax1.add_patch(box4)
    ax1.text(8, 4.5, 'INPUT', ha='center', va='center', fontsize=11, fontweight='bold', color='red')
    ax1.text(8, 4.1, '(Degraded)', ha='center', va='center', fontsize=9)
    ax1.text(8, 3.8, 'RE-normalized!', ha='center', va='center', fontsize=8, 
             style='italic', fontweight='bold', color='red')
    
    # Problem indicator
    arrow4a = FancyArrowPatch((2, 5.5), (2, 3), arrowstyle='->', mutation_scale=20, linewidth=2, color='blue')
    ax1.add_artist(arrow4a)
    arrow4b = FancyArrowPatch((8, 3.5), (8, 3), arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax1.add_artist(arrow4b)
    
    # Problem box
    problem_box = FancyBboxPatch((1, 0.5), 8, 2, boxstyle="round,pad=0.1", 
                                 edgecolor='red', facecolor='#ffeeee', linewidth=3)
    ax1.add_patch(problem_box)
    ax1.text(5, 2, '⚠️ PROBLEM ⚠️', ha='center', va='center', fontsize=12, 
             fontweight='bold', color='red')
    ax1.text(5, 1.5, 'Input and Target have', ha='center', va='center', fontsize=10)
    ax1.text(5, 1.1, 'DIFFERENT normalization scales!', ha='center', va='center', fontsize=10)
    ax1.text(5, 0.7, 'Inconsistent training!', ha='center', va='center', fontsize=9, style='italic')
    
    # ============ AFTER (Right) ============
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('AFTER FIX (Consistent)', fontsize=16, fontweight='bold', color='green')
    
    # Step 1: Load GT
    box1 = FancyBboxPatch((1, 10), 8, 1, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax2.add_patch(box1)
    ax2.text(5, 10.5, 'Load GT Image', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrow down
    arrow1 = FancyArrowPatch((5, 10), (5, 9), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax2.add_artist(arrow1)
    
    # Step 2: Initial normalize
    box2 = FancyBboxPatch((1, 8), 8, 1, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax2.add_patch(box2)
    ax2.text(5, 8.5, 'Initial Normalize', ha='center', va='center', fontsize=10)
    
    # Split into two paths
    arrow2a = FancyArrowPatch((3, 8), (2, 7), arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax2.add_artist(arrow2a)
    arrow2b = FancyArrowPatch((7, 8), (8, 7), arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax2.add_artist(arrow2b)
    
    # Left path: Target (will be normalized)
    box3a = FancyBboxPatch((0.5, 6), 3, 1, boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor='lightcyan', linewidth=2)
    ax2.add_patch(box3a)
    ax2.text(2, 6.5, 'Keep as TARGET', ha='center', va='center', fontsize=10)
    
    # Right path: Input (gets degraded)
    box3b = FancyBboxPatch((6.5, 6), 3, 1, boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax2.add_patch(box3b)
    ax2.text(8, 6.5, 'Apply Degradation', ha='center', va='center', fontsize=10)
    
    # Arrows to normalization step
    arrow3a = FancyArrowPatch((2, 6), (3, 5), arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax2.add_artist(arrow3a)
    arrow3b = FancyArrowPatch((8, 6), (7, 5), arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax2.add_artist(arrow3b)
    
    # NEW: Unified normalization step
    unified_box = FancyBboxPatch((2, 3.5), 6, 1.5, boxstyle="round,pad=0.1", 
                                 edgecolor='green', facecolor='#ccffcc', linewidth=3)
    ax2.add_patch(unified_box)
    ax2.text(5, 4.7, '✓ UNIFIED NORMALIZATION ✓', ha='center', va='center', fontsize=11, 
             fontweight='bold', color='green')
    ax2.text(5, 4.3, 'Calculate TARGET 99th percentile', ha='center', va='center', fontsize=9)
    ax2.text(5, 3.9, 'Normalize BOTH with same reference', ha='center', va='center', fontsize=9)
    
    # Output arrows
    arrow4a = FancyArrowPatch((3, 3.5), (2, 2.5), arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax2.add_artist(arrow4a)
    arrow4b = FancyArrowPatch((7, 3.5), (8, 2.5), arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax2.add_artist(arrow4b)
    
    # Final outputs
    box5a = FancyBboxPatch((0.5, 1), 3, 1.5, boxstyle="round,pad=0.1", 
                           edgecolor='green', facecolor='lightcyan', linewidth=2)
    ax2.add_patch(box5a)
    ax2.text(2, 2, 'TARGET', ha='center', va='center', fontsize=11, fontweight='bold', color='green')
    ax2.text(2, 1.6, '(Ground Truth)', ha='center', va='center', fontsize=9)
    ax2.text(2, 1.2, 'Normalized: max=1.0', ha='center', va='center', fontsize=8, style='italic')
    
    box5b = FancyBboxPatch((6.5, 1), 3, 1.5, boxstyle="round,pad=0.1", 
                           edgecolor='green', facecolor='lightyellow', linewidth=2)
    ax2.add_patch(box5b)
    ax2.text(8, 2, 'INPUT', ha='center', va='center', fontsize=11, fontweight='bold', color='green')
    ax2.text(8, 1.6, '(Degraded)', ha='center', va='center', fontsize=9)
    ax2.text(8, 1.2, 'Same scale as target!', ha='center', va='center', fontsize=8, 
             style='italic', fontweight='bold', color='green')
    
    # Solution indicator
    solution_box = FancyBboxPatch((1, -0.5), 8, 0.8, boxstyle="round,pad=0.1", 
                                  edgecolor='green', facecolor='#eeffee', linewidth=2)
    ax2.add_patch(solution_box)
    ax2.text(5, -0.1, '✓ CONSISTENT: Both use same normalization reference!', 
             ha='center', va='center', fontsize=10, fontweight='bold', color='green')
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    from pathlib import Path
    
    print("Creating normalization flow diagram...")
    fig = create_normalization_diagram()
    
    output_path = Path(__file__).parent.parent / 'normalization_flow_diagram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved diagram to: {output_path}")
    
    plt.show()

