"""
scratch/generate_pipeline_diagram.py
------------------------------------
Generates a professional, high-resolution block diagram (flowchart)
of the two-stage EDSR + LaMa neural restoration architecture.
Saves the flowchart to paper_visuals/pipeline_flowchart.png at 300 DPI.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add workspace root to python path to run local package imports
workspace_root = "/Users/rajnishsingh/Downloads/files (3)"
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


def main():
    print("=== PIPELINE FLOWCHART GENERATOR ===")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Define styles
    box_props_input = dict(
        boxstyle="round,pad=0.4",
        facecolor="#F1F5F9",
        edgecolor="#64748B",
        linewidth=1.5,
    )
    box_props_stage1 = dict(
        boxstyle="round,pad=0.5",
        facecolor="#EFF6FF",
        edgecolor="#2563EB",
        linewidth=2.0,
    )
    box_props_stage2 = dict(
        boxstyle="round,pad=0.5",
        facecolor="#FDF2F8",
        edgecolor="#DB2777",
        linewidth=2.0,
    )
    box_props_data = dict(
        boxstyle="round,pad=0.4",
        facecolor="#ECFDF5",
        edgecolor="#10B981",
        linewidth=1.5,
    )

    # 1. Damaged Input Box (Left-Center)
    ax.text(
        1.2,
        3.0,
        "Damaged Input Image\n(256 $\\times$ 256 $\\times$ 3)",
        ha="center",
        va="center",
        bbox=box_props_input,
        fontsize=9,
        fontweight="bold",
        color="#1E293B",
    )

    # 2. Bicubic Downsample Box (Upper-Left)
    ax.text(
        3.8,
        4.5,
        "Bicubic Downsample\n(128 $\\times$ 128 $\\times$ 3)",
        ha="center",
        va="center",
        bbox=box_props_data,
        fontsize=8,
        color="#047857",
    )

    # 3. Stage 1: EDSR Box
    ax.text(
        6.8,
        4.5,
        "Stage 1: EDSR Network\nSuper-Resolution upscaling",
        ha="center",
        va="center",
        bbox=box_props_stage1,
        fontsize=9,
        fontweight="bold",
        color="#1D4ED8",
    )

    # 4. Auto-Masking Box (Lower-Left)
    ax.text(
        3.8,
        1.5,
        "Heuristic Auto-Masking\n(White/Saturation Detect)",
        ha="center",
        va="center",
        bbox=box_props_data,
        fontsize=8,
        color="#047857",
    )

    # 5. Damage Mask Box
    ax.text(
        6.8,
        1.5,
        "Binary Damage Mask\n(256 $\\times$ 256 $\\times$ 1)",
        ha="center",
        va="center",
        bbox=box_props_input,
        fontsize=9,
        color="#1E293B",
    )

    # 6. Feature Concatenation Circle/Box
    ax.text(
        8.8,
        3.0,
        "Concat\n(4 channels)",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="circle,pad=0.3",
            facecolor="#F5F5F4",
            edgecolor="#78716C",
            linewidth=1.5,
        ),
        fontsize=8,
        fontweight="semibold",
    )

    # 7. Stage 2: LaMa Generator
    ax.text(
        8.8,
        4.8,
        "Stage 2: LaMa\nInpainting Generator\n(FFC Bottleneck)",
        ha="center",
        va="center",
        bbox=box_props_stage2,
        fontsize=9,
        fontweight="bold",
        color="#BE185D",
    )

    # 8. Restored Output Box
    ax.text(
        5.0,
        3.0,
        "Restored Output\n(256 $\\times$ 256 $\\times$ 3)",
        ha="center",
        va="center",
        bbox=box_props_data,
        fontsize=9,
        fontweight="bold",
        color="#065F46",
    )

    # Draw connections (Arrows)
    arrow_style = dict(arrowstyle="->", color="#475569", lw=1.5, mutation_scale=15)

    # Input -> Downsample & Input -> Auto-Masking
    ax.annotate("", xy=(2.6, 4.5), xytext=(1.2, 3.5), arrowprops=arrow_style)
    ax.annotate("", xy=(2.6, 1.5), xytext=(1.2, 2.5), arrowprops=arrow_style)

    # Downsample -> EDSR
    ax.annotate("", xy=(5.2, 4.5), xytext=(5.0, 4.5), arrowprops=arrow_style)

    # Auto-Masking -> Mask
    ax.annotate("", xy=(5.2, 1.5), xytext=(5.0, 1.5), arrowprops=arrow_style)

    # EDSR Output -> Concat
    ax.annotate("", xy=(8.8, 3.5), xytext=(8.0, 4.5), arrowprops=arrow_style)

    # Mask -> Concat
    ax.annotate("", xy=(8.8, 2.5), xytext=(8.0, 1.5), arrowprops=arrow_style)

    # Concat -> LaMa
    ax.annotate("", xy=(8.8, 4.1), xytext=(8.8, 3.6), arrowprops=arrow_style)

    # LaMa -> Restored Output
    ax.annotate("", xy=(6.2, 3.0), xytext=(7.4, 4.8), arrowprops=arrow_style)

    # Save image
    output_dir = "/Users/rajnishsingh/Downloads/files (3)/paper_visuals"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "pipeline_flowchart.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Pipeline flowchart successfully generated at: {save_path}")


if __name__ == "__main__":
    main()
