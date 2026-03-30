from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from workflow_16s.utils.logger import get_logger


COMPREHENSIVE_V_REGIONS = {
    "V1": {"fwd_pos": 69, "rev_pos": 99, "leeway": 40},
    "V2": {"fwd_pos": 137, "rev_pos": 242, "leeway": 40},
    "V3": {"fwd_pos": 433, "rev_pos": 497, "leeway": 40},
    "V4": {"fwd_pos": 576, "rev_pos": 682, "leeway": 50},
    "V5": {"fwd_pos": 822, "rev_pos": 879, "leeway": 40},
    "V6": {"fwd_pos": 986, "rev_pos": 1043, "leeway": 40},
    "V7": {"fwd_pos": 1117, "rev_pos": 1173, "leeway": 40},
    "V8": {"fwd_pos": 1243, "rev_pos": 1294, "leeway": 40},
    "V9": {"fwd_pos": 1435, "rev_pos": 1465, "leeway": 40},
    "V1-V2": {"fwd_pos": 27, "rev_pos": 338, "leeway": 40},
    "V1-V3": {"fwd_pos": 27, "rev_pos": 534, "leeway": 50},
    "V2-V3": {"fwd_pos": 338, "rev_pos": 534, "leeway": 50},
    "V3-V4": {"fwd_pos": 341, "rev_pos": 805, "leeway": 50},
    "V4-V5": {"fwd_pos": 515, "rev_pos": 926, "leeway": 50},
    "V5-V7": {"fwd_pos": 785, "rev_pos": 1100, "leeway": 60},
    "V6-V8": {"fwd_pos": 926, "rev_pos": 1392, "leeway": 75},
    "V7-V9": {"fwd_pos": 1100, "rev_pos": 1492, "leeway": 100},
    "Full-Length": {"fwd_pos": 27, "rev_pos": 1492, "leeway": 100},
}

def create_alignment_plot(
    results_df: pd.DataFrame, 
    output_path: Path
):
    """Generates and saves a plot visualizing the predicted alignment regions for each run."""
    logger = get_logger("workflow_16s")
    predictions = results_df[results_df['is_prediction']].copy()
    if predictions.empty:
        logger.warning("No successful predictions to plot. Skipping alignment plot generation.")
        return

    # Sort runs by their start position for a cleaner plot
    predictions.sort_values('vsearch_avg_start', inplace=True)
    run_order = predictions['run_accession'].tolist()

    # Create a dynamic figure size based on the number of runs
    fig_height = max(6, len(run_order) * 0.4 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # 1. Plot the full 16S gene backbone
    ax.hlines(
        y=run_order, 
        xmin=1, 
        xmax=1542, 
        color='lightgrey', 
        alpha=0.7, 
        linewidth=5, 
        label='16S Gene Backbone'
    )

    # 2. Plot the predicted alignment for each run
    for _, row in predictions.iterrows():
        run_id = row['run_accession']
        start, end = row['vsearch_avg_start'], row['vsearch_avg_end']
        ax.hlines(
            y=run_id, 
            xmin=start, 
            xmax=end, 
            color='coral', 
            linewidth=5, 
            label=f'Predicted Fragment ({row["region"]})'
        )
        ax.text(
            end + 10, 
            run_id, 
            row['region'], 
            va='center', 
            ha='left', 
            fontsize=9, 
            color='black'
        )

    # 3. Overlay the canonical V-regions for reference
    for region, params in COMPREHENSIVE_V_REGIONS.items():
        ax.axvspan(
            params['fwd_pos'], 
            params['rev_pos'], 
            alpha=0.15, 
            color='skyblue', 
            zorder=0
        )
        ax.text(
            (params['fwd_pos'] + params['rev_pos']) / 2, 
            ax.get_ylim()[1], region, 
            ha='center', 
            va='bottom', 
            fontsize=8, 
            color='blue', 
            alpha=0.8
        )

    # Formatting
    ax.set_xlabel("Position on 16S rRNA Gene (bp)", fontsize=12)
    ax.set_ylabel("Run Accession", fontsize=12)
    ax.set_title(
        "Predicted 16S Subfragment Alignments", 
        fontsize=14, 
        weight='bold'
    )
    ax.set_xlim(0, 1600)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Clean up legend (remove duplicate labels)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')

    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Alignment visualization saved to: {output_path}")
    except Exception as e: 
        logger.error(f"Failed to save alignment plot: {e}")
    finally: 
        plt.close(fig)