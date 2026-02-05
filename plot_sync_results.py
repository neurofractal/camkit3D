"""
Plotting functions for video synchronization analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
import matplotlib.patches as mpatches


def plot_sync_results(
    results: Dict,
    trial_folder: str,
    figsize: tuple = (16, 12),
    save_plots: bool = True,
    show_plots: bool = True
):
    """
    Create comprehensive plots for synchronization analysis.
    
    Plots:
    1. Absolute frame times (original vs synchronized) vs ideal
    2. Timing error from ideal (original vs synchronized)
    3. Frame duration over time (scatter plot)
    4. Dropped frames for each camera over time
    
    Args:
        results: Results dictionary from synchronize_videos_to_ideal_fps()
        trial_folder: Path to trial folder
        figsize: Figure size (width, height)
        save_plots: Whether to save plots to disk
        show_plots: Whether to display plots
    
    Returns:
        Dictionary of figure objects
    """
    trial_folder = Path(trial_folder)
    
    # Extract info from results
    timestamps_original = results['timestamps_original']
    target_fps = results['target_fps']
    camera_ids = results['camera_ids']
    frame_count = results['frame_count']
    ideal_times = results['ideal_times']
    
    # Get synchronized timestamps (mapped to ideal times)
    timestamps_synced = {}
    
    for cam_id in camera_ids:
        # Get the mapped indices
        frame_map = results['frame_mappings'][cam_id]
        # Map to the times of the selected frames
        timestamps_synced[cam_id] = timestamps_original[cam_id][frame_map]
    
    # Reference time (global start)
    t0 = results['global_start']
    
    # Define bright, distinct colors for each camera
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(camera_ids), 9)))[:len(camera_ids)]
    cam_colors = {cam_id: colors[i] for i, cam_id in enumerate(camera_ids)}
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # PLOT 1: Absolute Frame Times (Original) vs Ideal
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    for cam_id in camera_ids:
        times_orig = timestamps_original[cam_id]
        # Relative to start of cam0
        times_rel = times_orig - t0
        ax1.plot(times_rel, 
                linewidth=2, 
                alpha=0.8,
                color=cam_colors[cam_id],
                label=f'Cam {cam_id} (original)')
    
    # Plot ideal
    ideal_rel = ideal_times - t0
    ax1.plot(ideal_rel, 
            'k--', 
            linewidth=1.5, 
            alpha=0.5,
            label=f'Ideal {target_fps} fps')
    
    ax1.set_xlabel('Frame Number', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Time Since Start (s)', fontsize=11, fontweight='bold')
    ax1.set_title('Original Frame Times vs Ideal', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 2: Absolute Frame Times (Synchronized) vs Ideal
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    for cam_id in camera_ids:
        times_sync = timestamps_synced[cam_id]
        times_rel = times_sync - t0
        ax2.plot(times_rel, 
                linewidth=2, 
                alpha=0.8,
                color=cam_colors[cam_id],
                label=f'Cam {cam_id} (synced)')
    
    # Plot ideal
    ax2.plot(ideal_rel, 
            'k--', 
            linewidth=1.5, 
            alpha=0.5,
            label=f'Ideal {target_fps} fps')
    
    ax2.set_xlabel('Frame Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Time Since Start (s)', fontsize=11, fontweight='bold')
    ax2.set_title('Synchronized Frame Times vs Ideal', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 3: Timing Error from Ideal (Original)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    for cam_id in camera_ids:
        times_orig = timestamps_original[cam_id]
        
        # For original, we need to match to ideal based on temporal proximity
        # Use first len(ideal_times) timestamps
        n_compare = min(len(times_orig), len(ideal_times))
        error_ms = (times_orig[:n_compare] - ideal_times[:n_compare]) * 1000
        
        mean_error = np.mean(error_ms)
        
        ax3.plot(error_ms, 
                linewidth=2, 
                alpha=0.8,
                color=cam_colors[cam_id],
                label=f'Cam {cam_id} (mean: {mean_error:.2f}ms)')
    
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Frame Number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Timing Error (ms)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Original: Error from Ideal {target_fps} FPS', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 4: Timing Error from Ideal (Synchronized)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    for cam_id in camera_ids:
        times_sync = timestamps_synced[cam_id]
        error_ms = (times_sync - ideal_times) * 1000
        mean_error = np.mean(error_ms)
        
        ax4.plot(error_ms, 
                linewidth=2, 
                alpha=0.8,
                color=cam_colors[cam_id],
                label=f'Cam {cam_id} (mean: {mean_error:.2f}ms)')
    
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Frame Number', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Timing Error (ms)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Synchronized: Error from Ideal {target_fps} FPS', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 5: Frame Duration Over Time (Scatter)
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, :])
    
    for cam_id in camera_ids:
        times_orig = timestamps_original[cam_id]
        
        # Calculate frame durations (time between consecutive frames)
        if len(times_orig) > 1:
            durations_ms = np.diff(times_orig) * 1000  # Convert to ms
            frame_indices = np.arange(len(durations_ms))
            
            ax5.scatter(frame_indices, 
                       durations_ms,
                       s=3,  # Small dots
                       alpha=0.6,
                       color=cam_colors[cam_id],
                       label=f'Cam {cam_id}')
    
    # Add ideal frame duration line
    ideal_duration_ms = 1000.0 / target_fps
    ax5.axhline(y=ideal_duration_ms, 
               color='k', 
               linestyle='--', 
               linewidth=2, 
               alpha=0.7,
               label=f'Ideal ({ideal_duration_ms:.2f}ms @ {target_fps}fps)')
    
    ax5.set_xlabel('Frame Number', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frame Duration (ms)', fontsize=11, fontweight='bold')
    ax5.set_title('Frame Duration Over Time (Original)', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=9, markerscale=3)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(bottom=0)
    
    # =========================================================================
    # PLOT 6: Dropped Frames for All Cameras Over Time
    # =========================================================================
    ax6 = fig.add_subplot(gs[3, :])
    
    # Analyze dropped frames for each camera
    for cam_id in camera_ids:
        cam_times = timestamps_original[cam_id]
        
        # Detect dropped frames
        if len(cam_times) > 1:
            durations = np.diff(cam_times)
            expected_duration = 1.0 / target_fps
            
            # Define "dropped" as duration > 1.5x expected
            # Number of dropped frames = round(duration / expected) - 1
            dropped_counts = np.maximum(0, np.round(durations / expected_duration).astype(int) - 1)
            
            # Cumulative dropped frames
            cumulative_dropped = np.cumsum(dropped_counts)
            frame_indices = np.arange(len(cumulative_dropped))
            
            total_dropped = int(cumulative_dropped[-1]) if len(cumulative_dropped) > 0 else 0
            
            # Plot cumulative dropped frames
            ax6.plot(frame_indices, 
                    cumulative_dropped,
                    linewidth=2.5,
                    color=cam_colors[cam_id],
                    label=f'Cam {cam_id} (total: {total_dropped})',
                    alpha=0.8)
    
    ax6.set_xlabel('Original Frame Number', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Cumulative Dropped Frames', fontsize=11, fontweight='bold')
    ax6.set_title(f'Dropped Frames Over Time (All Cameras)', 
                 fontsize=12, fontweight='bold')
    ax6.legend(loc='best', fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(bottom=0)
    
    # =========================================================================
    # Add overall title
    # =========================================================================
    fig.suptitle(f'Synchronization Analysis - {trial_folder.name}', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # =========================================================================
    # Save plots
    # =========================================================================
    figs = {'main': fig}
    
    if save_plots:
        output_dir = trial_folder / "synchronization_plots"
        output_dir.mkdir(exist_ok=True)
        
        plot_path = output_dir / "sync_analysis.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plots saved to: {plot_path}")
        
        # Also save as PDF for publication quality
        pdf_path = output_dir / "sync_analysis.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"📊 PDF saved to: {pdf_path}")
    
    if show_plots:
        plt.show()
    
    return figs


def plot_sync_summary_stats(results: Dict, trial_folder: str, save_plots: bool = True):
    """
    Create summary statistics plots.
    
    Plots:
    1. Bar chart of mean timing errors per camera
    2. Box plot of timing errors per camera
    
    Args:
        results: Results dictionary from synchronize_videos_to_ideal_fps()
        trial_folder: Path to trial folder
        save_plots: Whether to save plots
    
    Returns:
        Figure object
    """
    trial_folder = Path(trial_folder)
    
    camera_ids = results['camera_ids']
    sync_metrics = results['sync_metrics']
    target_fps = results['target_fps']
    
    # Define colors
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(camera_ids), 9)))[:len(camera_ids)]
    cam_colors = {cam_id: colors[i] for i, cam_id in enumerate(camera_ids)}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # =========================================================================
    # PLOT 1: Mean Timing Errors Bar Chart
    # =========================================================================
    cam_labels = []
    mean_errors = []
    rms_errors = []
    max_errors = []
    bar_colors = []
    
    for cam_id in camera_ids:
        metrics = sync_metrics[cam_id]
        cam_labels.append(f'Cam {cam_id}')
        mean_errors.append(metrics['mean_diff_ms'])
        rms_errors.append(metrics['rms_diff_ms'])
        max_errors.append(metrics['max_diff_ms'])
        bar_colors.append(cam_colors[cam_id])
    
    x = np.arange(len(cam_labels))
    width = 0.25
    
    ax1.bar(x - width, mean_errors, width, label='Mean', color=bar_colors, alpha=0.8)
    ax1.bar(x, rms_errors, width, label='RMS', color=bar_colors, alpha=0.6)
    ax1.bar(x + width, max_errors, width, label='Max', color=bar_colors, alpha=0.4)
    
    ax1.set_xlabel('Camera', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Timing Error (ms)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Timing Errors vs Ideal {target_fps} FPS', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cam_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # PLOT 2: Box Plot of Timing Errors
    # =========================================================================
    timing_diffs_all = []
    box_colors = []
    
    for cam_id in camera_ids:
        metrics = sync_metrics[cam_id]
        timing_diffs_all.append(metrics['time_diffs_ms'])
        box_colors.append(cam_colors[cam_id])
    
    bp = ax2.boxplot(timing_diffs_all, 
                     labels=cam_labels,
                     patch_artist=True,
                     widths=0.6)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_xlabel('Camera', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Timing Error (ms)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Timing Error Distribution vs Ideal {target_fps} FPS', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    
    fig.suptitle(f'Synchronization Summary - {trial_folder.name}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        output_dir = trial_folder / "synchronization_plots"
        output_dir.mkdir(exist_ok=True)
        
        plot_path = output_dir / "sync_summary.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Summary plots saved to: {plot_path}")
    
    return fig


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    """
    Example usage - paste into notebook after running synchronization:
    
    from sync_by_timestamps import synchronize_videos_to_ideal_fps
    from plot_sync_results import plot_sync_results, plot_sync_summary_stats
    
    # Run synchronization to ideal FPS
    trial_dir = "./recordings/2026-02-04_12-00-00"
    results = synchronize_videos_to_ideal_fps(
        trial_folder=trial_dir,
        target_fps=30.0  # Your target FPS
    )
    
    # Create comprehensive plots
    figs = plot_sync_results(
        results=results,
        trial_folder=trial_dir,
        save_plots=True,
        show_plots=True
    )
    
    # Create summary statistics plots
    fig_summary = plot_sync_summary_stats(
        results=results,
        trial_folder=trial_dir,
        save_plots=True
    )
    """
    pass
