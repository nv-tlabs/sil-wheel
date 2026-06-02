# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
import re
import sqlite3
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_trajectory_data(clip_to_idx, traj_mmap, clip_id):
    """Get full trajectory data for a clip.

    Columns: x, y, z, speed, accel, jerk, curvature
    """
    start, end = clip_to_idx[clip_id]
    return traj_mmap[start:end, :]


def _safe_slug(name: str) -> str:
    try:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_").lower()
    except Exception:
        return "dataset"


def visualize_trajectories(
    all_clip_ids,
    clip_to_idx,
    traj_mmap,
    n_samples=100000,
    seed=42,
    out_dir=None,
    dataset_name=None,
):
    """
    Visualize trajectory time series data by randomly sampling clips.
    Creates plots for positions, speed, acceleration, jerk, and curvature.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Sample clip IDs (use global all_clip_ids loaded from safetensors)
    n_samples = min(n_samples, len(all_clip_ids))
    sampled_clip_ids = random.sample(all_clip_ids, n_samples)

    print(
        f"\nSampling {n_samples:,} clips from {len(all_clip_ids):,} total clips..."
    )

    # Collect statistics from all trajectories
    all_speeds = []
    all_accelerations = []
    all_jerks = []
    all_curvatures = []
    all_velocities_x = []
    all_velocities_y = []

    # Per-clip statistics
    clip_avg_speeds = []
    clip_max_speeds = []
    clip_avg_accelerations = []
    clip_max_accelerations = []
    clip_avg_jerks = []
    clip_max_jerks = []
    clip_avg_curvatures = []
    clip_max_curvatures = []
    clip_avg_velocity_y = []
    clip_max_velocity_y = []

    # For position plots, we'll sample a larger subset for visualization
    position_sample_size = min(1000, n_samples)
    position_clips = sampled_clip_ids[:position_sample_size]

    # Time interval for velocity computation (0.1 seconds)
    dt = 0.1

    print("Collecting trajectory data...")
    for clip_id in tqdm(sampled_clip_ids, desc="Processing clips"):
        traj_data = get_trajectory_data(clip_to_idx, traj_mmap, clip_id)

        # Extract position data (columns 0 and 1)
        pos_x = traj_data[:, 0]
        pos_y = traj_data[:, 1]

        # Compute velocity using np.gradient with 0.1s time intervals
        if len(pos_x) > 1:
            vel_x = np.gradient(pos_x, dt)
            vel_y = np.gradient(pos_y, dt)
            all_velocities_x.extend(vel_x.tolist())
            all_velocities_y.extend(vel_y.tolist())

            # Velocity magnitude
            vel_magnitude = np.sqrt(vel_x**2 + vel_y**2)
            clip_avg_velocity_y.append(np.mean(vel_y))
            clip_max_velocity_y.append(np.max(vel_y))

        # Collect all values for distribution plots
        speeds = traj_data[:, 3]
        accelerations = traj_data[:, 4]
        jerks = traj_data[:, 5]
        curvatures = traj_data[:, 6]

        all_speeds.extend(speeds.tolist())
        all_accelerations.extend(accelerations.tolist())
        all_jerks.extend(jerks.tolist())
        all_curvatures.extend(curvatures.tolist())

        # Per-clip statistics
        clip_avg_speeds.append(np.mean(speeds))
        clip_max_speeds.append(np.max(speeds))
        clip_avg_accelerations.append(np.mean(accelerations))
        clip_max_accelerations.append(np.max(accelerations))
        clip_avg_jerks.append(np.mean(jerks))
        clip_max_jerks.append(np.max(jerks))
        clip_avg_curvatures.append(np.mean(curvatures))
        clip_max_curvatures.append(np.max(curvatures))

    # Convert to numpy arrays for efficiency
    all_speeds = np.array(all_speeds)
    all_accelerations = np.array(all_accelerations)
    all_jerks = np.array(all_jerks)
    all_curvatures = np.array(all_curvatures)
    all_velocities_x = np.array(all_velocities_x)
    all_velocities_y = np.array(all_velocities_y)

    # Per-clip arrays
    clip_avg_speeds = np.array(clip_avg_speeds)
    clip_max_speeds = np.array(clip_max_speeds)
    clip_avg_accelerations = np.array(clip_avg_accelerations)
    clip_max_accelerations = np.array(clip_max_accelerations)
    clip_avg_jerks = np.array(clip_avg_jerks)
    clip_max_jerks = np.array(clip_max_jerks)
    clip_avg_curvatures = np.array(clip_avg_curvatures)
    clip_max_curvatures = np.array(clip_max_curvatures)
    clip_avg_velocity_y = np.array(clip_avg_velocity_y)
    clip_max_velocity_y = np.array(clip_max_velocity_y)

    print(f"\nTotal data points collected: {len(all_speeds):,}")

    # Set up matplotlib style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # Create figure with multiple subplots (2x5 layout)
    fig = plt.figure(figsize=(25, 12))
    fig.suptitle(
        f"Trajectory Data Visualization (n={n_samples:,} clips)",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    # Color palette
    colors = plt.cm.viridis(np.linspace(0, 1, position_sample_size))

    # ===== 1. Position Trajectories (X vs Y) - Show more trajectories =====
    ax1 = fig.add_subplot(2, 5, 1)
    n_traj_to_plot = min(500, len(position_clips))
    for i, clip_id in enumerate(position_clips[:n_traj_to_plot]):
        traj_data = get_trajectory_data(clip_to_idx, traj_mmap, clip_id)
        ax1.plot(
            traj_data[:, 0],
            traj_data[:, 1],
            alpha=0.25,
            linewidth=0.4,
            color=colors[i % len(colors)],
        )
    ax1.set_xlabel("X Position", fontsize=11)
    ax1.set_ylabel("Y Position", fontsize=11)
    ax1.set_title(
        f"Position Trajectories ({n_traj_to_plot} samples)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="datalim")

    # ===== 2. Velocity 2D Density Heatmap =====
    ax_vel = fig.add_subplot(2, 5, 2)
    # Clip to percentiles for better visualization
    vx_p1, vx_p99 = np.percentile(all_velocities_x, [1, 99])
    vy_p1, vy_p99 = np.percentile(all_velocities_y, [1, 99])
    mask = (
        (all_velocities_x >= vx_p1)
        & (all_velocities_x <= vx_p99)
        & (all_velocities_y >= vy_p1)
        & (all_velocities_y <= vy_p99)
    )
    vx_clipped = all_velocities_x[mask]
    vy_clipped = all_velocities_y[mask]

    # Create 2D histogram / heatmap
    heatmap, xedges, yedges = np.histogram2d(vx_clipped, vy_clipped, bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Use log scale for better visibility
    heatmap_log = np.log10(heatmap.T + 1)
    im_vel = ax_vel.imshow(
        heatmap_log,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="hot",
        interpolation="bilinear",
    )
    ax_vel.set_xlabel("Velocity X (units/s)", fontsize=11)
    ax_vel.set_ylabel("Velocity Y (units/s)", fontsize=11)
    ax_vel.set_title(
        "Velocity Distribution (2D Density, log scale)",
        fontsize=12,
        fontweight="bold",
    )
    plt.colorbar(im_vel, ax=ax_vel, label="log10(count + 1)", shrink=0.8)

    # ===== 3. Velocity Y Distribution (1D) =====
    ax_vy = fig.add_subplot(2, 5, 3)
    vy_clipped_1d = np.clip(all_velocities_y, vy_p1, vy_p99)
    ax_vy.hist(
        vy_clipped_1d, bins=100, color="#3498DB", edgecolor="white", alpha=0.8
    )
    ax_vy.axvline(
        np.mean(all_velocities_y),
        color="#E74C3C",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(all_velocities_y):.3f}",
    )
    ax_vy.axvline(
        np.median(all_velocities_y),
        color="#2ECC71",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(all_velocities_y):.3f}",
    )
    ax_vy.set_xlabel("Velocity Y (units/s)", fontsize=11)
    ax_vy.set_ylabel("Count", fontsize=11)
    ax_vy.set_title(
        "Velocity Y Distribution (1-99 percentile)",
        fontsize=12,
        fontweight="bold",
    )
    ax_vy.legend(fontsize=10)
    ax_vy.grid(True, alpha=0.3)

    # ===== 4. Speed Distribution =====
    ax3 = fig.add_subplot(2, 5, 4)
    # Clip outliers for better visualization
    speed_p1, speed_p99 = np.percentile(all_speeds, [1, 99])
    speed_clipped = np.clip(all_speeds, speed_p1, speed_p99)
    ax3.hist(
        speed_clipped, bins=100, color="#E67E22", edgecolor="white", alpha=0.8
    )
    ax3.axvline(
        np.mean(all_speeds),
        color="#E74C3C",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(all_speeds):.3f}",
    )
    ax3.axvline(
        np.median(all_speeds),
        color="#2ECC71",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(all_speeds):.3f}",
    )
    ax3.set_xlabel("Speed", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title(
        "Speed Distribution (1-99 percentile)", fontsize=12, fontweight="bold"
    )
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ===== 5. Acceleration Distribution =====
    ax4 = fig.add_subplot(2, 5, 5)
    accel_p1, accel_p99 = np.percentile(all_accelerations, [1, 99])
    accel_clipped = np.clip(all_accelerations, accel_p1, accel_p99)
    ax4.hist(
        accel_clipped, bins=100, color="#27AE60", edgecolor="white", alpha=0.8
    )
    ax4.axvline(
        np.mean(all_accelerations),
        color="#E74C3C",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(all_accelerations):.3f}",
    )
    ax4.axvline(
        np.median(all_accelerations),
        color="#3498DB",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(all_accelerations):.3f}",
    )
    ax4.set_xlabel("Acceleration", fontsize=11)
    ax4.set_ylabel("Count", fontsize=11)
    ax4.set_title(
        "Acceleration Distribution (1-99 percentile)",
        fontsize=12,
        fontweight="bold",
    )
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # ===== 6. Jerk Distribution =====
    ax5 = fig.add_subplot(2, 5, 6)
    jerk_p1, jerk_p99 = np.percentile(all_jerks, [1, 99])
    jerk_clipped = np.clip(all_jerks, jerk_p1, jerk_p99)
    ax5.hist(
        jerk_clipped, bins=100, color="#9B59B6", edgecolor="white", alpha=0.8
    )
    ax5.axvline(
        np.mean(all_jerks),
        color="#E74C3C",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(all_jerks):.3f}",
    )
    ax5.axvline(
        np.median(all_jerks),
        color="#1ABC9C",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(all_jerks):.3f}",
    )
    ax5.set_xlabel("Jerk", fontsize=11)
    ax5.set_ylabel("Count", fontsize=11)
    ax5.set_title(
        "Jerk Distribution (1-99 percentile)", fontsize=12, fontweight="bold"
    )
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # ===== 7. Curvature Distribution =====
    ax6 = fig.add_subplot(2, 5, 7)
    curv_p1, curv_p99 = np.percentile(all_curvatures, [1, 99])
    curv_clipped = np.clip(all_curvatures, curv_p1, curv_p99)
    ax6.hist(
        curv_clipped, bins=100, color="#F1C40F", edgecolor="white", alpha=0.8
    )
    ax6.axvline(
        np.mean(all_curvatures),
        color="#E74C3C",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(all_curvatures):.3f}",
    )
    ax6.axvline(
        np.median(all_curvatures),
        color="#8E44AD",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(all_curvatures):.3f}",
    )
    ax6.set_xlabel("Curvature", fontsize=11)
    ax6.set_ylabel("Count", fontsize=11)
    ax6.set_title(
        "Curvature Distribution (1-99 percentile)",
        fontsize=12,
        fontweight="bold",
    )
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # ===== 8. Sample Time Series - Speed =====
    ax7 = fig.add_subplot(2, 5, 8)
    time_series_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, clip_id in enumerate(sampled_clip_ids[:20]):
        traj_data = get_trajectory_data(clip_to_idx, traj_mmap, clip_id)
        time_axis = np.arange(len(traj_data))
        ax7.plot(
            time_axis,
            traj_data[:, 3],
            alpha=0.7,
            linewidth=1.0,
            color=time_series_colors[i],
        )
    ax7.set_xlabel("Frame", fontsize=11)
    ax7.set_ylabel("Speed", fontsize=11)
    ax7.set_title(
        "Speed Time Series (20 sample trajectories)",
        fontsize=12,
        fontweight="bold",
    )
    ax7.grid(True, alpha=0.3)

    # ===== 9. Sample Time Series - Acceleration =====
    ax8 = fig.add_subplot(2, 5, 9)
    for i, clip_id in enumerate(sampled_clip_ids[:20]):
        traj_data = get_trajectory_data(clip_to_idx, traj_mmap, clip_id)
        time_axis = np.arange(len(traj_data))
        ax8.plot(
            time_axis,
            traj_data[:, 4],
            alpha=0.7,
            linewidth=1.0,
            color=time_series_colors[i],
        )
    ax8.set_xlabel("Frame", fontsize=11)
    ax8.set_ylabel("Acceleration", fontsize=11)
    ax8.set_title(
        "Acceleration Time Series (20 sample trajectories)",
        fontsize=12,
        fontweight="bold",
    )
    ax8.grid(True, alpha=0.3)

    # ===== 10. Feature Correlation Heatmap =====
    ax9 = fig.add_subplot(2, 5, 10)
    # Sample for correlation computation
    sample_size_corr = min(1000000, len(all_speeds))
    indices = np.random.choice(len(all_speeds), sample_size_corr, replace=False)
    feature_matrix = np.column_stack(
        [
            all_speeds[indices],
            all_accelerations[indices],
            all_jerks[indices],
            all_curvatures[indices],
        ]
    )
    corr_matrix = np.corrcoef(feature_matrix.T)

    im = ax9.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    feature_names = ["Speed", "Accel", "Jerk", "Curvature"]
    ax9.set_xticks(range(4))
    ax9.set_yticks(range(4))
    ax9.set_xticklabels(feature_names, fontsize=11)
    ax9.set_yticklabels(feature_names, fontsize=11)
    ax9.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold")

    # Add correlation values as text
    for i in range(4):
        for j in range(4):
            text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax9.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax9, shrink=0.8, label="Correlation")

    plt.tight_layout()

    # Save the figure(s)
    base_dir = Path(out_dir or ".")
    base_dir.mkdir(parents=True, exist_ok=True)
    ds_slug = _safe_slug(dataset_name or "all")
    output_path = base_dir / f"trajectory_visualization_{ds_slug}.png"
    try:
        plt.savefig(
            output_path, dpi=150, bbox_inches="tight", facecolor="white"
        )
        print(f"\nSaved visualization to {output_path}")
    except Exception as e:
        print(f"Could not save to {output_path}: {e}")

    # ===== Create additional detailed plots =====

    # Figure 2: Box plots and log-scale histograms
    fig2, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig2.suptitle(
        "Trajectory Feature Statistics (All Data Points)",
        fontsize=16,
        fontweight="bold",
    )

    features = [
        (all_speeds, "Speed", "#E67E22"),
        (all_accelerations, "Acceleration", "#27AE60"),
        (all_jerks, "Jerk", "#9B59B6"),
        (all_curvatures, "Curvature", "#F1C40F"),
    ]

    for idx, (data, name, color) in enumerate(features):
        # Box plot
        ax_box = axes[0, idx]
        bp = ax_box.boxplot([data], vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)
        ax_box.set_ylabel(name, fontsize=11)
        ax_box.set_title(f"{name} Box Plot", fontsize=12, fontweight="bold")
        ax_box.set_xticklabels([""])
        ax_box.grid(True, alpha=0.3)

        # Log-scale histogram for full range
        ax_hist = axes[1, idx]
        # Remove zeros and negatives for log scale if needed
        data_positive = (
            data[data > 0] if np.any(data > 0) else np.abs(data[data != 0])
        )
        if len(data_positive) > 0:
            ax_hist.hist(
                data_positive,
                bins=100,
                color=color,
                edgecolor="white",
                alpha=0.8,
            )
            ax_hist.set_yscale("log")
        else:
            ax_hist.hist(
                data, bins=100, color=color, edgecolor="white", alpha=0.8
            )
        ax_hist.set_xlabel(name, fontsize=11)
        ax_hist.set_ylabel("Count (log scale)", fontsize=11)
        ax_hist.set_title(
            f"{name} Full Range (Log Scale)", fontsize=12, fontweight="bold"
        )
        ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()

    stats_out = base_dir / f"trajectory_statistics_{ds_slug}.png"
    plt.savefig(stats_out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved statistics plot to {stats_out}")

    # ===== Figure 3: Per-Clip Average Distributions =====
    fig3, axes3 = plt.subplots(2, 5, figsize=(25, 10))
    fig3.suptitle(
        "Per-Clip Average Distributions", fontsize=16, fontweight="bold"
    )

    clip_avg_features = [
        (clip_avg_speeds, "Speed", "#E67E22"),
        (clip_avg_accelerations, "Acceleration", "#27AE60"),
        (clip_avg_jerks, "Jerk", "#9B59B6"),
        (clip_avg_curvatures, "Curvature", "#F1C40F"),
        (clip_avg_velocity_y, "Velocity Y", "#3498DB"),
    ]

    for idx, (data, name, color) in enumerate(clip_avg_features):
        # Histogram of per-clip averages
        ax_avg = axes3[0, idx]
        data_p1, data_p99 = np.percentile(data, [1, 99])
        data_clipped = np.clip(data, data_p1, data_p99)
        ax_avg.hist(
            data_clipped, bins=100, color=color, edgecolor="white", alpha=0.8
        )
        ax_avg.axvline(
            np.mean(data),
            color="#E74C3C",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(data):.3f}",
        )
        ax_avg.axvline(
            np.median(data),
            color="black",
            linestyle=":",
            linewidth=2,
            label=f"Median: {np.median(data):.3f}",
        )
        ax_avg.set_xlabel(f"Avg {name}", fontsize=11)
        ax_avg.set_ylabel("Count", fontsize=11)
        ax_avg.set_title(f"Per-Clip Avg {name}", fontsize=12, fontweight="bold")
        ax_avg.legend(fontsize=8)
        ax_avg.grid(True, alpha=0.3)

    # ===== Per-Clip Maximum Distributions =====
    clip_max_features = [
        (clip_max_speeds, "Speed", "#E67E22"),
        (clip_max_accelerations, "Acceleration", "#27AE60"),
        (clip_max_jerks, "Jerk", "#9B59B6"),
        (clip_max_curvatures, "Curvature", "#F1C40F"),
        (clip_max_velocity_y, "Velocity Y", "#3498DB"),
    ]

    for idx, (data, name, color) in enumerate(clip_max_features):
        # Histogram of per-clip maximums
        ax_max = axes3[1, idx]
        data_p1, data_p99 = np.percentile(data, [1, 99])
        data_clipped = np.clip(data, data_p1, data_p99)
        ax_max.hist(
            data_clipped, bins=100, color=color, edgecolor="white", alpha=0.8
        )
        ax_max.axvline(
            np.mean(data),
            color="#E74C3C",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(data):.3f}",
        )
        ax_max.axvline(
            np.median(data),
            color="black",
            linestyle=":",
            linewidth=2,
            label=f"Median: {np.median(data):.3f}",
        )
        ax_max.set_xlabel(f"Max {name}", fontsize=11)
        ax_max.set_ylabel("Count", fontsize=11)
        ax_max.set_title(f"Per-Clip Max {name}", fontsize=12, fontweight="bold")
        ax_max.legend(fontsize=8)
        ax_max.grid(True, alpha=0.3)

    plt.tight_layout()

    per_clip_out = base_dir / f"trajectory_per_clip_stats_{ds_slug}.png"
    plt.savefig(per_clip_out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved per-clip statistics plot to {per_clip_out}")

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (All Data Points)")
    print("=" * 80)
    print(f"\nTotal clips sampled: {n_samples:,}")
    print(f"Total data points: {len(all_speeds):,}")

    print(
        f"\n{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Median':>12}"
    )
    print("-" * 80)

    for name, data in [
        ("Speed", all_speeds),
        ("Acceleration", all_accelerations),
        ("Jerk", all_jerks),
        ("Curvature", all_curvatures),
        ("Velocity X", all_velocities_x),
        ("Velocity Y", all_velocities_y),
    ]:
        print(
            f"{name:<20} {np.mean(data):>12.4f} {np.std(data):>12.4f} "
            f"{np.min(data):>12.4f} {np.max(data):>12.4f} {np.median(data):>12.4f}"
        )

    print("\n" + "=" * 80)
    print("PER-CLIP AVERAGE STATISTICS")
    print("=" * 80)
    print(
        f"\n{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Median':>12}"
    )
    print("-" * 80)

    for name, data in [
        ("Avg Speed", clip_avg_speeds),
        ("Avg Acceleration", clip_avg_accelerations),
        ("Avg Jerk", clip_avg_jerks),
        ("Avg Curvature", clip_avg_curvatures),
        ("Avg Velocity Y", clip_avg_velocity_y),
    ]:
        print(
            f"{name:<20} {np.mean(data):>12.4f} {np.std(data):>12.4f} "
            f"{np.min(data):>12.4f} {np.max(data):>12.4f} {np.median(data):>12.4f}"
        )

    print("\n" + "=" * 80)
    print("PER-CLIP MAXIMUM STATISTICS")
    print("=" * 80)
    print(
        f"\n{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Median':>12}"
    )
    print("-" * 80)

    for name, data in [
        ("Max Speed", clip_max_speeds),
        ("Max Acceleration", clip_max_accelerations),
        ("Max Jerk", clip_max_jerks),
        ("Max Curvature", clip_max_curvatures),
        ("Max Velocity Y", clip_max_velocity_y),
    ]:
        print(
            f"{name:<20} {np.mean(data):>12.4f} {np.std(data):>12.4f} "
            f"{np.min(data):>12.4f} {np.max(data):>12.4f} {np.median(data):>12.4f}"
        )

    print("\n" + "=" * 80)
    print("PERCENTILES (All Data Points)")
    print("=" * 80)
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print(f"\n{'Feature':<20}", end="")
    for p in percentiles:
        print(f"{'P'+str(p):>10}", end="")
    print()
    print("-" * 90)

    for name, data in [
        ("Speed", all_speeds),
        ("Acceleration", all_accelerations),
        ("Jerk", all_jerks),
        ("Curvature", all_curvatures),
        ("Velocity X", all_velocities_x),
        ("Velocity Y", all_velocities_y),
    ]:

        print(f"{name:<20}", end="")
        for p in percentiles:
            print(f"{np.percentile(data, p):>10.4f}", end="")
        print()

    # Persist summary statistics as JSON per dataset
    def stats_dict(arr: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    summary = {
        "dataset": dataset_name or "all",
        "n_clips_sampled": int(n_samples),
        "n_points": int(len(all_speeds)),
        "features": {
            "Speed": stats_dict(all_speeds),
            "Acceleration": stats_dict(all_accelerations),
            "Jerk": stats_dict(all_jerks),
            "Curvature": stats_dict(all_curvatures),
            "Velocity X": stats_dict(all_velocities_x),
            "Velocity Y": stats_dict(all_velocities_y),
        },
        "per_clip_avg": {
            "Speed": stats_dict(clip_avg_speeds),
            "Acceleration": stats_dict(clip_avg_accelerations),
            "Jerk": stats_dict(clip_avg_jerks),
            "Curvature": stats_dict(clip_avg_curvatures),
            "Velocity Y": stats_dict(clip_avg_velocity_y),
        },
        "per_clip_max": {
            "Speed": stats_dict(clip_max_speeds),
            "Acceleration": stats_dict(clip_max_accelerations),
            "Jerk": stats_dict(clip_max_jerks),
            "Curvature": stats_dict(clip_max_curvatures),
            "Velocity Y": stats_dict(clip_max_velocity_y),
        },
        "percentiles": {
            name: {f"p{p}": float(np.percentile(arr, p)) for p in percentiles}
            for name, arr in [
                ("Speed", all_speeds),
                ("Acceleration", all_accelerations),
                ("Jerk", all_jerks),
                ("Curvature", all_curvatures),
                ("Velocity X", all_velocities_x),
                ("Velocity Y", all_velocities_y),
            ]
        },
    }

    out_dir_path = Path(out_dir or ".")
    out_dir_path.mkdir(parents=True, exist_ok=True)
    ds_slug = _safe_slug(dataset_name or "all")
    summary_path = out_dir_path / f"trajectory_summary_{ds_slug}.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary stats to {summary_path}")
    except Exception as e:
        print(f"Failed to write summary JSON {summary_path}: {e}")

    return {
        "speeds": all_speeds,
        "accelerations": all_accelerations,
        "jerks": all_jerks,
        "curvatures": all_curvatures,
        "velocities_x": all_velocities_x,
        "velocities_y": all_velocities_y,
        "clip_avg_speeds": clip_avg_speeds,
        "clip_max_speeds": clip_max_speeds,
        "clip_avg_accelerations": clip_avg_accelerations,
        "clip_max_accelerations": clip_max_accelerations,
        "clip_avg_jerks": clip_avg_jerks,
        "clip_max_jerks": clip_max_jerks,
        "clip_avg_curvatures": clip_avg_curvatures,
        "clip_max_curvatures": clip_max_curvatures,
        "clip_avg_velocity_y": clip_avg_velocity_y,
        "clip_max_velocity_y": clip_max_velocity_y,
        "sampled_clip_ids": sampled_clip_ids,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract trajectory stats plot")
    parser.add_argument(
        "--data_database",
        default="/path/to/wheel-data/annotations_latest_schema.db",
        help="Path to database",
    )
    parser.add_argument(
        "--trajectory_data",
        default="/path/to/wheel-data/trajectory_data/",
        help="Path to trajectory data",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write PNGs (defaults to trajectory_data directory)",
    )
    parser.add_argument(
        "--samples", type=int, default=100000, help="Max sampled clips"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    t0 = time.perf_counter()
    # Parse the data to be visualized from the memory map
    clip_to_idx_path = Path(args.trajectory_data) / "clip_to_idx.json"
    with open(clip_to_idx_path, "r") as f:
        raw = json.load(f)
    clip_to_idx = {k: tuple(v) for k, v in raw.items()}

    path_to_traj_mmap = Path(args.trajectory_data) / "trajectory_data.dat"
    itemsize = np.dtype(np.float32).itemsize
    rows = os.path.getsize(path_to_traj_mmap) // (itemsize * 7)
    traj_mmap = np.memmap(
        path_to_traj_mmap, dtype=np.float32, mode="r", shape=(rows, 7)
    )
    t1 = time.perf_counter()
    print(f"Loaded trajectory index+memmap in {(t1 - t0):.3f}s")

    # Connect to data database and collect clip_ids per data source
    conn = sqlite3.connect(args.data_database)
    conn.row_factory = sqlite3.Row
    q = "SELECT clip_id, data_source FROM clips"
    ds_to_clip_ids = {}
    total_rows = 0
    t2 = time.perf_counter()
    with conn:
        for row in conn.execute(q):
            total_rows += 1
            clip_id = row["clip_id"]
            ds_field = row["data_source"] or ""
            if not ds_field:
                continue
            for token in ds_field.split(","):
                ds = token.strip()
                if ds:
                    ds_to_clip_ids.setdefault(ds, set()).add(clip_id)
    t3 = time.perf_counter()
    print(
        f"Scanned {total_rows:,} rows from DB in {(t3 - t2):.3f}s; found {len(ds_to_clip_ids)} data sources"
    )

    # Build union list of all clip_ids with trajectory data
    all_clip_ids = []
    missing = 0
    for ds, ids in ds_to_clip_ids.items():
        have = [cid for cid in ids if cid in clip_to_idx]
        miss = len(ids) - len(have)
        missing += miss
        all_clip_ids.extend(have)
        print(f"  {ds}: {len(have):,} with trajectories (+{miss:,} missing)")

    # Deduplicate while preserving order
    seen = set()
    all_clip_ids = [
        cid for cid in all_clip_ids if not (cid in seen or seen.add(cid))
    ]
    t4 = time.perf_counter()
    print(
        f"Prepared {len(all_clip_ids):,} unique clip_ids in {(t4 - t3):.3f}s; total prep {(t4 - t0):.3f}s"
    )

    # Visualize per data source and overall
    out_dir = args.output_dir or args.trajectory_data
    overall_start = time.perf_counter()
    for ds, ids in sorted(ds_to_clip_ids.items()):
        have = [cid for cid in ids if cid in clip_to_idx]
        if not have:
            continue
        ds_start = time.perf_counter()
        print(f"\nVisualizing dataset '{ds}' with {len(have):,} clips...")
        visualize_trajectories(
            have,
            clip_to_idx,
            traj_mmap,
            n_samples=args.samples,
            seed=args.seed,
            out_dir=out_dir,
            dataset_name=ds,
        )
        print(f"Done '{ds}' in {(time.perf_counter() - ds_start):.2f}s")

    print("\nVisualizing combined 'all' dataset...")
    visualize_trajectories(
        all_clip_ids,
        clip_to_idx,
        traj_mmap,
        n_samples=args.samples,
        seed=args.seed,
        out_dir=out_dir,
        dataset_name="all",
    )
    print(
        f"All visualizations completed in {(time.perf_counter() - overall_start):.2f}s"
    )
