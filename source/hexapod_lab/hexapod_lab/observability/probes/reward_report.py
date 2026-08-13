"""Post-hoc report: reward composition and training diagnostics as PNGs.

The lesson that keeps repeating on this project is that **the total reward hides
what matters**. The first M2 smoke run had a rising reward curve and looked fine
in aggregate, while the gait terms summed negative and the cheapest way to
satisfy them was to stop stepping. You only see that by reading the per-term
breakdown -- which, until now, meant grepping a text log for one iteration at a
time and holding the trend in your head.

This probe turns that into artifacts:

* ``overview.png``       -- reward, episode length, throughput, VRAM
* ``reward_terms.png``   -- every reward term over the run, small multiples
* ``reward_balance.png`` -- positive terms stacked up, penalties stacked down,
                            net on top: what is actually driving the policy
* ``ppo_diagnostics.png``-- losses, LR, action noise, tracking error
* ``summary.md``         -- metadata plus first/last values and the terms that
                            dominate at the end of the run

Everything is read back out of the TensorBoard event files, so it works on any
run directory -- including ones recorded before this module existed (the ANYmal
baseline included) via ``scripts/report_run.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in headless training runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

from ..context import RunContext  # noqa: E402
from ..probe import Probe  # noqa: E402

REWARD_PREFIX = "Episode_Reward/"


def load_scalars(log_dir: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Merge scalars from every event file in ``log_dir`` into {tag: (steps, values)}.

    Each file is read separately and merged by hand rather than pointing one
    accumulator at the directory: this run dir deliberately holds more than one
    event file (rsl-rl's, plus the VRAM probe's), and relying on the
    accumulator's directory handling to pick up both is a silent-failure risk.
    """
    log_dir = Path(log_dir)
    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for event_file in sorted(log_dir.glob("events.out.tfevents*")):
        acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            events = acc.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            if tag in merged:  # same tag in two files: concatenate and re-sort
                prev_steps, prev_values = merged[tag]
                steps = np.concatenate([prev_steps, steps])
                values = np.concatenate([prev_values, values])
                order = np.argsort(steps)
                steps, values = steps[order], values[order]
            merged[tag] = (steps, values)
    return merged


def _plot(ax, scalars, tag, title, color=None, ylabel=None):
    """Draw one tag, or an 'unavailable' placeholder so layouts stay stable."""
    if tag not in scalars:
        ax.text(0.5, 0.5, f"{tag}\nnot logged", ha="center", va="center", fontsize=8, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    steps, values = scalars[tag]
    ax.plot(steps, values, color=color, linewidth=1.4)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def write_overview(scalars, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    _plot(axes[0, 0], scalars, "Train/mean_reward", "Mean reward", "#1f77b4")
    _plot(axes[0, 1], scalars, "Train/mean_episode_length", "Mean episode length", "#2ca02c")
    _plot(axes[1, 0], scalars, "Perf/total_fps", "Throughput (steps/s)", "#ff7f0e")
    if "GPU/vram_used_gb" in scalars:
        steps, values = scalars["GPU/vram_used_gb"]
        axes[1, 1].plot(steps, values, color="#d62728", linewidth=1.4)
        axes[1, 1].set_title("VRAM used (GB) -- x is sample index, not iteration", fontsize=10)
        axes[1, 1].grid(alpha=0.25)
        if "GPU/vram_free_gb" in scalars:
            total = values[0] + scalars["GPU/vram_free_gb"][1][0]
            axes[1, 1].axhline(total, color="k", linestyle="--", linewidth=1, label=f"total {total:.1f} GB")
            axes[1, 1].legend(fontsize=8)
    else:
        _plot(axes[1, 1], scalars, "GPU/vram_used_gb", "VRAM used (GB)")
    for ax in axes.flat:
        ax.set_xlabel("iteration", fontsize=8)
    fig.suptitle("Training overview", fontsize=12)
    fig.tight_layout()
    path = out / "overview.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_reward_terms(scalars, out: Path) -> Path | None:
    tags = sorted(t for t in scalars if t.startswith(REWARD_PREFIX))
    if not tags:
        return None
    cols = 4
    rows = int(np.ceil(len(tags) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.2 * rows), squeeze=False)
    for ax, tag in zip(axes.flat, tags):
        steps, values = scalars[tag]
        positive = float(np.mean(values)) >= 0
        ax.plot(steps, values, color="#2ca02c" if positive else "#d62728", linewidth=1.3)
        ax.axhline(0, color="k", linewidth=0.6, alpha=0.4)
        ax.set_title(tag[len(REWARD_PREFIX) :], fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(tags) :]:
        ax.axis("off")
    fig.suptitle("Reward terms over training (green = reward, red = penalty)", fontsize=12)
    fig.tight_layout()
    path = out / "reward_terms.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_reward_balance(scalars, out: Path) -> Path | None:
    """Rewards stacked up, penalties stacked down, net line on top.

    This is the plot that answers 'what is the policy actually optimizing' at a
    glance -- and the one that would have caught the air-time/variance imbalance
    in the first smoke run without reading a single number.
    """
    tags = sorted(t for t in scalars if t.startswith(REWARD_PREFIX))
    if not tags:
        return None

    # resample everything onto the shared iteration axis of the longest series
    ref_steps = max((scalars[t][0] for t in tags), key=len)
    series = {}
    for tag in tags:
        steps, values = scalars[tag]
        series[tag] = np.interp(ref_steps, steps, values) if len(steps) != len(ref_steps) else values

    positives = [t for t in tags if np.mean(series[t]) >= 0]
    negatives = [t for t in tags if np.mean(series[t]) < 0]

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab20")
    if positives:
        ax.stackplot(
            ref_steps,
            *[np.clip(series[t], 0, None) for t in positives],
            labels=[t[len(REWARD_PREFIX) :] for t in positives],
            colors=[cmap(i % 20) for i in range(len(positives))],
            alpha=0.85,
        )
    if negatives:
        ax.stackplot(
            ref_steps,
            *[np.clip(series[t], None, 0) for t in negatives],
            labels=[t[len(REWARD_PREFIX) :] for t in negatives],
            colors=[cmap((i + len(positives)) % 20) for i in range(len(negatives))],
            alpha=0.85,
        )
    net = np.sum([series[t] for t in tags], axis=0)
    ax.plot(ref_steps, net, color="k", linewidth=2.0, label="net (sum of terms)")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("per-episode reward contribution")
    ax.set_title("Reward composition: what is driving the policy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    path = out / "reward_balance.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_ppo_diagnostics(scalars, out: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    _plot(axes[0, 0], scalars, "Loss/surrogate", "Surrogate loss", "#1f77b4")
    _plot(axes[0, 1], scalars, "Loss/value", "Value loss", "#ff7f0e")
    _plot(axes[0, 2], scalars, "Loss/entropy", "Entropy", "#9467bd")
    _plot(axes[1, 0], scalars, "Loss/learning_rate", "Learning rate (adaptive)", "#8c564b")
    _plot(axes[1, 1], scalars, "Policy/mean_std", "Action noise std", "#e377c2")
    _plot(axes[1, 2], scalars, "Metrics/base_velocity/error_vel_xy", "Velocity tracking error (m/s)", "#17becf")
    for ax in axes.flat:
        ax.set_xlabel("iteration", fontsize=8)
    fig.suptitle("PPO diagnostics", fontsize=12)
    fig.tight_layout()
    path = out / "ppo_diagnostics.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_summary(scalars, ctx: RunContext, out: Path) -> Path:
    """Markdown digest: the numbers you would otherwise grep for."""
    lines = [f"# Run report: {Path(ctx.log_dir).name}", ""]
    lines += [
        f"- task: `{ctx.task}`",
        f"- envs: {ctx.num_envs}, max iterations: {ctx.max_iterations}, device: `{ctx.device}`",
    ]
    for key, value in ctx.results.items():
        lines.append(f"- {key}: {value}")
    machine = ctx.describe_machine()
    lines += ["", "## Machine", ""] + [f"- {k}: {v}" for k, v in machine.items()]

    def endpoints(tag: str) -> str:
        if tag not in scalars:
            return "n/a"
        _, values = scalars[tag]
        return f"{values[0]:.4g} -> {values[-1]:.4g} (min {values.min():.4g}, max {values.max():.4g})"

    lines += ["", "## Headline scalars", ""]
    for tag in (
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Perf/total_fps",
        "Metrics/base_velocity/error_vel_xy",
        "Metrics/base_velocity/error_vel_yaw",
        "Policy/mean_std",
    ):
        lines.append(f"- `{tag}`: {endpoints(tag)}")

    reward_tags = sorted(t for t in scalars if t.startswith(REWARD_PREFIX))
    if reward_tags:
        final = {t: float(scalars[t][1][-1]) for t in reward_tags}
        ranked = sorted(final.items(), key=lambda kv: -abs(kv[1]))
        lines += ["", "## Reward terms at end of run (by magnitude)", ""]
        lines += ["| term | final value |", "| --- | --- |"]
        for tag, value in ranked:
            lines.append(f"| {tag[len(REWARD_PREFIX):]} | {value:+.4f} |")
        total_pos = sum(v for v in final.values() if v > 0)
        total_neg = sum(v for v in final.values() if v < 0)
        lines += [
            "",
            f"Rewards total **{total_pos:+.3f}**, penalties total **{total_neg:+.3f}**, "
            f"net **{total_pos + total_neg:+.3f}**.",
        ]

    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_report(log_dir: str | Path, ctx: RunContext | None = None) -> list[Path]:
    """Generate every artifact for a run directory. Safe to call on old runs."""
    log_dir = Path(log_dir)
    ctx = ctx or RunContext(log_dir=str(log_dir))
    scalars = load_scalars(log_dir)
    if not scalars:
        print(f"[report] no scalars found in {log_dir}")
        return []
    out = ctx.reports_dir
    written = [p for p in (
        write_overview(scalars, out),
        write_reward_terms(scalars, out),
        write_reward_balance(scalars, out),
        write_ppo_diagnostics(scalars, out),
        write_summary(scalars, ctx, out),
    ) if p is not None]
    (out / "run_summary.json").write_text(
        json.dumps({"context": ctx.__dict__ | {"machine": ctx.describe_machine()}}, indent=2, default=str),
        encoding="utf-8",
    )
    return written


class RewardReportProbe(Probe):
    """Render the report once training finishes."""

    name = "reward_report"
    stop_order = 20  # after live probes have closed their event files

    def stop(self, ctx: RunContext, failed: bool = False) -> None:
        written = build_report(ctx.log_dir, ctx)
        for path in written:
            print(f"[report] wrote {path}")
        ctx.results["report_files"] = [p.name for p in written]
