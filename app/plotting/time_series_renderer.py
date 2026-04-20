import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def _strip_arm_prefix(name: str) -> str:
    lower = str(name).lower().strip()
    prefixes = (
        "left_", "right_", "l_", "r_",
        "left ", "right ", "l ", "r ",
    )
    for prefix in prefixes:
        if lower.startswith(prefix):
            return lower[len(prefix):]
    return lower


def _resolve_feature_names(feature_names, num_channels):
    names = list(feature_names)

    if len(names) == num_channels:
        return names

    if len(names) * 2 == num_channels:
        left = [f"LEFT {name}" for name in names]
        right = [f"RIGHT {name}" for name in names]
        return left + right

    if len(names) < num_channels:
        padded = names + [f"feature_{i}" for i in range(len(names), num_channels)]
        return padded

    return names[:num_channels]


def compute_grouped_ylims(feature_time, feature_names):
    ylims = {}
    lower_names = [_strip_arm_prefix(name) for name in feature_names]

    acc_idx = [
        i for i, name in enumerate(lower_names)
        if name in {"axis_x_acc", "axis_y_acc", "axis_z_acc"}
    ]

    if acc_idx:
        vals = feature_time[acc_idx, :]
        ymin = float(np.min(vals))
        ymax = float(np.max(vals))
        pad = 0.05 * max(ymax - ymin, 1e-8)
        shared = (ymin - pad, ymax + pad)
        for i in acc_idx:
            ylims[i] = shared

    for i in range(len(feature_names)):
        if i in ylims:
            continue
        vals = feature_time[i]
        ymin = float(np.min(vals))
        ymax = float(np.max(vals))
        pad = 0.05 * max(ymax - ymin, 1e-8)
        if ymax - ymin < 1e-8:
            pad = 0.1 if abs(ymax) < 1e-8 else 0.05 * abs(ymax)
        ylims[i] = (ymin - pad, ymax + pad)

    return ylims


def _render_single_panel(
    fig,
    spec,
    panel_title,
    feature_indices,
    feature_time,
    feature_scores,
    feature_heatmaps,
    feature_names,
    global_attribution,
    feature_cmap,
    heatmap_cmap,
    global_cmap,
    left_label,
    heatmap_label,
    global_title,
):
    n_features = len(feature_indices)

    panel = spec.subgridspec(
        nrows=n_features + 1,
        ncols=4,
        width_ratios=[1.8, 0.55, 23.0, 0.8],
        height_ratios=[1.0] * n_features + [0.7],
        hspace=0.18,
        wspace=0.035,
    )

    ylims = compute_grouped_ylims(feature_time[feature_indices], [feature_names[i] for i in feature_indices])

    prev_ax_ts = None
    main_im = None

    for local_i, feat_i in enumerate(feature_indices):
        ax_name = fig.add_subplot(panel[local_i, 0])
        ax_name.axis("off")
        ax_name.text(0.98, 0.5, feature_names[feat_i], ha="right", va="center", fontsize=11)

        ax_score = fig.add_subplot(panel[local_i, 1])
        score_img = np.array([[feature_scores[feat_i]]], dtype=np.float32)
        ax_score.imshow(
            score_img,
            cmap=feature_cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            interpolation="nearest",
        )
        ax_score.set_xticks([])
        ax_score.set_yticks([])
        for spine in ax_score.spines.values():
            spine.set_visible(False)

        if local_i == 0:
            ax_name.text(
                0.0,
                1.35,
                panel_title,
                transform=ax_name.transAxes,
                ha="left",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
            ax_name.text(
                0.98,
                1.10,
                left_label,
                transform=ax_name.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        sub = panel[local_i, 2].subgridspec(2, 1, height_ratios=[5.0, 1.2], hspace=0.02)
        ax_ts = fig.add_subplot(sub[0], sharex=prev_ax_ts)
        ax_hm = fig.add_subplot(sub[1], sharex=ax_ts)

        ax_ts.plot(feature_time[feat_i], linewidth=1.9)
        ax_ts.set_ylim(*ylims[local_i])
        ax_ts.grid(True, alpha=0.28)
        ax_ts.margins(x=0, y=0.02)
        ax_ts.tick_params(axis="x", labelbottom=False)

        band_hm = np.repeat(feature_heatmaps[feat_i][None, :], 12, axis=0)
        main_im = ax_hm.imshow(
            band_hm,
            aspect="auto",
            cmap=heatmap_cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            origin="lower",
        )
        ax_hm.set_yticks([])
        ax_hm.tick_params(axis="x", labelbottom=False)
        ax_hm.margins(x=0, y=0)
        for spine in ax_hm.spines.values():
            spine.set_visible(False)

        if local_i == 0:
            ax_hm.text(
                0.0,
                1.35,
                heatmap_label,
                transform=ax_hm.transAxes,
                ha="left",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        prev_ax_ts = ax_ts

    global_row = n_features
    ax_global_name = fig.add_subplot(panel[global_row, 0:2])
    ax_global_name.axis("off")
    ax_global_name.text(0.98, 0.5, global_title, ha="right", va="center", fontsize=11, fontweight="bold")

    ax_global = fig.add_subplot(panel[global_row, 2], sharex=prev_ax_ts)
    band_global = np.repeat(global_attribution[None, :], 10, axis=0)
    ax_global.imshow(
        band_global,
        aspect="auto",
        cmap=global_cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        origin="lower",
    )
    ax_global.set_yticks([])
    ax_global.set_xlabel("Time step")
    ax_global.margins(x=0)
    for spine in ax_global.spines.values():
        spine.set_visible(False)

    cbar_ax = fig.add_subplot(panel[:n_features, 3])
    cbar = fig.colorbar(main_im, cax=cbar_ax)
    return cbar


def render_time_series_base(
    result,
    feature_cmap="coolwarm",
    heatmap_cmap="coolwarm",
    global_cmap="coolwarm",
    feature_label="Attribution strength (normalized)",
    left_label="Feature Occlusion",
    heatmap_label="Integrated Gradients",
    global_title="GradCAM",
    figsize_per_feature=3.15,
):
    feature_time = np.asarray(result.feature_time, dtype=np.float32)
    feature_scores = np.asarray(result.feature_occlusion_scores, dtype=np.float32)
    feature_heatmaps = np.asarray(result.ig_heatmaps, dtype=np.float32)
    global_attribution = np.asarray(result.gradcam_map, dtype=np.float32)

    feature_names = result.feature_names
    class_names = result.class_names
    true_label = result.true_idx
    pred_class = result.pred_idx
    confidence = result.confidence

    feature_names = _resolve_feature_names(feature_names, feature_time.shape[0])
    num_features = min(feature_time.shape[0], len(feature_names), feature_scores.shape[0], feature_heatmaps.shape[0])
    feature_time = feature_time[:num_features]
    feature_scores = feature_scores[:num_features]
    feature_heatmaps = feature_heatmaps[:num_features]
    feature_names = feature_names[:num_features]

    pred_text = class_names[pred_class] if pred_class is not None and pred_class < len(class_names) else pred_class
    true_text = class_names[true_label] if true_label is not None and true_label < len(class_names) else true_label

    is_dual_arm = num_features % 2 == 0 and num_features >= 2

    if is_dual_arm:
        half = num_features // 2
        fig = plt.figure(
            figsize=(30, figsize_per_feature * half + 1.5),
            constrained_layout=True,
        )

        outer = gridspec.GridSpec(
            nrows=1,
            ncols=2,
            figure=fig,
            wspace=0.12,
        )

        left_indices = list(range(0, half))
        right_indices = list(range(half, num_features))

        cbar_left = _render_single_panel(
            fig=fig,
            spec=outer[0, 0],
            panel_title="LEFT",
            feature_indices=left_indices,
            feature_time=feature_time,
            feature_scores=feature_scores,
            feature_heatmaps=feature_heatmaps,
            feature_names=feature_names,
            global_attribution=global_attribution,
            feature_cmap=feature_cmap,
            heatmap_cmap=heatmap_cmap,
            global_cmap=global_cmap,
            left_label=left_label,
            heatmap_label=heatmap_label,
            global_title=global_title,
        )
        cbar_left.set_label(feature_label)

        cbar_right = _render_single_panel(
            fig=fig,
            spec=outer[0, 1],
            panel_title="RIGHT",
            feature_indices=right_indices,
            feature_time=feature_time,
            feature_scores=feature_scores,
            feature_heatmaps=feature_heatmaps,
            feature_names=feature_names,
            global_attribution=global_attribution,
            feature_cmap=feature_cmap,
            heatmap_cmap=heatmap_cmap,
            global_cmap=global_cmap,
            left_label=left_label,
            heatmap_label=heatmap_label,
            global_title=global_title,
        )
        cbar_right.set_label(feature_label)

    else:
        fig = plt.figure(
            figsize=(16, figsize_per_feature * num_features + 1.2),
            constrained_layout=True,
        )

        outer = gridspec.GridSpec(
            nrows=1,
            ncols=1,
            figure=fig,
        )

        cbar = _render_single_panel(
            fig=fig,
            spec=outer[0, 0],
            panel_title="",
            feature_indices=list(range(num_features)),
            feature_time=feature_time,
            feature_scores=feature_scores,
            feature_heatmaps=feature_heatmaps,
            feature_names=feature_names,
            global_attribution=global_attribution,
            feature_cmap=feature_cmap,
            heatmap_cmap=heatmap_cmap,
            global_cmap=global_cmap,
            left_label=left_label,
            heatmap_label=heatmap_label,
            global_title=global_title,
        )
        cbar.set_label(feature_label)

    title_parts = []
    if true_text is not None:
        title_parts.append(f"True: {true_text}")
    if pred_text is not None:
        title_parts.append(f"Pred: {pred_text}")
    if confidence is not None:
        title_parts.append(f"Conf: {confidence:.3f}")

    fig.suptitle(" | ".join(title_parts), fontsize=13)
    return fig


def render_combined_explanation(result):
    return render_time_series_base(
        result,
        left_label="Feature Occlusion",
        heatmap_label="Integrated Gradients",
        global_title="GradCAM",
        feature_label="Attribution strength (normalized)",
    )


def render_sliding_window_explanation(result):
    return render_time_series_base(
        result,
        left_label="Mean window occlusion",
        heatmap_label="Sliding Window Occlusion",
        global_title="Window overlap",
        feature_label="Attribution strength (normalized)",
    )


def render_analysis_result(result):
    mode = getattr(result, "mode", "combined")

    if mode == "combined":
        return render_combined_explanation(result)

    if mode == "sliding_window":
        return render_sliding_window_explanation(result)

    raise ValueError(f"Unsupported result.mode: {mode}")