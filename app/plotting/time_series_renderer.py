import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def compute_grouped_ylims(feature_time, feature_names):
    ylims = {}
    lower_names = [str(name).lower() for name in feature_names]

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

    num_features = min(feature_time.shape[0], len(feature_names))
    feature_time = feature_time[:num_features]
    feature_scores = feature_scores[:num_features]
    feature_heatmaps = feature_heatmaps[:num_features]

    ylims = compute_grouped_ylims(feature_time, feature_names[:num_features])

    pred_text = class_names[pred_class] if pred_class is not None and pred_class < len(class_names) else pred_class
    true_text = class_names[true_label] if true_label is not None and true_label < len(class_names) else true_label

    n_outer_rows = num_features + 1

    fig = plt.figure(
        figsize=(16, figsize_per_feature * num_features + 1.2),
        constrained_layout=True
    )

    outer = gridspec.GridSpec(
        nrows=n_outer_rows,
        ncols=4,
        figure=fig,
        width_ratios=[3.0, 1.0, 18.0, 0.9],
        height_ratios=[1.0] * num_features + [0.7],
        hspace=0.18,
        wspace=0.08,
    )

    prev_ax_ts = None
    main_im = None

    for i in range(num_features):
        # feature name
        ax_name = fig.add_subplot(outer[i, 0])
        ax_name.axis("off")
        ax_name.text(
            0.98, 0.5, feature_names[i],
            ha="right", va="center", fontsize=11
        )

        # left feature score block
        ax_score = fig.add_subplot(outer[i, 1])
        score_img = np.array([[feature_scores[i]]], dtype=np.float32)
        ax_score.imshow(
            score_img,
            cmap=feature_cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            interpolation="nearest"
        )
        ax_score.set_xticks([])
        ax_score.set_yticks([])
        for spine in ax_score.spines.values():
            spine.set_visible(False)

        if i == 0:
            ax_name.text(
                0.98, 1.10, left_label,
                transform=ax_name.transAxes,
                ha="right", va="bottom",
                fontsize=10, fontweight="bold"
            )

        # signal + per-feature heatmap
        sub = outer[i, 2].subgridspec(2, 1, height_ratios=[5.0, 1.2], hspace=0.02)
        ax_ts = fig.add_subplot(sub[0], sharex=prev_ax_ts)
        ax_hm = fig.add_subplot(sub[1], sharex=ax_ts)

        ax_ts.plot(feature_time[i], linewidth=1.9)
        ax_ts.set_ylim(*ylims[i])
        ax_ts.grid(True, alpha=0.28)
        ax_ts.margins(x=0, y=0.02)
        ax_ts.tick_params(axis="x", labelbottom=False)

        band_hm = np.repeat(feature_heatmaps[i][None, :], 12, axis=0)
        main_im = ax_hm.imshow(
            band_hm,
            aspect="auto",
            cmap=heatmap_cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            origin="lower"
        )
        ax_hm.set_yticks([])
        ax_hm.tick_params(axis="x", labelbottom=False)
        ax_hm.margins(x=0, y=0)
        for spine in ax_hm.spines.values():
            spine.set_visible(False)

        if i == 0:
            ax_hm.text(
                0.0, 1.35, heatmap_label,
                transform=ax_hm.transAxes,
                ha="left", va="bottom",
                fontsize=10, fontweight="bold"
            )

        prev_ax_ts = ax_ts

    # global strip
    row = num_features

    ax_global_name = fig.add_subplot(outer[row, 0:2])
    ax_global_name.axis("off")
    ax_global_name.text(
        0.98, 0.5, global_title,
        ha="right", va="center", fontsize=11, fontweight="bold"
    )

    ax_global = fig.add_subplot(outer[row, 2], sharex=prev_ax_ts)
    band_global = np.repeat(global_attribution[None, :], 10, axis=0)
    ax_global.imshow(
        band_global,
        aspect="auto",
        cmap=global_cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        origin="lower"
    )
    ax_global.set_yticks([])
    ax_global.set_xlabel("Time step")
    ax_global.margins(x=0)
    for spine in ax_global.spines.values():
        spine.set_visible(False)

    # colorbar
    cbar_ax = fig.add_subplot(outer[:num_features, 3])
    cbar = fig.colorbar(main_im, cax=cbar_ax)
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