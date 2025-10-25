import numpy as np
import plotly.graph_objects as go
import torch
from typing import List


def visualize_decoding(
    log_probs: torch.Tensor,
    beam_history: List[List[dict]],
    itos: List[str],
    prob_threshold: float = 1e-4,
):
    T, C = log_probs.shape
    probs = log_probs.detach().cpu().exp().numpy().T  # (C, T)

    # === фильтруем редко активные классы ===
    active_mask = probs.max(axis=1) > prob_threshold
    probs = probs[active_mask]
    y_labels = [itos[i] if i < len(itos) else str(i) for i in range(C)]
    y_labels = [lbl for lbl, m in zip(y_labels, active_mask) if m]
    active_indices = np.nonzero(active_mask)[0]

    fig = go.Figure()

    # === фон: тепловая карта вероятностей ===
    fig.add_trace(
        go.Heatmap(
            z=probs,
            x=list(range(T)),
            y=y_labels,
            colorscale=[[0.0, "rgb(255,255,255)"], [1.0, "rgb(0,0,0)"]],
            showscale=True,
            colorbar=dict(title="Probability"),
            hoverinfo="x+y+z",
            zmin=0.0,
            zmax=1.0,
        )
    )

    colors = [
        "red",
        "orange",
        "limegreen",
        "royalblue",
        "magenta",
        "gold",
        "cyan",
        "purple",
        "brown",
        "pink",
        "olive",
        "teal",
        "navy",
        "darkred",
        "gray",
    ]

    # === определяем финальные выжившие последовательности ===
    final_step = beam_history[-1] if beam_history else []
    final_kept_dict = {}
    for h in final_step:
        if not h["kept"]:
            continue
        seq_key = tuple(h["seq"])
        if (
            seq_key not in final_kept_dict
            or h["score"] > final_kept_dict[seq_key]["score"]
        ):
            final_kept_dict[seq_key] = h
    final_kept = set(final_kept_dict.keys())

    shown_legend = set()
    kept_count = 0

    # === рисуем все траектории ===
    for t, step_data in enumerate(beam_history):
        for h in step_data:
            seq, score, kept = h["seq"], h["score"], h["kept"]
            if len(seq) < 2:
                continue

            xs, ys = [], []
            for i, token in enumerate(seq):
                if token >= C or token not in active_indices:
                    continue
                xs.append(i)
                ys.append(y_labels[list(active_indices).index(token)])

            if not xs:
                continue

            decoded = "".join(
                [
                    itos[i]
                    for i in seq
                    if 0 <= i < len(itos) and not itos[i].startswith("<")
                ]
            )

            seq_key = tuple(seq)
            is_final = kept and seq_key in final_kept

            if is_final and seq_key not in shown_legend:
                label = f"'{decoded or '?'}' ({score:.2f})"
                show_in_legend = True
                shown_legend.add(seq_key)
            else:
                label = f"_hidden_{t}_{len(seq)}"
                show_in_legend = False

            # === цвет и стиль ===
            if is_final:
                color = colors[kept_count % len(colors)]
                kept_count += 1
                line_style = dict(color=color, width=3)
                marker_style = dict(size=5)
                opacity = 1.0
            elif kept:
                color = "gray"
                line_style = dict(color=color, width=1, dash="dot")
                marker_style = dict(size=3)
                opacity = 0.3
            else:
                color = "lightgray"
                line_style = dict(color=color, width=1, dash="dot")
                marker_style = dict(size=2)
                opacity = 0.15

            # === добавляем трассу ===
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    line=line_style,
                    marker=marker_style,
                    name=label,
                    opacity=opacity,
                    hoverinfo="x+y+name",
                    showlegend=show_in_legend,
                    legendgroup=str(seq_key),
                )
            )

    # === оформление ===
    fig.update_layout(
        title=dict(text="Beam Search — Final vs Discarded Hypotheses", x=0.5),
        xaxis=dict(title="Time step (t)", tickfont=dict(size=12)),
        yaxis=dict(title="Character", tickfont=dict(size=12), automargin=True),
        width=1500,
        height=1000,
        plot_bgcolor="white",
        legend=dict(
            title="Final Hypotheses",
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=120, r=60, t=80, b=150),
    )

    fig.show()
