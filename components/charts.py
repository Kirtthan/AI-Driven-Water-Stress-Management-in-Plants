# components/charts.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------- Shared helpers --------------
HOV_NUM_2DP = "%{x:.2f}"
HOV_NUM_Y = "%{y:.2f}"
HOV_PCT = "%{x:.2%}"

def _transition(fig, dur=400):
    fig.update_layout(transition=dict(duration=dur, easing="cubic-in-out"))
    return fig

# -------------- Tab 1: Prediction --------------
def class_probability_bar(classes: np.ndarray, probs: np.ndarray):
    df = pd.DataFrame({"Class": classes, "Probability": probs})
    df = df.sort_values("Probability", ascending=True)

    fig = px.bar(
        df, x="Probability", y="Class", orientation="h",
        text=df["Probability"].map(lambda v: f"{v:.2%}"),
        color="Probability",
        color_continuous_scale=[ "#ef4444", "#f59e0b", "#22c55e" ],  # danger->warning->success
        range_x=[0, 1],
    )
    fig.update_traces(hovertemplate="Class: %{y}<br>Probability: %{x:.2%}<extra></extra>", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, height=280)
    return _transition(fig, 500)

def feature_importance_bar(importance: np.ndarray, features: list[str]):
    top_idx = np.argsort(importance)[-5:]
    df = pd.DataFrame({"Feature": np.array(features)[top_idx], "Importance": importance[top_idx]})
    df = df.sort_values("Importance", ascending=True)
    fig = px.bar(df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis")
    fig.update_traces(hovertemplate="Feature: %{y}<br>Importance: %{x:.3f}<extra></extra>")
    fig.update_layout(coloraxis_showscale=False, height=220)
    return _transition(fig, 400)

def confidence_gauge(confidence_pct: float):
    # confidence_pct expected 0..100
    v = confidence_pct
    bar_color = "#22c55e" if v >= 80 else ("#f59e0b" if v >= 60 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={'suffix': "%", 'valueformat': ".1f"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 60], "color": "rgba(239,68,68,0.25)"},
                {"range": [60, 80], "color": "rgba(245,158,11,0.25)"},
                {"range": [80, 100], "color": "rgba(34,197,94,0.25)"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    fig.update_layout(height=220, margin=dict(l=16, r=16, t=16, b=8))
    return _transition(fig, 300)

# -------------- Tab 2: Model Comparison --------------
def radar_metrics(metrics_dict: dict[str, float]):
    axes = ["accuracy", "precision", "recall", "f1_score"]
    r = [metrics_dict[a] for a in axes]
    fig = go.Figure(go.Scatterpolar(
        r=r, theta=[a.title().replace("_", " ") for a in axes],
        fill="toself", line=dict(width=2)
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=420, showlegend=False)
    return _transition(fig, 400)

def confusion_matrix_animated(cm: np.ndarray, classes: list[str]):
    # staged reveal via frame opacity on annotations
    z = cm.astype(int)
    fig = px.imshow(z, x=classes, y=classes, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(height=420)

    # Add text annotations with zero opacity first, then animate to 1.0
    annotations = []
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            annotations.append(dict(x=classes[j], y=classes[i], text=str(z[i, j]),
                                    showarrow=False, font=dict(size=14, color="#0f172a"), opacity=0.0))
    fig.update_layout(annotations=annotations)

    # Create a single "reveal" frame
    reveal_ann = []
    for a in annotations:
        ra = a.copy(); ra["opacity"] = 1.0
        reveal_ann.append(ra)
    fig.frames = [go.Frame(layout=go.Layout(annotations=reveal_ann))]

    fig.update_layout(updatemenus=[dict(type="buttons", showactive=False,
        buttons=[dict(label="Reveal", method="animate", args=[[0], {"transition": {"duration": 0}, "frame": {"duration": 600}}])],
        x=1.0, xanchor="right", y=1.15)])
    return fig

# -------------- Tab 3: Data Visualization --------------
def hist_with_box(df: pd.DataFrame, feature: str, class_col: str):
    hist = px.histogram(df, x=feature, color=class_col, barmode="overlay", opacity=0.7,
                        marginal="box")
    hist.update_traces(hovertemplate=f"{feature}: %{ { 'x' } }<br>{class_col}: %{ { 'legendgroup' } }<extra></extra>")
    hist.update_layout(height=420)
    return _transition(hist, 350)

def box_by_class(df: pd.DataFrame, feature: str, class_col: str):
    box = px.box(df, x=class_col, y=feature, color=class_col)
    box.update_traces(
        hovertemplate=f"{class_col}: %{{x}}<br>{feature}: %{{y:.2f}}<extra></extra>"
    )

    box.update_layout(height=420, showlegend=False)
    return _transition(box, 350)

def correlation_heatmap(df_numeric: pd.DataFrame):
    corr = df_numeric.corr(numeric_only=True)
    heat = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     labels=dict(color="Correlation"), aspect="auto")
    heat.update_traces(hovertemplate="x: %{x}<br>y: %{y}<br>ρ: %{z:.2f}<extra></extra>")
    heat.update_layout(height=640)
    return heat

def class_distribution_donut(series: pd.Series):
    vc = series.value_counts()
    donut = px.pie(values=vc.values, names=vc.index, hole=0.5)
    donut.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{percent} (%{value})<extra></extra>")
    donut.update_layout(height=420, annotations=[dict(text=f"N={vc.sum()}", x=0.5, y=0.5, font_size=14, showarrow=False)])
    return donut
