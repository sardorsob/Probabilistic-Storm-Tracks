import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import DashboardBundle
from app.pages.overview import MODEL_COLORS, MODEL_LABELS

OBSERVED_COLOR = "#173B57"


def _map_view(points: pd.DataFrame) -> tuple[dict[str, float], float]:
    latitude = points["latitude"]
    longitude = points["longitude"]
    radians = longitude.map(math.radians)
    center_lon = math.degrees(
        math.atan2(radians.map(math.sin).mean(), radians.map(math.cos).mean())
    )
    relative_lon = (longitude - center_lon + 180) % 360 - 180
    center = {
        "lat": float((latitude.min() + latitude.max()) / 2),
        "lon": float(center_lon),
    }
    latitude_span = float(latitude.max() - latitude.min())
    longitude_span = float(relative_lon.max() - relative_lon.min())
    span = max(
        latitude_span,
        longitude_span * abs(math.cos(math.radians(center["lat"]))),
        0.5,
    )
    zoom = min(8.0, max(0.0, math.log2(360 / (span * 1.4)) - 1))
    return center, zoom


def build_track_figure(points: pd.DataFrame) -> go.Figure:
    ordered = points.sort_values("time_utc", kind="stable").copy()
    ordered["time_label"] = ordered["time_utc"].dt.strftime("%Y-%m-%d %H:%M")
    ordered["map_segment"] = ordered["longitude"].diff().abs().gt(180).cumsum()
    figure = go.Figure()
    for segment_index, (_, segment) in enumerate(ordered.groupby("map_segment", sort=False)):
        figure.add_trace(
            go.Scattermap(
                lat=segment["latitude"],
                lon=segment["longitude"],
                mode="lines+markers",
                line={"color": OBSERVED_COLOR, "width": 4},
                marker={"color": OBSERVED_COLOR, "size": 8},
                customdata=segment[["time_label"]],
                hovertemplate=(
                    "UTC time %{customdata[0]}<br>Latitude %{lat:.2f}°<br>"
                    "Longitude %{lon:.2f}°<extra></extra>"
                ),
                name="Observed track",
                showlegend=segment_index == 0,
            )
        )
    endpoints = [
        (ordered.iloc[0], "Start", "#64748B"),
        (ordered.iloc[-1], "End", "#172033"),
    ]
    for point, name, color in endpoints:
        figure.add_trace(
            go.Scattermap(
                lat=[point["latitude"]],
                lon=[point["longitude"]],
                mode="markers+text",
                marker={"color": color, "size": 20},
                text=[name[0]],
                textfont={"color": "white", "size": 10},
                textposition="middle center",
                hovertemplate=f"{name}<br>UTC time {point['time_label']}<extra></extra>",
                name=name,
            )
        )
    center, zoom = _map_view(ordered)
    figure.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 12, "b": 0},
        legend={"orientation": "h", "y": 1.05, "x": 0},
        map={"style": "white-bg", "center": center, "zoom": zoom},
    )
    return figure


def _build_storm_error_figure(performance: pd.DataFrame) -> go.Figure:
    chart_data = performance.assign(
        model_label=performance["model"].map(MODEL_LABELS).fillna(performance["model"])
    ).sort_values(["model_label", "forecast_horizon_hours"])
    figure = px.line(
        chart_data,
        x="forecast_horizon_hours",
        y="rmse_km",
        color="model_label",
        markers=True,
        color_discrete_map={MODEL_LABELS[key]: value for key, value in MODEL_COLORS.items()},
        labels={
            "forecast_horizon_hours": "Forecast horizon (hours)",
            "rmse_km": "RMSE (km)",
            "model_label": "Model",
        },
    )
    figure.update_traces(line={"width": 3}, marker={"size": 8})
    figure.update_layout(
        template="plotly_white",
        height=350,
        margin={"l": 12, "r": 12, "t": 20, "b": 12},
        legend={"orientation": "h", "y": 1.15, "x": 0},
    )
    figure.update_xaxes(gridcolor="#E7EBEF")
    figure.update_yaxes(rangemode="tozero", gridcolor="#E7EBEF")
    return figure


def render(data: DashboardBundle) -> None:
    st.title("Storm explorer")
    if data.tracks is None or data.metadata is None:
        st.info(
            "Observed track points are not included in this dashboard build. "
            "Rebuild the data contract with the optional tracks input to enable the map."
        )
        return

    splits = sorted(data.tracks["split"].dropna().astype(str).unique())
    default_split = splits.index(data.headline_split) if data.headline_split in splits else 0
    split_column, storm_column = st.columns([1, 2])
    with split_column:
        split = st.selectbox("Evaluation split", splits, index=default_split)
    storm_ids = sorted(
        data.tracks.loc[data.tracks["split"] == split, "storm_id"].dropna().astype(str).unique()
    )
    with storm_column:
        storm_id = st.selectbox("Storm", storm_ids)

    points = data.tracks.loc[
        (data.tracks["split"] == split) & (data.tracks["storm_id"].astype(str) == storm_id)
    ]
    metadata = data.metadata.loc[
        (data.metadata["split"] == split) & (data.metadata["storm_id"].astype(str) == storm_id)
    ].iloc[0]
    start_metric, end_metric, points_metric = st.columns(3)
    start_metric.metric("Track start (UTC)", metadata["start_time_utc"].strftime("%b %d · %H:%M"))
    end_metric.metric("Track end (UTC)", metadata["end_time_utc"].strftime("%b %d · %H:%M"))
    points_metric.metric("Observed points", f"{int(metadata['observation_count']):,}")

    st.subheader("Observed track")
    st.plotly_chart(
        build_track_figure(points),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
    st.caption(
        "Points follow UTC time order. This data contract contains observed coordinates "
        "and forecast errors, but not predicted coordinates."
    )

    storm_performance = data.performance.loc[
        (data.performance["split"] == split)
        & (data.performance["storm_id"].astype(str) == storm_id)
    ]
    if not storm_performance.empty:
        st.subheader("Forecast error for this storm")
        st.plotly_chart(
            _build_storm_error_figure(storm_performance),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
