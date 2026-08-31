import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import DashboardBundle
from app.pages.overview import MODEL_COLORS, MODEL_LABELS


def build_distribution_figure(errors: pd.DataFrame) -> go.Figure:
    chart_data = errors.assign(
        model_label=errors["model"].map(MODEL_LABELS).fillna(errors["model"])
    )
    figure = px.histogram(
        chart_data,
        x="error_km",
        color="model_label",
        barmode="overlay",
        histnorm="probability density",
        opacity=0.62,
        nbins=36,
        color_discrete_map={MODEL_LABELS[key]: value for key, value in MODEL_COLORS.items()},
        labels={"error_km": "Position error (km)", "model_label": "Model"},
    )
    figure.update_layout(
        template="plotly_white",
        height=400,
        margin={"l": 12, "r": 12, "t": 20, "b": 12},
        legend={"orientation": "h", "y": 1.14, "x": 0},
        yaxis_title="Density",
    )
    figure.update_xaxes(rangemode="tozero", gridcolor="#E7EBEF")
    figure.update_yaxes(gridcolor="#E7EBEF")
    return figure


def build_improvement_figure(comparison: pd.DataFrame) -> go.Figure:
    chart_data = comparison.assign(
        result=comparison["better_model"].map(
            {"kalman": "Kalman lower RMSE", "persistence": "Persistence lower RMSE", "tie": "Tie"}
        )
    )
    figure = px.scatter(
        chart_data,
        x="forecast_horizon_hours",
        y="rmse_improvement_km",
        color="result",
        hover_name="storm_id",
        color_discrete_map={
            "Kalman lower RMSE": MODEL_COLORS["kalman"],
            "Persistence lower RMSE": MODEL_COLORS["persistence"],
            "Tie": "#64748B",
        },
        labels={
            "forecast_horizon_hours": "Forecast horizon (hours)",
            "rmse_improvement_km": "Persistence RMSE − Kalman RMSE (km)",
            "result": "Lower RMSE",
        },
    )
    figure.update_traces(marker={"size": 10, "opacity": 0.78})
    figure.add_hline(line_color="#64748B", line_dash="dash", y=0)
    figure.update_layout(
        template="plotly_white",
        height=400,
        margin={"l": 12, "r": 12, "t": 20, "b": 12},
        legend={"orientation": "h", "y": 1.14, "x": 0},
    )
    figure.update_xaxes(gridcolor="#E7EBEF")
    figure.update_yaxes(gridcolor="#E7EBEF")
    return figure


def render(data: DashboardBundle) -> None:
    st.title("Error analysis")
    st.caption("Filters apply to both model distributions and paired storm comparisons.")

    splits = sorted(data.errors["split"].dropna().astype(str).unique())
    default_split = splits.index(data.headline_split) if data.headline_split in splits else 0
    split_column, horizon_column = st.columns([1, 2])
    with split_column:
        split = st.selectbox("Evaluation split", splits, index=default_split)
    available_horizons = sorted(
        data.errors.loc[data.errors["split"] == split, "forecast_horizon_hours"].unique()
    )
    with horizon_column:
        horizons = st.multiselect(
            "Forecast horizons (hours)",
            available_horizons,
            default=available_horizons,
        )
    if not horizons:
        st.info("Select at least one forecast horizon.")
        return

    errors = data.errors.loc[
        (data.errors["split"] == split) & (data.errors["forecast_horizon_hours"].isin(horizons))
    ]
    comparison = data.comparison.loc[
        (data.comparison["split"] == split)
        & (data.comparison["forecast_horizon_hours"].isin(horizons))
    ]

    st.subheader("Position error distributions")
    st.caption(f"{len(errors):,} model forecasts · Error in kilometers")
    st.plotly_chart(
        build_distribution_figure(errors),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    st.subheader("Paired RMSE differences")
    st.caption(
        "Each point is one storm and horizon. Positive values mean the Kalman filter "
        "has lower RMSE."
    )
    st.plotly_chart(
        build_improvement_figure(comparison),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
