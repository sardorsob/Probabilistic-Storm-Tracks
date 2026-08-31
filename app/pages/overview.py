import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import DashboardBundle

MODEL_LABELS = {"persistence": "Persistence", "kalman": "Kalman filter"}
MODEL_COLORS = {"persistence": "#C26D0A", "kalman": "#087F8C"}


def build_rmse_figure(summary: pd.DataFrame) -> go.Figure:
    chart_data = summary.assign(
        model_label=summary["model"].map(MODEL_LABELS).fillna(summary["model"])
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
        height=430,
        margin={"l": 12, "r": 12, "t": 20, "b": 12},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    figure.update_xaxes(dtick=6, gridcolor="#E7EBEF")
    figure.update_yaxes(rangemode="tozero", gridcolor="#E7EBEF")
    return figure


def render(data: DashboardBundle) -> None:
    split = data.headline_split
    summary = data.summary.loc[data.summary["split"] == split].copy()
    errors = data.errors.loc[data.errors["split"] == split]
    cases = errors[["storm_id", "origin_idx", "forecast_horizon_hours"]].drop_duplicates()

    st.title("Storm track forecast evaluation")
    st.caption(
        f"{split.title()} split · Position error in kilometers · Persistence and Kalman forecasts"
    )

    split_metric, storms_metric, cases_metric, horizons_metric = st.columns(4)
    split_metric.metric("Evaluation split", split.title())
    storms_metric.metric("Storms", f"{cases['storm_id'].nunique():,}")
    cases_metric.metric("Forecast cases", f"{len(cases):,}")
    horizons_metric.metric("Horizons", f"{cases['forecast_horizon_hours'].nunique():,}")

    st.subheader("RMSE by forecast horizon")
    st.plotly_chart(
        build_rmse_figure(summary),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    with st.expander("Metric definitions"):
        definitions = data.manifest.get("metric_definitions", {})
        st.markdown(
            "\n".join(
                [
                    "- **Position error:** "
                    f"{definitions.get('error_km', 'Distance in kilometers.')}",
                    "- **RMSE:** "
                    f"{definitions.get('rmse_km', 'Root mean squared error in kilometers.')}",
                    "- **Forecast case:** One storm, forecast origin, and forecast horizon. "
                    "Each model evaluates the same cases.",
                ]
            )
        )

    generated = data.manifest.get("generated_at_utc")
    if generated:
        generated_at = pd.to_datetime(generated, utc=True).strftime("%Y-%m-%d %H:%M UTC")
        st.caption(f"Data contract generated {generated_at}.")
