import streamlit as st

from app.data_loader import DashboardBundle

REPORT_URL = (
    "https://github.com/sardorsob/Probabilistic-Storm-Tracks/blob/main/Report/"
    "Sequential_Bayesian_Inference_Applied_to_Hurricane_Tracking.pdf"
)


def render(data: DashboardBundle) -> None:
    st.title("Methodology")

    st.subheader("Forecast setup")
    st.markdown(
        "The persistence model advances the storm using its latest observed motion. "
        "The Kalman filter estimates position and velocity as a continuous state and "
        "updates that state as observations "
        "arrive. Both models are evaluated on matching forecast origins and horizons."
    )

    st.subheader("Evaluation")
    split_policy = data.manifest.get("split_policy", {})
    st.markdown(
        f"Dashboard summaries use the **{data.headline_split}** split. "
        f"{split_policy.get('storm_membership', 'Storms remain within one declared split.')}"
    )
    definitions = data.manifest.get("metric_definitions", {})
    st.markdown(
        "\n".join(
            [
                f"- **Position error:** {definitions.get('error_km', 'Distance in kilometers.')}",
                "- **RMSE:** "
                f"{definitions.get('rmse_km', 'Root mean squared error in kilometers.')}",
                "- **Paired difference:** Persistence RMSE minus Kalman RMSE. "
                "Positive values indicate lower Kalman RMSE.",
            ]
        )
    )

    st.subheader("Technical report")
    st.markdown(
        f"Read the [project report]({REPORT_URL}) for the model equations and full discussion."
    )
