import sys
from pathlib import Path

import streamlit as st

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) in sys.path:
        sys.path.remove(str(project_root))
    sys.path.insert(0, str(project_root))

from app.data_loader import DashboardDataError, load_dashboard_data, resolve_data_dir
from app.pages import error_analysis, methodology, overview, storm_explorer


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #DCE2E8;
            border-left: 4px solid #087F8C;
            border-radius: 0.65rem;
            padding: 0.8rem 1rem;
        }
        [data-testid="stMetricLabel"] {
            color: #526274;
        }
        @media (max-width: 640px) {
            .block-container {
                padding: 1.25rem 1rem 2.5rem;
            }
            h1 {
                font-size: 2.1rem !important;
                line-height: 1.15 !important;
            }
            [data-testid="stMetric"] {
                padding: 0.7rem 0.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run() -> None:
    st.set_page_config(
        page_title="Storm track evaluation",
        page_icon="🌀",
        layout="wide",
        initial_sidebar_state="auto",
    )
    apply_styles()
    try:
        data = load_dashboard_data(resolve_data_dir())
    except DashboardDataError as exc:
        st.title("Storm track forecast evaluation")
        st.warning(str(exc))
        st.markdown(
            "Run `python scripts/build_dashboard_data.py --help` for the required inputs, then set "
            "`STORM_TRACKS_DASHBOARD_DATA_DIR` to the output directory."
        )
        st.stop()

    def show_overview() -> None:
        overview.render(data)

    def show_error_analysis() -> None:
        error_analysis.render(data)

    def show_storm_explorer() -> None:
        storm_explorer.render(data)

    def show_methodology() -> None:
        methodology.render(data)

    pages = [
        st.Page(show_overview, title="Overview", default=True),
        st.Page(show_error_analysis, title="Error analysis"),
        st.Page(show_storm_explorer, title="Storm explorer"),
        st.Page(show_methodology, title="Methodology"),
    ]
    st.navigation(pages, position="sidebar").run()


run()
