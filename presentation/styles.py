"""
Module for handling UI styling for the stock analysis dashboard.
"""

import streamlit as st


def apply_custom_styling():
    """Apply custom CSS styling to the Streamlit application."""
    st.markdown(
        """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        position: sticky !important;
        top: 0;
        background-color: #262730;
        z-index: 999;
        padding: 4px 0px;
        margin-top: -15px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 4px 4px 0 0;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs {
        padding-top: 15px;
    }
    .metrics-container {
        margin-bottom: 20px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def create_metrics_container(content_callable):
    """Create a container with proper styling for metrics.

    Args:
        content_callable: A callable that will render the content inside the container
    """
    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
    content_callable()
    st.markdown("</div>", unsafe_allow_html=True)


def get_dataframe_config():
    """Get standard configuration for dataframes in the dashboard.

    Returns:
        dict: Configuration parameters for st.dataframe
    """
    return {
        "height": 400,  # Limit height to ensure it doesn't take too much space
        "use_container_width": True,  # Use container width instead of fixed width
        "hide_index": True,
    }
