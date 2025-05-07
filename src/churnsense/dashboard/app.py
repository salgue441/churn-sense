"""Main application file for ChurnSense Dashboard."""

import os
from pathlib import Path
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from churnsense.config import config
from churnsense.utils.logging import setup_logger
from churnsense.dashboard.layout import create_layout
from churnsense.dashboard.callbacks import register_callbacks

logger = setup_logger(__name__)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    title="ChurnSense Dashboard",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.layout = create_layout(app)

register_callbacks(app)
server = app.server

if __name__ == "__main__":
    logger.info("Starting ChurnSense Dashboard")
    port = int(os.environ.get("PORT", config.port))
    app.run(host=config.host, port=port, debug=config.debug)
