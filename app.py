
import dash
from dash import html, Dash, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
# from dash.exceptions import  PreventUpdate

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "10rem",
    "padding": "4rem 2rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "10rem 1rem",
}

app = Dash(__name__, use_pages=True, external_scripts=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True,
           prevent_initial_callbacks=True)

server = app.server

sidebar = dbc.Nav(
    [
        dbc.NavLink(
            [
                html.Div(page["name"], className="ms-2"),
            ],
            href=page["path"],
            active="exact",
            style={'padding-top': 30, 'padding-bottom': 30}

        )
        for page in dash.page_registry.values()
    ],
    vertical=True,
    pills=True,
    className="bg-light",
    style=SIDEBAR_STYLE
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div("Live stock market",
                         style={'fontSize': 50, 'textAlign': 'center'}))
    ]),

    html.Hr(),

    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),

            dbc.Col(
                [
                    dash.page_container
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        ]
    )
], fluid=True)

if __name__ == "__main__":
    app.run_server(debug=True,port=8080)
