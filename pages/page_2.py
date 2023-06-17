import dash
import yfinance as yf
import yahooquery as yq
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio
from dash import html, Dash, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

valid_intervals = ['1m', '2m', '5m', '15m', '30m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

symbol_df = pd.read_csv("Ticker_list.csv")


stocks = [c for c in symbol_df[symbol_df['Type'] == 'stock']['Symbol']]

# valid pairs for interval and period
valid_pairs = {
    'max': ['1d', '5d', '1wk', '1mo', '3mo'],
    '2y': ['1h'],
    '1mo': ['90m', '30m', '15m', '5m', '2m'],
    '5d': ['1m']
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

dash.register_page(__name__, name="DTW")

layout = html.Div(
    [
        html.Div(
            [

                html.Div(
                    [

                        dmc.MultiSelect(
                            id='stock-list-com',
                            label='Stock',
                            data=stocks[:100],
                            clearable=True,
                            searchable=True,
                            placeholder='Select stock... '
                        )
                    ],
                    style={"flex": "25%",
                           "textAlign": "center"}
                ),
                html.Div(
                    [
                        dmc.MultiSelect(
                            id='crypto-list-com',
                            label='Crypto',
                            data=symbol_df[symbol_df['Type'] == 'crypto']['Symbol'].tolist(),
                            clearable=True,
                            searchable=True,
                            placeholder='Select crypto...'
                        )
                    ],
                    style={"flex": "25%",
                           "textAlign": "center"}
                ),
                html.Div(
                    [
                        dmc.MultiSelect(
                            id='gold-list-com',
                            label='Gold',
                            data=symbol_df[symbol_df['Type'] == 'gold']['Symbol'].tolist(),
                            clearable=True,
                            searchable=True,
                            placeholder='Select gold...'
                        )

                    ],
                    style={"flex": "25%",
                           "textAlign": "center"}
                ),
                html.Div(
                    [
                        dmc.MultiSelect(
                            id='forex-list-com',
                            label='Forex',
                            data=symbol_df[symbol_df['Type'] == 'forex']['Symbol'].tolist(),
                            searchable=True,
                            clearable=True,
                            placeholder='Select currency...'
                        )
                    ],
                    style={"flex": "25%",
                           "textAlign": "center"}
                )
            ],
            style={"display": 'flex'}
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H4("Price",
                                style={'textAlign': 'center'}),
                        dcc.RadioItems(
                            id='log-return',
                            options=[
                                {"label": 'Mean Price', "value": "Mean_Price"},
                                {"label": "Percent Change", "value": "PrcChange"},
                                {"label": "Log Return", "value": 'LogReturn'},
                                {"label": "Open", "value": "Open"},
                                {"label": "Close", "value": "Close"},
                                {"label": "High", "value": "High"},
                                {"label": "Low", "value": "Low"},
                            ],
                            value='Mean_Price',
                            inline=True
                        )
                    ],
                    style={"display": "50%"}
                ),

                html.Div(
                    [
                        html.H4("Interval",
                                style={'textAlign': 'center'}),
                        dcc.RadioItems(
                            id='interval-range',
                            value='1mo',
                            inline=True,
                            options=[{"label": c, 'value': c} for c in valid_intervals],
                        )

                    ],
                    style={'display': "50%"}
                )
            ],
            style={'display': 'flex'}
        ),
        html.Div(
            children=
            [
                html.H3("Day shift"),
                dcc.Slider(
                    min=1,
                    max=30,
                    step=1,
                    marks=None,
                    value=1,
                    id='shift-slider',
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ],

            id='shift-slider-container'
        ),

        html.Div(
            id='dtw-chart'
        )

    ],

    style=CONTENT_STYLE
)


# show shift slider
@callback(
    Output(component_id='shift-slider-container', component_property='style'),
    Input(component_id='log-return', component_property='value')
)
def show_shift_slider(price_type):
    if price_type in ['LogReturn', 'PrcChange']:
        return {"display": 'block'}
    else:
        return {'display': 'none'}


# comparison plot
@callback(
    Output(component_id='dtw-chart', component_property='children'),
    Input(component_id='stock-list-com', component_property='value'),
    Input(component_id='crypto-list-com', component_property='value'),
    Input(component_id='gold-list-com', component_property='value'),
    Input(component_id='forex-list-com', component_property='value'),
    Input(component_id='interval-range', component_property='value'),
    Input(component_id='log-return', component_property='value'),
    Input(component_id='shift-slider', component_property='value')

)
def dtw_chart(stock, crypto, gold, forex, interval, price_type, shift_n):
    all_list = []

    if forex is not None:
        all_list.extend(forex)
    if stock is not None:
        all_list.extend(stock)
    if crypto is not None:
        all_list.extend(crypto)
    if gold is not None:
        all_list.extend(gold)

    if len(all_list) > 0:
        period = [d for d in valid_pairs.keys() if interval in valid_pairs[d]][0]

        figure = go.Figure()

        for symbol in all_list:
            if forex is not None and symbol in forex:
                df_inner = yf.Ticker(symbol + '=X').history(period=period, interval=interval).reset_index()
            else:
                df_inner = yf.Ticker(symbol).history(period=period, interval=interval).reset_index()
            df_inner['Date'] = pd.to_datetime(df_inner.iloc[:, 0])
            df_inner['Mean_Price'] = (df_inner['Open'] + df_inner['Close']) / 2
            df_inner['PrcChange'] = df_inner['Mean_Price'].pct_change()
            df_inner['LogReturn'] = np.log(df_inner['Mean_Price']) - np.log(df_inner['Mean_Price'].shift(shift_n))

            figure.add_trace(
                go.Scatter(
                    x=df_inner['Date'],
                    y=df_inner[price_type],
                    name=symbol_df[symbol_df['Symbol'] == symbol]['Name'].tolist()[0],
                    mode='lines'
                ))
            figure.update_layout(
                template=pio.templates['plotly_white']
            )

        return html.Div(
            dcc.Graph(
                figure=figure
            )
        )
