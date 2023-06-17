import dash
import yfinance as yf
import yahooquery as yq
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio
from dash import html, Dash, dcc, callback, Input, Output, dash_table, ctx
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

dash.register_page(__name__, name="Comparison of prices")

layout = html.Div(
    [
        html.H1("Comparison of time series",
                style={'textAlign': 'center'}),
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
                    style={"flex": "20%",
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
                    style={"flex": "20%",
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
                    style={"flex": "20%",
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
                    style={"flex": "20%",
                           "textAlign": "center"}
                ),
                html.Div(
                    [
                        dmc.TextInput(
                            label='Custom symbol',
                            id='custom-symbol-input',
                            placeholder='Write ticker name...(APPL)',
                        )
                    ],
                    style={"flex": "20%"}
                ),
                html.Div(
                    [
                        html.Button(
                            "Search",
                            id='submit',
                            n_clicks=0

                        )
                    ],
                    style={'flex': '10%'}
                ),
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
                                # {"label": 'Mean Price', "value": "Mean_Price"},
                                # {"label": "Percent Change", "value": "PrcChange"},
                                # {"label": "Log Return", "value": 'LogReturn'},
                                # {"label": "Open", "value": "Open"},
                                # {"label": "Close", "value": "Close"},
                                # {"label": "High", "value": "High"},
                                # {"label": "Low", "value": "Low"},
                                "Mean Price", "Percent Change", "Log Return", "Open", "Close", "High", "Low"
                            ],
                            value='Mean Price',
                            inline=True
                        )
                    ],
                    style={"display": "35%"}
                ),

                html.Div(
                    [

                    ],
                    style={'flex': '5%'}
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
                    style={'display': "35%"}
                ),

                html.Div(
                    [
                        dmc.Select(
                            id='custom-select-symbol'
                        )
                    ],
                    id='custom-symbol-container',
                    style={'display': "none"}
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
    if price_type in ['Log Return', 'Percent Change']:
        return {"display": 'block'}
    else:
        return {'display': 'none'}


# custom input container
@callback(
    Output(component_id='custom-symbol-container', component_property='children'),
    Input(component_id='custom-symbol-input', component_property='value'),
    Input(component_id='submit', component_property='n_clicks'),
)
def return_custom_symbol(user_input, submit):
    value = []
    if 'submit' == ctx.triggered_id:
        custom_symbol = str.upper(user_input)
        if len(yf.Ticker(custom_symbol).history()) > 0:
            value.append(user_input)
        elif len(yf.Ticker(user_input + "=X").history()) > 0:
            # if custom symbol is currency ticker, assign custom_symbol_is_currency to True
            value.append(user_input)

    if len(value) != 0:
        return html.Div(
            [
                dmc.Select(
                    id='custom-select-symbol',
                    data=[value],
                    value=value
                )
            ],
            style={'display': 'block'}
        )

# tell user that input was not valid
# @callback(
#     Output(component_id='custom-select-symbol',component_property='pattern'),
#     Input(component_id='custom-symbol-input',component_property='value'),
#     Input(component_id='submit', component_property='n_clicks')
# )
# def if_valid(user_value,submit):
#
#     if ('submit' == ctx.triggered_id) and (len(user_value)==0):
#         return "Ticker name not found"


# comparison plot

@callback(
    Output(component_id='dtw-chart', component_property='children'),
    Input(component_id='stock-list-com', component_property='value'),
    Input(component_id='crypto-list-com', component_property='value'),
    Input(component_id='gold-list-com', component_property='value'),
    Input(component_id='forex-list-com', component_property='value'),
    Input(component_id='interval-range', component_property='value'),
    Input(component_id='log-return', component_property='value'),
    Input(component_id='shift-slider', component_property='value'),
    Input(component_id='custom-select-symbol', component_property='value'),

)
def dtw_chart(stock, crypto, gold, forex, interval, price_type, shift_n, custom_symbol):
    all_list = []

    if forex is not None:
        all_list.extend(forex)
    if stock is not None:
        all_list.extend(stock)
    if crypto is not None:
        all_list.extend(crypto)
    if gold is not None:
        all_list.extend(gold)
    if custom_symbol:
        all_list.extend(custom_symbol)

    # this variable is needed to check if custom symbol is currency symbol or not

    if len(all_list) > 0:
        period = [d for d in valid_pairs.keys() if interval in valid_pairs[d]][0]

        figure = go.Figure()

        for symbol in all_list:
            symbol = str.upper(symbol)

            print(symbol)

            if len(yf.Ticker(symbol).history()) != 0:
                df_inner = yf.Ticker(symbol).history(period=period, interval=interval).reset_index()
            else:
                df_inner = yf.Ticker(symbol + '=X').history(period=period, interval=interval).reset_index()

            df_inner['Date'] = pd.to_datetime(df_inner.iloc[:, 0])
            df_inner['Mean Price'] = (df_inner['Open'] + df_inner['Close']) / 2
            df_inner['Percent Change'] = df_inner['Close'].pct_change()
            df_inner['Log Return'] = np.log(df_inner['Close']) - np.log(df_inner['Close'].shift(shift_n))

            figure.add_trace(
                go.Scatter(
                    x=df_inner['Date'],
                    y=df_inner[price_type],
                    name=symbol_df[symbol_df['Symbol'] == symbol]['Name'].tolist()[0],
                    mode='lines'
                ))
            figure.update_layout(
                template=pio.templates['plotly_white'],
                title=symbol,
                xaxis_title='Date',
                yaxis_title=price_type
            )

        return html.Div(
            dcc.Graph(
                figure=figure
            )
        )
