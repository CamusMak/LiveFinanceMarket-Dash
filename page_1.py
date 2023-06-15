import dash
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import html, Dash, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

# ticker_list = ['DJIA', 'DOW', 'EXPE', 'PXD', 'MCHP', 'CRM', 'NRG', 'NOW']

valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
valid_intervals = ['1m', '2m', '5m', '15m', '30m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

symbol_df = pd.read_csv("Ticker_list.csv")[['Symbol', 'Type']]

# valid pairs for interval and period
valid_pairs = {
    'max': ['1d', '5d', '1wk', '1mo', '3mo'],
    '2y': ['1h'],
    '1mo': ['90m', '30m', '15m', '5m', '2m'],
    '5d': ['1m']
}

# title for candle chart

title_names = {

    '1d': " daily",
    '5d': " of 5 days",
    '1wk': " weekly",
    '1mo': ' monthly',
    '3mo': ' quarterly',
    '1h': ' hourly',
    '90m': ' for 90 minutes',
    '30m': ' for 30 minutes',
    '15m': ' for 15 minutes',
    '5m': ' for 5 minutes',
    '2m': ' for 2 minutes',
    '1m': ' for 1 minutes'
}

"######################################################################################################################"
"                                                                                                                     "
"                                          LIVE STOCK MARKET DASHBOARD                                                "
"                                                                                                                     "
"######################################################################################################################"

"######################################################################################################################"
"                                                   APP LAYOUT                                                         "
"######################################################################################################################"

app = Dash(__name__,
           external_scripts=[dbc.themes.DARKLY])

server = app.server

"PAGE 1"
"######################################################################################################################"

"SYMBOL LIST"

symbol_tabs = dcc.Tabs(
    children=[
        dcc.Tab(id='stock', label='Stock', value='stock'),
        dcc.Tab(id='crypto', label='Crypto', value='crypto'),
        dcc.Tab(id='gold', label='Gold', value='gold')
    ],
    id='symbol-type',
    value='crypto'
)

"CANDLE CHART"

candle_chart_div = html.Div(
    children=[
        html.Div(style={'display': 'flex'},
                 children=[

                     html.Div(
                         children=
                         [
                             html.H1("Symbol list"),
                             dcc.Dropdown(
                                 id='symbol-list',
                                 style={"width": '200px',
                                        'flex': '50%'}
                             )
                         ],

                         style={'flex': '50%'}

                     ),

                     html.Div(
                         children=
                         [
                             html.H1('Time interval'),
                             dcc.RadioItems(
                                 id='valid-interval',
                                 value='1mo',
                                 inline=True,
                                 options=[{"label": c, 'value': c} for c in valid_intervals],
                             )
                         ],

                         style={'flex': '50%'}
                     )

                 ]
                 ),

        #     first page
        # candle chart

        html.Br(),
        dcc.Checklist(
            id='line-chart-to-candle',
            options=[{"label": c, 'value': c} for c in ['Open', 'Close', 'High', 'Low']],
            inline=True
        ),
        html.Div(
            id='candle-chart-df'
        )

    ]

)

"HISTOGRAM & "

histogram_div = html.Div(
    id='hist-pie-div',
    style={"display": "flex"},
    children=
    [
        html.Div(
            id='hist-chart',
            style={"flex": '50%'},
            children=
            [
                dcc.RadioItems(
                    id='price-type',
                    value='Open',
                    options=[{'label': c, 'value': c} for c in ['Open', 'Close', 'High', 'Low']],
                    inline=True
                )
            ]
        ),

        html.Div(
            id='pie-chart',
            style={'flex': '50%'}

        )

    ]
)

page_1 = html.Div(
    id='page_1',
    children=
    [
        html.H1("Title"),
        symbol_tabs,
        html.Br(),

        #     candle chart
        candle_chart_div
    ]
)

sidebar = dbc.Navbar(
    [
        dbc.NavLink(
            [
                page_1
            ]
        )
    ]
)
"LAYOUT"
"######################################################################################################################"

app.layout = dbc.Container(

    [
        dbc.Row(
            [
                dbc.Col(
                    sidebar
                ),
                dbc.Col(
                    dash.page_container
                )
            ]
        )

    ]
)

"######################################################################################################################"
"                                                   APP SERVER                                                         "
"######################################################################################################################"


# symbol list
@app.callback(
    Output(component_id='symbol-list', component_property='options'),
    Input(component_id='symbol-type', component_property='value'),
)
def symbol_options(symbol_type):
    return symbol_df[symbol_df['Type'] == symbol_type]['Symbol'].tolist()


@app.callback(
    Output(component_id='symbol-list', component_property='value'),
    Input(component_id='symbol-list', component_property='options')
)
def symbol_values(symbol_list):
    return symbol_list[0]


# data table
@app.callback(
    Output(component_id='candle-chart-df', component_property='children'),
    Input(component_id='symbol-list', component_property='value'),
    Input(component_id='valid-interval', component_property='value'),
    Input(component_id='line-chart-to-candle', component_property='value')
)
def candle_chart_df(ticker, interval, price_type):
    # trace name for candle chart
    candle_trace = None

    if price_type:
        candle_trace = 'Candle chart'

    period = [d for d in valid_pairs.keys() if interval in valid_pairs[d]][0]

    df = yf.Ticker(ticker).history(period=period, interval=interval).reset_index()

    df['Date'] = pd.to_datetime(df.iloc[:, 0])

    candle_chart = go.Figure()

    candle_chart.add_trace(
        go.Candlestick(
            open=df['Open'],
            close=df['Close'],
            high=df['High'],
            low=df['Low'],
            x=df['Date'],
            name=candle_trace
        )
    )

    candle_chart.update_layout(
        xaxis_rangeslider_visible=False,
        template=pio.templates['plotly_white']
    )

    if price_type:
        df_long = pd.melt(df, id_vars=['Date'])
        # variable is the name of new created categorical column
        # value is the name of new created numerical value
        df_long = df_long[df_long['variable'].isin(price_type)]

        df_long = df_long.sort_values(['Date', 'variable'])

        candle_chart.update_layout()

        colors = [0, 1, 2, 3]

        for i in range(len(price_type)):
            df_ = df_long[df_long['variable'] == price_type[i]]
            candle_chart.add_trace(
                go.Scatter(
                    x=df_['Date'],
                    y=df_['value'],
                    mode='lines',
                    name=price_type[i]
                )

            )

    return html.Div(
        children=[
            dcc.Graph(figure=candle_chart,
                      config={"scrollZoom": True}),
            html.Div(
                dash_table.DataTable(
                    df.to_dict('records'), [{"name": c, "id": c} for c in df.columns], id='candle-table'
                ),
                style={"display": 'none'}
            )
        ]
    )


"Page callback"

# if __name__ == '__main__':
#     app.run_server(debug=True)
