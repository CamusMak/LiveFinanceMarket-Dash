import dash
import yfinance as yf
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import html, Dash, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc

valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
valid_intervals = ['1m', '2m', '5m', '15m', '30m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

symbol_df = pd.read_csv("Ticker_list.csv")

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

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}
"######################################################################################################################"
"                                                                                                                     "
"                                          LIVE STOCK MARKET DASHBOARD                                                "
"                                                                                                                     "
"######################################################################################################################"

"######################################################################################################################"
"                                                   APP LAYOUT                                                         "
"######################################################################################################################"

dash.register_page(__name__, name='Candle chart')

"######################################################################################################################"

"SYMBOL LIST"

symbol_tabs = dcc.Tabs(
    children=[
        dcc.Tab(id='stock', label='Stock', value='stock'),
        dcc.Tab(id='crypto', label='Crypto', value='crypto'),
        dcc.Tab(id='gold', label='Gold', value='gold'),
        dcc.Tab(id='forex', label='Forex', value='forex')
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
                             html.H3("Symbol list"),
                             dcc.Dropdown(
                                 id='symbol-list',
                                 style={"width": '200px',
                                        'flex': '50%'}
                             )
                         ],

                         style={'flex': '30%'}

                     ),

                     html.Div(
                         children=
                         [
                             html.H3('Time interval'),
                             dcc.RadioItems(
                                 id='valid-interval',
                                 value='1mo',
                                 inline=True,
                                 options=[{"label": c, 'value': c} for c in valid_intervals],
                             )
                         ],

                         style={'flex': '70%'}
                     )

                 ]
                 ),

        #     first page
        # candle chart

        html.Br(),
        html.Div(
            [
                html.Div(
                    dcc.Checklist(
                        id='line-chart-to-candle',
                        options=[{"label": c, 'value': c} for c in ['Open', 'Close', 'High', 'Low', 'MA']],
                        inline=True),
                    style={"flex": "50%"}
                ),
                html.Br(),
                html.Div(
                    id='ma-range',
                    children=
                    [
                        dcc.Slider(
                            min=2,
                            max=20,
                            step=1,
                            value=7,
                            id='ma-days-range',
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ],

                    style={"flex": "60%"}
                )

            ],
            style={"display": 'flex'},

        ),
        html.Div(
            id='candle-chart-df'
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
        candle_chart_div,
        html.Br()

    ],
    style=CONTENT_STYLE
)

"LAYOUT"
"######################################################################################################################"

layout = html.Div(
    page_1
)

"######################################################################################################################"
"                                                   APP SERVER                                                         "
"######################################################################################################################"


# symbol list
@callback(
    Output(component_id='symbol-list', component_property='options'),
    Input(component_id='symbol-type', component_property='value'),
)
def symbol_options(symbol_type):
    return symbol_df[symbol_df['Type'] == symbol_type]['Symbol'].tolist()


@callback(
    Output(component_id='symbol-list', component_property='value'),
    Input(component_id='symbol-list', component_property='options')
)
def symbol_values(symbol_list):
    return symbol_list[0]


# ma range

@callback(
    Output(component_id='ma-days-range', component_property='style'),
    Input(component_id='line-chart-to-candle', component_property='value')
)
def ma_range(price_type):
    if price_type and 'MA' in price_type:
        return {'flex': "60%", 'display': 'block'}
    else:
        return {'flex': "60%", 'display': 'none'}


# data table
# candle chart
@callback(
    Output(component_id='candle-chart-df', component_property='children'),
    Input(component_id='symbol-list', component_property='value'),
    Input(component_id='valid-interval', component_property='value'),
    Input(component_id='line-chart-to-candle', component_property='value'),
    Input(component_id='symbol-type', component_property='value'),
    Input(component_id='ma-days-range', component_property='value'),
    config_prevent_initial_callbacks=True
)
def candle_chart_df(ticker, interval, price_type, symbol_type, ma_range_):
    # trace name for candle chart

    candle_trace = None

    # title
    title = symbol_df[symbol_df['Symbol'] == ticker]['Name'].tolist()[0]

    if symbol_type.lower() == 'forex':
        ticker = ticker + "=X"

    period = [d for d in valid_pairs.keys() if interval in valid_pairs[d]][0]

    df = yf.Ticker(ticker).history(period=period, interval=interval).reset_index()

    df['Date'] = pd.to_datetime(df.iloc[:, 0])

    # calculating moving average
    df['MA'] = df['Close'].rolling(7).mean()

    if price_type:
        candle_trace = 'Candle chart'
        if price_type[0] == 'MA':
            df['MA'] = df['Close'].rolling(ma_range_).mean()

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
        template=pio.templates['plotly_white'],
        title=title
    )

    if price_type:
        df_long = pd.melt(df, id_vars=['Date'])
        # variable is the name of new created categorical column
        # value is the name of new created numerical value
        df_long = df_long[df_long['variable'].isin(price_type)]

        df_long = df_long.sort_values(['Date', 'variable'])

        candle_chart.update_layout()

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

    # pie chart
    # change
    df['Change_Num'] = df['Close'] - df['Open']
    # df['Change_Num'] = np.where(df['Change_Num'] <= 0, -1, 1)
    df['Change'] = np.where(df['Change_Num'] <= 0, 'Loss', 'Gain')

    df['Change_Num'] = abs(df['Change_Num'])
    pie_chart_counts = px.pie(data_frame=df,
                              names=df['Change'])
    pie_chart_counts.update_layout(
        title="Count of Gain & Loss"
    )

    pie_chart_value = px.pie(data_frame=df,
                             names=df['Change'],
                             values=df['Change_Num'])

    pie_chart_value.update_layout(
        title="Sum of Gain & Loss"
    )

    # mean and change
    df['RafoChange'] = (df['Close'] - df['Open']) / df['Close']
    df['Voaltility'] = (df['High'] - df['Low']) / df['Low']

    # Rafo dist plot
    rafo_hist = px.histogram(data_frame=df,
                             x=df['RafoChange'])

    rafo_hist.update_layout(
        title='Loss & Gain',
        template=pio.templates['plotly_white']

    )

    vol_hist = px.histogram(data_frame=df,
                            x=df['Voaltility'])

    vol_hist.update_layout(
        title='Volatility',
        template=pio.templates['plotly_white']
    )


    return html.Div(
        children=[
            dbc.Row(
                [
                    dcc.Graph(figure=candle_chart,
                              config={"scrollZoom": True}),
                ]
            ),

            html.Hr(),
            dbc.Row(
                [
                    html.H3("Pie charts",
                            style={"textAlign": 'center'}),

                    html.Div(
                        id='hist-pie-div',
                        style={"display": "flex"},
                        children=
                        [

                            html.Div(
                                style={"flex": '50%'},
                                children=
                                [
                                    # dcc.RadioItems(
                                    #     id='price-type-hist',
                                    #     value='Open',
                                    #     options=[{'label': c, 'value': c} for c in
                                    #              ['Open & Close', 'High & Low', 'MA']],
                                    #     inline=True
                                    # ),
                                    dcc.Graph(

                                        figure=pie_chart_counts
                                    )
                                ]
                            ),

                            html.Div(
                                id='pie-chart',
                                style={'flex': '50%'},
                                children=
                                [
                                    dcc.Graph(
                                        figure=pie_chart_value
                                    )
                                ]

                            )

                        ]
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                [
                    html.H3("Histograms",
                            style={'textAlign':"center"}),
                    html.Div(
                        children=
                        [
                            html.Div(
                                children=
                                [
                                    dcc.Graph(
                                        figure=rafo_hist
                                    )
                                ],
                                style={"flex": "50%"}

                            ),
                            html.Div(
                                children=
                                [
                                    dcc.Graph(
                                        figure=vol_hist
                                    )
                                ],
                                style={"flex":"50%"}
                            )

                        ],
                        style={"display":'flex'}
                    )
                ]
            ),

            html.Div(
                dash_table.DataTable(
                    df.to_dict('records'), [{"name": c, "id": c} for c in df.columns], id='candle-table'
                ),
                style={"display": 'none'}
            )

        ]
    )
