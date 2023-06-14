import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import html, Dash, dcc, callback, Input, Output

ticker_list = ['DJIA', 'DOW', 'EXPE', 'PXD', 'MCHP', 'CRM', 'NRG', 'NOW']

valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
valid_intervals = ['1m', '2m', '5m', '15m', '30m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

symbol_df = pd.read_csv("Ticker_list.csv")[['Symbol', 'Type']]

# valid_tickers = symbol_df['Symbol'].to_numpy()

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

app = Dash(__name__)
server = app.server

# symbol dive
symbol_div = html.Div([

    dcc.RadioItems(
        id='type_',
        options=[{"label": c, "value": c} for c in symbol_df['Type'].unique()],
        value='stock',
        inline=True
    ),

    html.Hr(),

    dcc.Dropdown(
        id='ticker_name'),

    html.Hr(),

    dcc.RadioItems(
        id='valid_interval',
        options=[{"label": c, "value": c} for c in valid_intervals],
        value='1d',
        inline=True
    ),

    html.Hr(),


    dcc.Graph(
        id='candle_chart',
        config={'scrollZoom': True}
    )
    ],
        style={"color":'blue','fontSize':14}

)

app.layout = html.Div([
    symbol_div
])


@app.callback(
    Output(component_id='ticker_name', component_property='options'),
    Input(component_id='type_', component_property='value')
)
def return_ticker_options(type_):
    # return [{"label": c, "value": c} for c in symbol_df[symbol_df['Type'] == type_]['Symbol'].tolist()]
    return symbol_df[symbol_df['Type'] == type_]['Symbol'].tolist()


@app.callback(
    Output(component_id='ticker_name', component_property='value'),
    Input(component_id='ticker_name', component_property='options')
)
def return_ticker_name(ticker_name):
    # return [ticker['value'] for ticker in ticker_name]
    return ticker_name[0]


@app.callback(
    Output(component_id='candle_chart', component_property='figure'),
    Input(component_id='ticker_name', component_property='value'),
    Input(component_id='valid_interval', component_property='value')
)
def candle_chart(ticker_name, interval):
    # match to the correspondent interval

    period = [x for x in valid_pairs.keys() if interval in valid_pairs[x]][0]

    df = yf.Ticker(ticker_name).history(interval=interval,
                                        period=period).reset_index()
    df['Date'] = pd.to_datetime(df.iloc[:, 0])

    # title
    title_part = title_names[interval]
    title = ticker_name + ' Candle chart' + title_part

    if len(title_part.split()) == 1:
        title = ticker_name + title_part + ' Candle chart'

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            close=df['Close'],
            high=df['High'],
            low=df['Low']
        )
    )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template=pio.templates['plotly_white']

    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
