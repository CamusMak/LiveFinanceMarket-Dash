import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import html, Dash, dcc, callback, Input, Output

ticker_list = ['DJIA', 'DOW', 'EXPE', 'PXD', 'MCHP', 'CRM', 'NRG', 'NOW']

data_frames = {}

for ticker in ticker_list:
    data = yf.Ticker(ticker).history(period='max').reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data_frames[ticker] = data

app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(id='ticker_name',
                 options=[c for c in data_frames.keys()],
                 value='DJIA'),
    dcc.Graph(
        id='candle_chart'
    )
])


@app.callback(
    Output(component_id='candle_chart', component_property='figure'),
    Input(component_id='ticker_name', component_property='value')
)

def candle_chart(ticker_name):
    df = data_frames[ticker_name]

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
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
