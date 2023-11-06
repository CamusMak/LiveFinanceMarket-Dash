import sys

sys.path.append("../../LiveFinanceMarket-Dash")


# import yfinance as yf
import pandas as pd
import numpy as np

# import plotly.graph_objects as go
# import plotly.io as pio
# import plotly.express as px
from dash import html, Dash, dcc, callback, Input, Output, dash_table
import dash

from icecream import ic
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

import json
import datetime as dt

from portfolio_manager.portfolio_creator import PortfolioManager
from portfolio_manager.portfolio_visualizer import PortfolioVisualizer




# load watch list
with open("data/stock/watch_list.json","r") as file:

    watch_list = json.load(file)


# create manager instance of PortfolioManager class to manage portfolio 
manager = PortfolioManager(watch_list=watch_list)
# create visualizer intance of PortfolioVisualizer class to visualize portfolio
visualizer = PortfolioVisualizer()




linear_portfolio = manager.linear_portfolio()
# manager.update_portfolio(linear_portfolio)
manager.get_all_properties(linear_portfolio)



CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}




# register page
dash.register_page(__name__,name="Portfolio")

model_tabs = dcc.Tabs(
    children=[
        dcc.Tab(id='linear-portfolio',label='ARIMA & GARCH',value='linear-portfolio'),
        dcc.Tab(id='dense-network-portfolio',label='Dense NN',value='ense-network'),
        dcc.Tab(id='lstm-portfolio',label='LSTM',value='lstm')],
    
    id = 'model-types',
    value='linear-portfolio'
        )


layout = html.Div(

    [

    model_tabs,

    html.Div(
        [
        # html.Div(id='portfolio-proportions-pi-chart')
        html.Div([

        ],
        id='pie-chart-containier')
        ]
    )


    ],
    style=CONTENT_STYLE
)



@callback(
    Output(component_id='pie-chart-containier',component_property='children'),
    Input(component_id='model-types',component_property='value'))


def plot_proportions(portfolio_creation_type):


    


    now = dt.datetime.now()
    print(now)
    
    model_type = portfolio_creation_type

    # print(model_type)

    if model_type == 'linear-portfolio':
        # print(model_type)
        figure = visualizer.visualize_portfolio(linear_portfolio)

        return html.Div([dcc.Graph(figure=figure)])

    else:
        # print(model_type)
        return html.Div([])



# def ff():
#     pass

