import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from icecream import ic

import sys
sys.path.append("../../LiveFinanceMarket-Dash")


from portfolio_manager.portfolio import Portfolio


class PortfolioVisualizer:

    def __init__(self):
        pass


    @classmethod
    def pie_chart(cls,portfolio):

        df = portfolio.proportions_df

        figure = px.pie(df,names='symbol',values='proportion')

        return figure


    @classmethod
    def perfomance_line_chart(cls,portfolio):

        pp = portfolio.portfolio_performance

        figure = make_subplots(1,2,horizontal_spacing=0.3)

        figure.add_trace(
            go.Scatter(
            x=pp['date'],
            y=pp['portfolio_return'],
            name='Portoflio profitability'),
            row=1,
            col=1
        )


        figure.add_trace(
            go.Scatter(
            x=pp['date'],
            y=pp['total_amount'],
            name='Portoflio monetary value'),
            row=1,
            col=2
        )
        figure.update_layout(template=pio.templates['plotly_white'])

        figure.update_xaxes(title_text="Date/datetime",row=1,col=1)
        figure.update_xaxes(title_text="Date/datetime",row=1,col=2)


        figure.update_yaxes(title_text="Return",row=1,col=1)
        figure.update_yaxes(title_text="Total value",row=1,col=2)


        figure.update_layout(width = 1500,
                            height = 800,
                            title='Portfolio return(%) and total amount changes over time')


        return figure
    
    @classmethod
    def visualize_portfolio(cls,portfolio):

        pp  = portfolio.portfolio_performance
        df = portfolio.proportions_df.dropna()

        figure = make_subplots(2,2,
                       horizontal_spacing=0.3,
                       vertical_spacing=.2,
                       specs=[[{"colspan": 2,"type":"pie"},None],[{},{}]],
                       subplot_titles=["Stock & cash proportions in portfolio<br> ",
                                       "Portfolio's return(%) change over time<br> ",
                                       "Portfolio's total amount change over time<br> "])



        figure.add_trace(
            go.Pie(
                labels=df['stock_name'],
                values=df['proportion']
            ),
            row=1,
            col=1
        )

        figure.add_trace(
            go.Scatter(
            x=pp['date'],
            y=pp['portfolio_return'],
            name='Portoflio <br>profitability'),
            row=2,
            col=1
        )


        figure.add_trace(
            go.Scatter(
            x=pp['date'],
            y=pp['total_amount'],
            name='Portoflio <br>monetary value'),
            row=2,
            col=2
        )
        figure.update_layout(template=pio.templates['plotly_white'])

        figure.update_xaxes(title_text="Date/datetime",row=2,col=1)
        figure.update_xaxes(title_text="Date/datetime",row=2,col=2)


        figure.update_yaxes(title_text="Return",row=2,col=1)
        figure.update_yaxes(title_text="Total value",row=2,col=2)


        figure.update_layout(width = 1200,
                            height = 700,
                            margin=dict(l=100, r=100, t=100, b=100))



        return figure



