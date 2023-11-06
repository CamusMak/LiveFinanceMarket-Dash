import sys 
# add project path to be able to import portfolio_manager module

sys.path.append("../../LiveFinanceMarket-Dash")



from icecream import ic

import json

from portfolio_manager.portfolio import Portfolio
from portfolio_manager.linear_ts_model_optimization import optimize_arima,optimize_garch

# ic("Done")




# def linear_portfolio():

import pandas as pd

class PortfolioManager:

    def __init__(self,watch_list: dict):

        self.watch_list = watch_list

        self.portfolio_list = []
        pass


    def linear_portfolio(self,
                        n_scotks = 5,
                        BEST_LINEAR_MODEL_PARAM_PATH = 'data/model_data/best_linear_model_params.csv',
                        BEST_LINEAR_MODEL_PARAM_PATH_FULL = 'data/model_data/best_linear_model_params_full.csv',
                        LINEAR_PORTOFLIO_PATH = 'data/portfolio_data/linear_portfolio/linear_portfolio.json',
                        LINEAR_RETURNS_PATH = 'data/portfolio_data/linear_portfolio/linear_returns.csv',
                        LINEAR_RECORDS_PATH = 'data/portfolio_data/linear_portfolio/linear_records.csv',
                        PORTFOLIO_PERFORMACE_PATH = 'data/portfolio_data/linear_portfolio/linear_portfolio_perfomance.json'):
        
        portfolio = Portfolio(watch_list=self.watch_list,
                                   initial_amount=100_000,
                                   best_model_params_path=BEST_LINEAR_MODEL_PARAM_PATH,
                                   portfolio_path=LINEAR_PORTOFLIO_PATH,
                                   records_path=LINEAR_RECORDS_PATH,params_from_path=True,
                                   portfolio_performance_path=PORTFOLIO_PERFORMACE_PATH)
        
        self.portfolio_list.append(portfolio)
        

        return portfolio
    
    def get_all_properties(self,portfolio):

        portfolio.get_all_properties()


    def update_portfolio(self,portfolio,n_stocks=5):

        portfolio.create_portfolio(number_of_stocks=n_stocks,update_portfolio=True)

    def update_all_portfolios(self):

        for portfolio in self.portfolio_list:
            self.update_portfolio(portfolio)




# BEST_LINEAR_MODEL_PARAM_PATH = '../data/model_data/best_linear_model_params.csv'
# LINEAR_PORTOFLIO_PATH = '../data/portfolio_data/linear_portfolio.json'
# LINEAR_RETURNS_PATH = '../data/portfolio_data/linear_returns.csv'

# with open("../data/stock/watch_list.json",'r') as file:

#     watch_list = json.load(file)





# linear_portfolio = Portfolio(tickers=watch_list,initial_amount=100_000,best_model_params_path=BEST_LINEAR_MODEL_PARAM_PATH,params_from_path=True)

# linear_portfolio.create_portfolio(current_portfolio_json_path=LINEAR_PORTOFLIO_PATH,current_returns_path=LINEAR_RETURNS_PATH)

# print(linear_portfolio.portfolio)

# params = pd.read_csv("../data/model_data/best_model_params.csv")

# # print(params.columns)
 
  



# # print(portfolio.best_model_params)

# # portfolio.get_best_liner_model(short_run=True)

# # print(portfolio.next_day_returns)

# # print(type(pm.Portfolio))
# ic("Done!!!")

