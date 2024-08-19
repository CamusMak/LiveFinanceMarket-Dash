# [Python Program for Stock Price Analysis and Portfolio Management](https://live-stock-market.onrender.com/)

This Python project offers tools to visualize historical stock prices, calculate key financial metrics, predict future prices using advanced models, and reallocate portfolio proportions based on these predictions. The project also includes a Dash app for interactive data visualization and portfolio management.

## 1. Historical Price Visualization
- **Objective**: Visualize historical prices of selected stocks over a specified time interval.
- **Metrics**: Calculate and display returns, moving averages, and other financial indicators.
- **Comparison**: Enable the comparison of prices across different stocks.
- **Tools**: Visualizations are created using Matplotlib or Plotly for flexibility and clarity.

## 2. Portfolio Creation
- **Objective**: Build and manage a portfolio of selected stocks.
- **Functionality**: Compare stocks and determine the weight of each in the portfolio.
- **Visualization**: Display the proportions of each stock in the portfolio using a pie chart.

## 3. Price Prediction
- **Objective**: Forecast future stock prices using two distinct approaches.
  - **Linear Models**:
    - **ARIMA**: Captures autocorrelation patterns in stock prices.
    - **GARCH**: Models volatility and time-varying variance.
  - **Neural Network**:
    - **LSTM**: Leverages Long Short-Term Memory networks to capture sequential dependencies in stock data.
- **Implementation**: Each model is implemented in its own class, ensuring modularity and ease of testing.

## 4. Portfolio Reallocation
- **Objective**: Reallocate the portfolio based on predicted future stock prices.
- **Logic**:
  - Use the selected prediction model to forecast future prices.
  - Adjust stock proportions to optimize returns or minimize risk.
  - Update and visualize the portfolio with the new allocation.
- **Visualization**: The updated portfolio allocation is illustrated through a pie chart.

## 5. Dash App
- **Objective**: Provide an interactive, user-friendly interface for the entire process.
- **Features**:
  - Visualize historical stock data, calculate financial metrics, and predict prices.
  - Manage and compare stocks within the portfolio.
  - Display dynamic tables and charts based on user interactions.
- **Deployment**:
  - Push the code to GitHub for version control and collaboration.
  - Deploy the Dash app on Render.com to make it accessible online.

## Conclusion
This project seamlessly integrates stock price visualization, prediction, and portfolio management into a cohesive tool, enhanced with a Dash app for interactive and intuitive user experience.
