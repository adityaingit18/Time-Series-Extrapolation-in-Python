# Natural Gas Price Estimation for Arbitrage
- As a quantitative researcher for JP Morgan Chase & Co., I was given a task where I learn that the current process is to take a monthly snapshot of prices from a 
market data provider, which represents the market price of natural gas delivered at the end of each calendar month.

- I have to use this monthly snapshot data to produce a varying picture of the existing price data, as well as an extrapolation for an extra year, in case the client needs an 
indicative price for a longer-term storage contract.


- The main file which the model's script I built on is "Task1 (Final Code).py".
- In the "Task1 (Final Code).py" file, I used libraries like pandas and numpy to work this task out along side the library of matplotlib to visualize.
- In this file, I build a trendline (using np.polyfit) which represents the trend of prices from month to month from 2020 to 2024 with a slope m (representing how on average
the price's trend increased over time).
- I also build a variable named as "monthly_average_deviation" which measures the seasonal deviation of prices from month to month from the trendline and the model uses it to adjust accordingly
to predict the extrapolation and estimation of input date by any user.

# USAGE
This script prompts the user to input a reference date where the whole model takes this date as an input (in MM/DD/YY format) and then estimates the price value at the reference
date and also estimates (forcasts) the value of the price one year forward to the reference input entered by the user.
