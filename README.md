# TASK 1: Natural Gas Price Estimation for Arbitrage
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

# TASK 2: Price a Commodity Storage Contract
- The client wants to start trading as soon as possible. They believe the winter will be colder than expected, so they want to buy gas now to store and sell in winter in order to take advantage of the resulting increase in gas prices. They ask me to write a script that they can use to price the contract.
- I have to create a prototype pricing model that can go through further validation and testing before being put into production. this model may be the basis for fully automated quoting to clients, but for now, the desk will use it with manual oversight to explore options with the client.

# USAGE (TASK 2)
- I wrote the python code for this model in the "Task2.py" file, where the client can enter his inputs (multiple inputs) as follows:
(1) Injection dates,
(2) Withdrawal dates,
(3) The prices at which the commodity can be purchased/sold on those dates,
(4) The rate at which the gas can be injected/withdrawn,
(5) The maximum volume that can be stored,
(6) Storage costs.

and it calculates the final value of the contract according to the plans of client.

# TASK 3: Credit Risk Analysis & Expected Loss Model
- As a quantitative researcher, the risk management team provided me with a tabular dataset containing historical borrower data (Income, FICO scores, outstanding debt, employment history, and default status). 
- I was tasked with building a machine learning classification model to predict the Probability of Default (PD) for new loan applicants and calculate the Expected Bank Loss based on a 10% recovery rate assumption.
- The Python script for this model is located in the "Task3.py" file.
- In this file, I utilized `pandas` for data wrangling and `scikit-learn` to build the predictive models. I applied `StandardScaler` to normalize features with vastly different magnitudes (e.g., $80,000 income vs. 650 FICO score) to prevent data leakage and magnitude bias.
- I conducted a Comparative Analysis between a Logistic Regression model and a Random Forest Classifier, ultimately selecting Logistic Regression for its highly calibrated probabilistic outputs. 

# USAGE (TASK 3)
The script features an interactive terminal application using a `while True:` loop. A bank underwriter can input a prospective borrower's financial details (FICO, debt, income, etc.) live in the terminal. The model instantly scales the input, predicts the Probability of Default (PD), and outputs a professional financial ledger calculating the exact Expected Loss in dollars.

# TASK 4: FICO Score Quantization (Bucketing Algorithm)
- The risk manager requested that the continuous FICO scores from our dataset be converted into discrete risk tiers (buckets) to establish a practical, real-world banking policy for assigning interest rates.
- A naive approach of slicing FICO scores into equal numerical intervals is flawed because risk scales non-linearly (the risk difference between a 500 and 600 FICO is massive, whereas 750 and 850 is minimal).
- I wrote the solution in the "Task4.py" file, employing a Quantitative "Split-Domain" approach. Because a FICO score of 600 is a critical threshold separating subprime from prime borrowers, I split the dataset into two sub-populations (< 600 and >= 600).
- I utilized a `DecisionTreeRegressor` (optimizing for Mean Squared Error) on both sub-populations to mathematically determine the most optimal boundary lines where default rates drop the most drastically.
- The algorithm stitches these boundaries together to create a master 10-Tier Risk Policy.

# USAGE (TASK 4)
When executed, the script automatically parses the historical dataset, calculates the optimized boundaries, and prints a formatted 10-Tier Bank Policy ledger to the console, detailing the FICO range, borrower count, and exact default rate per tier. Additionally, it uses `matplotlib` to automatically generate and save a high-resolution visual dashboard (`FICO_10_Tier_Dashboard.png`) displaying the FICO distribution, the algorithm's boundary lines, and a color-coded bar chart of tier risk.
