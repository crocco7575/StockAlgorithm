## Table of Contents

1. [Background](#background)
2. [Requirements](#requirements)
3. [Live Trading-no ML](#live-trading-no-ML)
4. [Dataset Experimentation](#dataset-experimentation)
5. [Machine Learning Experimentation](#machine-learning-experimentation)
## Background

The main idea behind this project was to create a **automatic** trading bot, which buys and sells the live market based on various indicators and a neural network's prediction.  There are currently 3 different segments to this project: **Live Trading**, **Dataset Experimentation**, and **Machine Learning Experimentation**. The first portion of this project was focused on live trading through the Alpaca API using a simple indicator algorithm (no ML). Once we created a working algorithm, we noticed our selection of stocks was pretty poor. So, we began an experimental period where we ran trials for three adjustable values: buy spacer, stop loss, and trailing stop loss, to find the most profitable combination. We then had the idea to train a neural network to recognize and filter out poor performing trades. We experimented with various types of Gradient Boosting Machines to improve accuracy and then looked into various ways to scale our dataset. We are currently in the process of improving our dataset, and implementing our neural network in live trading. 

## Requirements


The following are the main imports for each section of this project. Note many helper imports were not included, for sake of brevity.

Live Trading:
```python
import alpaca
import yfinance
from talipp.ohlcv import OHLCVFactory
import schedule
```
Algorithm Optimization:
```python
import yfinance as yf
from talipp.ohlcv import OHLCVFactory
import talipp.indicators as ta
```

ML Dataset Generation:
```python
from talipp.ohlcv import OHLCVFactory
import matplotlib.pyplot
import numpy
import talipp.indicators as ta
from scipy.optimize import fsolve
from polygon import RESTClient
```

ML Experimentation:
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
```


## Live Trading - no ML

-  Our foundational algorithm contains 3 main features
	- Flagging Algorithm
	- Watch Algorithm
	- Selling Algorithm
#### Flagging Algorithm

- The main purpose of the flagging algorithm is to add potential volatile stocks to the "watchlist" (```watchlist.txt```)
- This algorithm runs through every ticker in the `condensed_tickers.txt` 
	- `condensed_tickers.txt` is a list of all current NASDAQ and SMP 500 stocks
- There are a lot of helper functions, but for this README we are only dive into the main functions, which are the basis of our algorithm
- Our flagging algorithm is based off 2 well known indicators for volatile activity: TTM Squeeze and RSI
	- `TTM_Squeeze_values()` and `rsi_values()` retrieve the last 15 minutes of TTM Squeeze and RSI data, respectively, for subsequent computation and analysis
	- ```check_rsi()``` is responsible for handing the RSI from `rsi_values()` to determine whether the stock fits our 'volatile' qualification
		- The volatile qualification for RSI is any dip under **30** 
	- `check_ttm()` is a bit more complicated than `check_rsi()`, because it takes TTM values and computes small **slopes** of the TTM graph
		- If the graph is on the earlier phase of a concave up graph, that will trigger a flag signal
		- Here is a diagram of what this looks like:
		- **![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfcLPgWU1AIRcA-iQnH_DcdHisJm_6IxP6XZnKrSR7PwBawSxj2WYu_toJflk0IKNob9__X_T8LsLeoYr6_8Kl4LFYEgZIPXsiIYdN6TufwPh5X6VUOTuRHUDOB2H7D6dYf2_8c-VnVy8qu6kYGvmalp2GT?key=YZDkUA69wi5Eck5_G4panQ)
	- The main function performing tasks in flagging algorithm is `main_BUY_function`
		- It checks whether `check_ttm` and `check_rsi` both triggered a flag signal, 
				- If we don't currently own the stock in question, we call `add_watchlist()`
				- `add_watchlist()` appends the ticker symbol to `watchlist.txt`, current price data to`prices.txt`, and time of flag to `timers.txt` , all in prep for the watchlist algorithm
	- This code takes about 10-15 minutes to run on average, so during the open hours of the day it's on a timer to run every 20 minutes

#### Watch Algorithm

- The main purpose of the watch algorithm is to examine prices of the flagged stocks to see if they begin to rise, and if they do we place an order through the Alpaca API
- `main_watch_function()` has the main logic of the algorithm
	- The `place_order()` function is triggered when the stock's current price exceeds the original flag price multiplied by the `buy_spacer` factor.
		- The `place_order()` function is the most technically involved piece of our logic
			- A factor is determined by the current performance of the NASDAQ and SMP 500 in `nasdaq_smp()` 
				- negative factor = poor performance
				- positive factor = positive performance
			- This factor is multiplied to `buy_cap_base`, a constant used to devise our $100k (paper money) into individual stocks, to obtain `buy_cap`, or the total amount of money we are going to use to buy the stock
				- The result of this is:
					- Bad SMP/NASDAQ performance = less money invested
					- Good SMP/NASDAQ performance = more money invested
				- An order is placed through Alpaca with the ticker and the number of quantities we are going to buy (`current_price / buy_cap`)
			- Once purchased, the stock and all its values are removed from the watchlist 
		- `main_watch_function()` also removes stocks from the watchlist if their timer expired
			- Because the indicators present short term changes, flagged stocks on the watchlist for more than 20 minutes may have moved to a completely different, erratic pattern
- This code takes about 45 seconds to run, so it is on a timer to run every minute during the trading hours of the day
#### Selling Algorithm

- The main purpose of the selling algorithm is to minimize losses when holdings are down and maximize our profit when holdings are up, by selling at strategic positions
- There are two pieces to the selling algorithm
	- Recurring sell
	- Final sell
- The final sell logic, contained in `finalSell.py` is the simplest portion of our live trading algorithm.
	- Every day at 3:55 PM EST, all positions are sold from our Alpaca account
	- `finalSell.py` prevents any possible overnight holdings, as this is a **day-trading algorithm** 
- The recurring sell logic occurs in `main_SELL_function()`, and is based off two values, `normal_stop_loss_constant` and `trailing_stop_loss_constant`
	- The function iterates through every stock that is currently owned
	- If the current stock price is below the original buy price, the stock is sold`normal_stop_loss_constant` multiplied by the original price
	- If the current stock price is above the original buy price, the stock is sold at `trailing_stop_loss_constant` multiplied by the **current price**
	- Here is a diagram to better explain this:
	- **![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeuJJhsxKZuJqJTvoSNRlB0i_l1wVTiMaICWrA7JqWIij1BhGf6jzt_YWrS7WBCQnFYO0AQ_x9OJ012-u0y6GxPsxTYqWbJfv9bpYDxR4cpoX7qPScC5FcQxEnf_al6J8Du4V8Uxhc-8-pUGehrkebNdX2i?key=YZDkUA69wi5Eck5_G4panQ)**
	- The sell logic takes about 10-35 seconds to run, depending on the number of current positions, and is run every minute during the live trading hours of the day
## Dataset Experimentation

- The main purposes of the dataset experimentation was to strengthen the existing algorithm and to get a large dataset to train our machine learning models
- There are two different experiments that we conducted for datasets:
	- Algorithm Improvement
	- ML Dataset Collection
#### Algorithm Improvement

- The purpose of this experiment was find optimal values for the three adjustable parameters in our foundational algorithm:
	- **Buy Spacer:** a critical component of the watch algorithm, serving as a multiplier to the flagged stock price. It acts as a threshold that the stock price must exceed to trigger an order placement.
	- **Trailing Stop Loss:** implemented in the sell algorithm, it is a multiplier of the **current price** used as a safety mechanism to conserve gains on a stock
	- **Normal Stop Loss:** implemented in the sell algorithm, it is a multiplier of the **original flag price** used as a sell threshold to conserve our losses on a stock. It is only used when the stock is below the original flag price.
- For each param, 20-25 values were plugged into the algorithm, and tested on the same, random week from the past
	- `yfinance` was utilized to get minute-by-minute data from a random week in the past
	- A new, combined file was created to condense our foundational algorithm
		- Each file ran a week at a time
	- 5-10 versions of that same file ran at the same time to expedite the data collecting process
	- Each python file produced 5 excel files (one for every day in the week), containing a list of trades made in that day along with information on each trade (the most important piece was profit/loss)
	- We tallied the total money lost/won for each day, and averaged the 5 days for a final profit gain/loss for that value
	- Each param had a `average.xlsx` where we plotted the averages from each tested value
	- We fit a function of best fit on each graph, and the maximum of that function was the "optimal value" that was going to be used in live trading
- Here are the three graphs that were created. Note the value on the function yielding the highest P/L was implemented in our algorithm
	- **![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcvGW1PNqRKl64-VEq2_UBp_tBj1X8-13HUYi1bN-mNXhrANbMidPeWFoQ2mM5BoxLCo-0gLGbHoDKYYImjhtVIEka1Wzz0CjJdMe0g0CyL1G8PT_ob4T50KA2Kdss-1N7WxYGErcBuPF1QXPzBWblJhkky?key=txX16itHtXxl4LvNg84HRQ "Chart")**
	- **![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdBiE2kKt-_Erj_O4mVhbzSjz6LbiFe6kh_J5Sx9zw79c3xrsD0CT-4bOhZoxa3RSaXqXxclR5svEUmmgicaJbG18idRIheO3S92RrweND4ZXgi94pFb1D2u1xP99fLE80gGMBOxwrhnrboeh42j8IvVEUj?key=txX16itHtXxl4LvNg84HRQ "Chart")**
	- **![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXePu-rQcMZLSVvBCt9yvEY6-QTuN3nnr1m9Q3IJwICU8UV_JHLBHE-8wAj4HWzGyaejXqmEWIkXwkbSKDOHdxg_eZ7GdZ0Q1xvOPa71odJQ7zFr-VoT6sjU6aTm8hXUYRoeg9vSMqOKLkI91rfO5Po4DaK1?key=txX16itHtXxl4LvNg84HRQ "Chart")

#### ML Dataset Collection

- The purpose of this experiment was to expand the training dataset, and obtain higher accuracy from our neural network
	- The way this was done was by getting more trades from our algorithm, which meant running our algorithm on more days
- There were two ways we attempted to get more trades
	- Polygon.io
	- Synthetic Generation
- Polygon.io is a trading API and platform that allows you to get historical, minute-by-minute data from the stock market
	- There is a paywall to get historical data, and the more you pay the more historical data you can get (up to 15+ years)
	- It is pretty expensive, so we started with just 5 years of historical data for $29/month
	- A new file was created to run the fundamental algorithm on this polygon data
		- The algorithm was able to produce 156,000 trades from this historical, minute-by-minute data, which was used in training of our neural networks
		- XGBoost was able to get **85% accuracy** predicting stocks going over/under -2% from this 156k dataset
- This data ended up being the most reliable, so we are currently working on getting funds for the higher tier dataset (10-15 years of data) 
- The other attempted method to collect a ML dataset was synthetic generation of trades
- 



## Machine Learning Experimentation

- There are two Gradient Boosting Machine (GBM) neural networks that were experimented with
	- Random Forest Classifier
	- XGBoost
	- Why GBMs?
		- They are proven to handle complex numeric tasks extraordinarily well, and are the industry standard for these kind of projects
- Both experiments had identical setups, the only thing different was how the models were imported
	- `P/L (%)` column was given a binary label
		- "0" if the P/L was under -2%
		- "1" if the P/L was over -2%
	- Multi-Class labeling for `P/L (%)` was also attempted, but accuracy was extremely low (35-40%)
	- The following were the feature columns used to predict this "binary label":
		- Flag Price, Buy Price, Flag Time, Buy Time, Sell Time, Time Between Flag and Buy, Lowest RSI, TTM Strength
	- After training, we used the accuracy score function from `sklearn.metrics` to obtain accuracy for our models
		- We achieved up to **85% accuracy** using the polygon data and the XGBoost neural network
- XGBoost was much faster and more accurate than the Random Forest Classifier (as it allowed for GPU optimization), so the majority of our work was using XGBoost
- Currently working to implement the ML model prediction into our live trading algorithm
	- It will go into the watch algorithm as a final parameter before placing an order
