## About/Setup
The main idea behind this project was to create a **automatic** trading bot, which buys and sells the live market based on various indicators. To implement this, the main imports we used were:
```python
import alpaca
import yfinance
from talipp.ohlcv import OHLCVFactory
```
(among many others)

## Implementation
* Our program isn't actually just "buying" and "selling" stocks. It first "flags" stocks based on the indicators of **TTM Squeeze** and **RSI**, and then watches the stock price (for a maximum of 20 minutes) to reach a value of ```buy_spacer * flag_price ```. If the ticker reaches this trigger price, __THEN__ we actually buy the stock. 
* Our sell condition is determined by two values, ```normal_stop_loss_constant``` and ```trailing_stop_loss_constant```. If the stock price is trending downwards after our "buy", the maximum it can drop before our program automatically sells it is determined by ```buy_price * (1 - normal_stop_loss_constant)  ```. If the stock price starts rising after our "buy", ```trailing_stop_loss_constant``` is used to protect our $$ that we just made off that trade. If a stock price starts dropping **AFTER** rising, the maximum the stock can drop is ```max_price * (1 - trailing_stop_loss_constant)```, where ```max_price``` is the maximum price the stock reached while we were holding it.
* Our main buy/flag program takes roughly 11-15 minutes to run, so we have a 

To better understand the project, I broke it down into segments.
1. Downloading all tickers
2. 

