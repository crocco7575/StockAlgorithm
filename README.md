## About
The main idea behind this project was to create a **automatic** trading bot, which buys and sells the live market based on various indiactors. To implement this, the main packages we used were:
```python
import alpaca
import yfinance
from talipp.ohlcv import OHLCVFactory
```
(among many others)

## Implementation
Our program isn't actually just "buying" and "selling" stocks. It first "flags" stocks based on the indicators of **TTM Squeeze** and **RSI**, and then watches the stock price (for a maximum of 20 minutes)

