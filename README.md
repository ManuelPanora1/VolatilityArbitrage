The purpose of this project is to train multiple machine learning models to learn the volatility of certain stocks given past data and then we can extrapolate this into the future! 

Potential Directions for this project: Find the optimal window when computing realized standard deviations. 

Some definitions:

Implied Volatility: When using the BS model, we plug in the current stock price and the current option price. We then solve for the volatility and this is the implied volatility

Choices:
The hidden dimension size for the LSTM was initially chosen to be 32, but later changed 
When implementing the LSTM the final 200 entries were used for testing while the initial 3271 entries were used for training


Conclusion and Results: 

In the first step of this project I compared LSTM predictions of volatility to the GARCH model with incredible results! The LSTM was far better than the GARCH model.


Further Directions: The rolling window for the volatility, we should test that right now and see which is the best! And we need to define what is best in our case! We can also test the lookback and see which one is the best.

6/5/25 Current Results: 
Strategy Metrics:
Total PnL: $-46.04
Win Rate: 44.2%
Profit Factor: 0.76