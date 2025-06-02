The purpose of this project is to train multiple machine learning models to learn the volatility of certain stocks given past data and then we can extrapolate this into the future!

Potential Directions for this project: Find the optimal window when computing realized standard deviations. 

Some definitions:

Implied Volatility: When using the BS model, we plug in the current stock price and the current option price. We then solve for the volatility and this is the implied volatility

Garch:

Choices:
The hidden dimension size for the LSTM was initially chosen to be 32, but later changed 
When implementing the LSTM the final 200 entries were used for testing while the initial 3271 entries were used for training