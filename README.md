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

6/5/25 Results: 
Strategy Metrics:
Total PnL: $-46.04
Win Rate: 44.2%
Profit Factor: 0.76

changes:
Increased holding period to 21 days

6/5/25 Results:
Strategy Metrics:
Total PnL: $-392.80
Win Rate: 38.4%
Profit Factor: 0.44

changes:
fixed bug with implied volatility

6/6/25 Results:
Strategy Metrics:
Total PnL: $268.80
Win Rate: 44.9%
Profit Factor: 1.42

Implement a monte carlo simulation to ensure that this is not just a fluke but instead statistically repeatable.
Null Hypothesis H0: This strategys returns are no better than random
Alternative Hypothesis H1: This strategy has genuine predictive power
Method of testing: Simulate 10,000 random versions of this strategy and establish a distribution of what's possible by chance. The p value for this test is p = .05 which means that we reject the null if we have a p value of .05 which is interpreted as only 5% of random simulations performed as well as my strategy

6/6/25 p-value = 0.3266 this is not statistically significant, maybe our model overfits. This is our first p-value computed with the standard LSTM with 32 hidden units.

6/7/25 TODO: The model does not suffer from overfitting, this means that there is likely poor strategy and is likely a result of only binary signals for trading. Learn Sharep Ratio and formalize the hypothesis testing!

6/13/25 Now why exactly are we getting some incredible returns? What is happening? I think it is in the way that I am computing things, so I will see what happens
Some crazy results today seen below, but all are a result from double counting the position sizes. Flaws with strategy and implemenation that led to this result: calculated position size twice, and in one of them calculated position size using realized gains, so we were using 10% of what we already earned. The reason we need to be careful about calculating the position size is that we may overestimate how much we actually have so we have to include the margin

Learned today:
Margin: Margin is the amount of capital that you must put up as collateral when taking a leveraged position, for example when buying options or futures. You don't have to pay the full notational value for the option, instead you purchase a fraction of it and that is the margin requirement. Then the position size depends on how much you are willing to risk divided by the marginal requirement per contract to avoid overleveraging. 

Strategy Metrics:
Total PnL: $1474560.04
Win Rate: 63.0%
Profit Factor: 16.33

6/16/25 
Learned Today:
Bid-Ask spread: A certain asset has its theoretical price. The bid for that asset is the amount of a buyer is willing to purchase it for. The spread is the amount a seller is willing to sell it for. The spread is the difference in these two quantities and this can be seen as the transaction cost since you typically lose around half of the spread. The ask is the highest price a buyer is willing to offer while the big is the lowest price the seller is currently willing to accept. This estimates the actual execution price based on liquidity.

Areas of focus (Why is the current model 1:Generating unrealistic results and 2:Having inconsistent p-values with MC testing):

"""Data Leakage"""
Check if LSTM sees future data during training:
Source of data leakage: Fitting scalar to all data then refitting it to just the testing data which results in invalid comparison. 
Verify time-series cross-validation is strictly walk-forward

Overfitting in Signal Generation
Signals may be too perfectly tuned to historical data
Test with simpler models (moving average of volatility)

Survivorship Bias
Are you using current option chains or historical ones that survived?
Missing delisted/expired options skew results

Improved Monte Carlo Testing

By fixing the data leakage and increasing the testing data by reducing the training data I was able to generate the following results, we can clearly see that the win factor and profit factor are much more realistic and reasonable bu the pval is really low, meaning that we should check our pval code: 
Strategy Metrics:
Total PnL: $109,957.70
Win Rate: 46.0%
Profit Factor: 1.14
Max Drawdown: $-112,048.53
Sharpe Ratio: 0.53
Total Costs: $686,961.88
This is your coveted pval: (Moment of truth)
0.9706

From another run of the same thing on main i get this, it is inconsistent. We will see what happens.:
Strategy Metrics:
Total PnL: $94,234.79
Win Rate: 45.6%
Profit Factor: 1.12
Max Drawdown: $-105,312.03
Sharpe Ratio: 0.46
Total Costs: $666,040.66

This is your coveted pval: (Moment of truth)
0.0069

6/17/25 Begin work on implementing short, maybe that will improve the results. Now we are losing like crazy, how can I improve this and make this statistically significant? I need to ensure that the data I am using is correct.

Issues to address:

Not worry about short straddles, they are too risky for now.

One source of testing in the future is creating the optimal signal threshold.

Here are those results: 
Strategy Metrics:
Total PnL: $-3,146.37
Win Rate: 38.9%
Profit Factor: 0.69
Max Drawdown: $-5,341.47
Sharpe Ratio: -2.57
Total Costs: $3,681.60

Meta information
We are already trading when we have a certain spread computed using our predicted volatility, but we are also using the market data as implied volatility is not random and we are using this to reinforce our decision to purchase or not purchase. By checking the 

Learned Today:

We finally have strategy that goes positive for some time and it is a result of (trading when implied volatility shows upward momentum and is greater compared to its recent previous average) here are the outcomes. One issue is that we only entered around 11 trades during this time:
Strategy Metrics:
Total PnL: $-1,584.34
Win Rate: 36.4%
Profit Factor: 0.61
Max Drawdown: $-2,143.75
Sharpe Ratio: -3.43
Total Costs: $1,149.43
