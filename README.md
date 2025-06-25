Project Overview:
Develops LSTM-based volatility forecasts to identify mispriced European call options via
- Comparative analysis against GARCH benchmarks
- Systematic trading strategy backtesting
- Rigorous Monte Carlo significance testing

Motivation: Predicting volatility better than market results in optimal options pricing. Capitalize on mispriced options.
Followed process of Data Generation, LSTM Model Training, Trade Signal Generation, Backtesting Engine, Risk Management, Performance Analysis

Methodology
Volatility Forecasting
LSTM Model
  - 32 hidden units (optimized using time series validation)
  - Input: 30-day rolling window of SPY (S&P 500) and implied volatility data
  - Output: 21-day volatility forecast
Benchmark Models: GARCH(1,1), Historical Volatility

2. Trading Strategy
Modified generate signal functions multiple times:
  - Outputs "long-straddle" signal if predicted volatility above dynamic threshold computed based on rolling averages of differences between actual option price and predicted option price
  - Dynamic signal sent only if spread threshold in upper 25% of past signals in last 30 days
  - Reinforced trades by relying on market knowledge through implied volatility
  - Area of future focus: Implement short straddle signal after sufficient testing on threshold for dispatch of signal

3. Backteseting Framework:
Key Features:
  - Realistic bid-ask spreads (0.05-0.10% of notional)
  - Margin requirements (20% of position size)
  - Transaction cost modeling including commission and slippage costs.

Results and Evolution:
Assumed 100k starting portfolio size.
Key Performance Metrics:
Date	  PnL	    Win Rate	Sharpe	Max DD	Profit Factor
6/5/25	-$46.04	44.2%	    -	      -	      0.76
6/6/25	$268.80	44.9%	    0.25	  -$1,181	1.42
6/16/25	$109k	  46.0%	    0.53	  -$112k	1.14

Key Findings:
- LSTM Superiority: 19% lower RMSE vs GARCH in volatility forecasting
- Strategy Sensitivity:
  - Optimal holding period: 14-21 days
  - Critical filters: VIX > 0.9×20D MA, IV trend > 0, added to increase chance of profitable trade execution.
 
Statistical Validation:
Implemented crude Monte Carlo Simulation with 10,000 trials and improved method to consider time-series nature of model. 
Further Statistical Analysis continued to be worked on.

Lessons Learned:
1. Critical Insights
  - Data integrity, eliminated look ahead bias present early subtly through the reuse of minmax normalizers. Introduced bias and unrealistic returns (16.1 profit factor) initially.
  - Fixed position sizing errors. Initially double counted position sizes allowing riskier decisions with no backing. These decisions were further supported by the biased data.
  - Throughouly cleaned strategy and code to ensure that these errors were handled and that position sizes were computed once, based on current portfolio size rather than overall portoflio size (which included winning trades).
  - Reducing frequency of trading resulted in less data to perform statistical tests on leading to insignificant results. Future implementations will account for this.
2. Market Insights
  - Short straddles abandonded temporarily due to risk as they require delta hedging. Future improvements will implement delta hedging to include short straddles.
  - Bid-ask spread consumes 15-20% of expected prodits. These were considered further on which made net PnL more realistic.

Future Improvements:
Signal Optimization
- Dynamic thresholding based on IV percentile
- Volatility regime filters
Further Directions:
- Implement robust MC testing to handle bigger data sets
- Implement costum derivative pricer by solving BS PDE numerically to extend project to American and Asian options.
