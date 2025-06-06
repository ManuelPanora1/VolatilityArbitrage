import numpy as np
import matplotlib.pyplot as plt

def monte_carlo(returns, n_sim=10000):
    sharpe_ratios = []
    for _ in range(n_sim):
        simulated = np.random.permutation(returns)  
        sr = np.mean(simulated) / np.std(simulated) * np.sqrt(252)
        sharpe_ratios.append(sr)
    
    actual_sr = np.mean(returns) / np.std(returns) * np.sqrt(252)
    p_value = (np.array(sharpe_ratios) >= actual_sr).mean()
    
    """
    plt.figure(figsize=(10, 6))
    plt.hist(sharpe_ratios, bins=20, alpha=0.7, label='Random Simulations')
    plt.axvline(actual_sr, color='red', linestyle='--', label=f'Your Strategy (p={p_value:.3f})')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    """
    
    return p_value


