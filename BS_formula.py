from scipy.stats import norm
import numpy as np
from prettytable import PrettyTable


# Define the Black-Scholes formula components for an option payoff
def d1(S0, K, r, q, sigma, T):
    return (np.log(S0 / K) + (r -q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S0, K, r, q, sigma, T):
    return (np.log(S0 / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def option_price(S0, r,q , sigma, T, K1, K2, K3, K4):
    # Calculate d1 and d2 for each strike
    d1_K1, d2_K1 = d1(S0, K1, r, q, sigma, T), d2(S0, K1, r, q, sigma, T)
    d1_K2, d2_K2 = d1(S0, K2, r, q, sigma, T), d2(S0, K2, r, q, sigma, T)
    d1_K3, d2_K3 = d1(S0, K3, r, q, sigma, T), d2(S0, K3, r, q, sigma, T)
    d1_K4, d2_K4 = d1(S0, K4, r, q, sigma, T), d2(S0, K4, r, q, sigma, T)

    # Calculate the option price based on the segments
    # Segment 1: K1 to K2
    seg1 = S0 * np.exp(-q * T) * (norm.cdf(-d1_K2) - norm.cdf(-d1_K1)) - K1 * np.exp(-r * T) * (norm.cdf(-d2_K2) - norm.cdf(-d2_K1))

    # Segment 2: K2 to K3 (fixed payoff K2 - K1)
    seg2 = (K2 - K1) * np.exp(-r * T) * (norm.cdf(-d2_K3) - norm.cdf(-d2_K2))

    # Segment 3: K3 to K4
    seg3 = -S0 * np.exp(-q * T) * (norm.cdf(-d1_K4) - norm.cdf(-d1_K3)) + K4 * np.exp(-r * T) * (norm.cdf(-d2_K4) - norm.cdf(-d2_K3))

    # Total option price is the sum of the segments
    return seg1 + seg2 + seg3 * (K2-K1)/(K4-K3)


S0, T, sigma, r, q, K1, K2, K3, K4 = [float(data) for data in input("please insert in the following format:"
                                                                    "\nS0, T, sigma, r, q, K1, K2, K3, K4\n").split(",")]


option_price_val = round(option_price(S0, r, q, sigma, T, K1, K2, K3, K4), 4)

num_simulations = 10000  # Number of simulated asset paths
op_prices = np.array([])
for i in range(20):
    # Monte Carlo simulation of asset paths
    Z = np.random.standard_normal(num_simulations)
    ST = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    # Payoff calculation for each simulation
    payoff = np.piecewise(ST,
                          [ST <= K1, (ST > K1) & (ST <= K2), (ST > K2) & (ST <= K3), (ST > K3) & (ST <= K4), ST > K4],
                          [0, lambda ST: (ST - K1), K2 - K1, lambda ST: ((K4 - ST)/(K4 - K3)) * (K2 - K1), 0])

    # Discount the expected payoff to present value
    option_price_mc = np.exp(-r * T) * np.mean(payoff)
    op_prices = np.append(op_prices, option_price_mc)
mean, sd = op_prices.mean(), op_prices.std()
ci_mc = (round(mean - 2 * sd, 4), round(mean + 2 * sd, 4))

input_table = PrettyTable()
input_table.field_names = ['S0', 'T', 'sigma', 'r', 'q', 'K1', 'K2', 'K3', 'K4']
input_table.add_row([S0, T, sigma, r, q, K1, K2, K3, K4])

output_table = PrettyTable()
output_table.field_names = ["Close Form", "Monte Carlo Simulation"]
output_table.add_row(
    [option_price_val, ci_mc]
)

print("Your Inputs are:\n")
print(input_table)
print("Your Outputs are:\n")
print(output_table)