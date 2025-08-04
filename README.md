# Financial Computation Models

A Python project implementing various option pricing models using both closed-form solutions and numerical methods.  
Covers Black-Scholes, Binomial Tree, Monte Carlo simulations, and advanced exotic options such as Asian, Rainbow, and Lookback options.

---

## Core Files (Before Refactoring)

Below are the first five core files before the project refactoring, with their purposes:

1. **black_scholes.py**  
   - Implements the **Black-Scholes closed-form solution** for European call and put options.  
   - Serves as a benchmark for comparing numerical methods.

2. **binomial_tree.py**  
   - Implements the **Cox-Ross-Rubinstein (CRR) binomial tree model**.  
   - Supports both **European** and **American** call/put options.

3. **asian_option.py**  
   - Prices **Asian options** using Monte Carlo simulation.  
   - Handles both **arithmetic** and **geometric** averaging methods.

4. **rainbow_option.py**  
   - Prices **Rainbow options** (multi-asset options) using Monte Carlo simulation.  
   - Implements **Cholesky decomposition** for correlated asset price generation.

5. **lookback_option.py**  
   - Prices **Lookback options** using Monte Carlo simulation.  
   - Tracks the **maximum or minimum** asset price during the option's life.

---

## Project Structure
```bash
financial_computation/
├── main.py # Main entry point, runs all models and outputs results
├── models/ # Option pricing model implementations
│ ├── black_scholes.py
│ ├── binomial_tree.py
│ ├── asian_option.py
│ ├── rainbow_option.py
│ ├── lookback_option.py
├── utils/ # Shared helper functions
│ ├── option_math.py # Math formulas (d1/d2, combinations, etc.)
│ ├── tree_builder.py # Stock price tree generation
│ ├── monte_carlo.py # Monte Carlo helpers (confidence intervals, etc.)
├── requirements.txt # Required Python packages
├── README.md
└── .gitignore
```


---

## Installation
```bash
git clone https://github.com/<your-username>/financial_computation.git
cd financial_computation
pip install -r requirements.txt
```

Usage
```bash
python main.py
Example Output
Example run produces a summary table comparing closed-form, binomial, and Monte Carlo prices for different option types.
```
Requirements
Python 3.8+

numpy

scipy

prettytable
