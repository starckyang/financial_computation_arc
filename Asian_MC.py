from scipy.stats import norm
import numpy as np
from prettytable import PrettyTable


# inputs
S0 = np.array([
    172, 173, 169, 165, 170
])
var_mat = np.array([
    [0.3, 0.25, 0.1, 0.2, 0.15],
    [0.25, 0.4, 0.22, 0.25, 0.08],
    [0.1, 0.22, 0.8, 0.29, 0.08],
    [0.2, 0.25, 0.29, 0.6, 0.17],
    [0.15, 0.08, 0.08, 0.17, 0.5]
])
K, r, T, N_SIM, B, n = 140, 0.02, 1, 100, 5, 5


def chlo_dec(matrix):
    size = matrix.shape[0]
    A = np.zeros([size, size])
    # step1
    for i in range(size):
        if i == 0:
            A[i, 0] = matrix[0, 0] ** (1/2)
        else:
            A[i, 0] = matrix[i, 0] / A[0, 0]
    # step2&3
    for i in range(size-1):
        if i != 0:
            d_sum = np.sum(A[i, :i] * A[i, :i])
            A[i, i] = (matrix[i, i] - d_sum) ** (1/2)
            for k in range(i):
                d_sum2 = np.sum(A[i, :i] * A[k, :i])
                A[k, i] = A[i, i] ** (-1) * (matrix[k, i] - d_sum2)
    d_sum3 = np.sum(A[size - 1, :size] * A[size - 1, :size])
    A[size - 1, size - 1] = (matrix[size - 1, size - 1] - d_sum3) ** (1 / 2)
    return A.T

# A0 = chlo_dec(var_mat)
# sim_avg = np.array([])
# for sims in range(B):
#     Z = np.random.normal(0, 1, (N_SIM, n))
#     data = Z.dot(A0)
#     E_ST = S0 * np.exp(r*T)
#     ST = data + E_ST - K
#     zero_column = np.zeros((N_SIM, 1))
#     ST = np.hstack((ST, zero_column))
#     max_values = np.amax(ST, axis=1)
#     max_values = max_values.reshape(N_SIM, 1)
#     avg_val = np.mean(max_values)
#     sim_avg = np.append(sim_avg, avg_val)
#
# final_mean = np.mean(sim_avg)
# final_std = np.std(sim_avg)
# print((round(final_mean - 1.96 * final_std, 4), round(final_mean + 1.96 * final_std, 4)))

def rainbow_option_pricing(S0, var_mat, K, r, T, N_SIM, B, n):
    """
    Calculate the rainbow option price using Monte Carlo simulation.

    Parameters:
        S0 (float): Initial asset price.
        var_mat (ndarray): Variance matrix.
        K (float): Strike price.
        r (float): Risk-free interest rate.
        T (float): Time to maturity.
        N_SIM (int): Number of simulations.
        B (int): Number of batches.
        n (int): Number of assets.

    Returns:
        tuple: Tuple containing the confidence interval for the option price.
    """
    A0 = chlo_dec(var_mat)
    sim_avg = np.array([])

    for _ in range(B):
        Z = np.random.normal(0, 1, (N_SIM, n))
        data = Z.dot(A0)
        E_ST = S0 * np.exp(r * T)
        ST = data + E_ST - K
        zero_column = np.zeros((N_SIM, 1))
        ST = np.hstack((ST, zero_column))
        max_values = np.amax(ST, axis=1)
        max_values = max_values.reshape(N_SIM, 1)
        avg_val = np.mean(max_values)
        sim_avg = np.append(sim_avg, avg_val)

    final_mean = np.mean(sim_avg)
    final_std = np.std(sim_avg)

    return (round(final_mean - 1.96 * final_std, 4), round(final_mean + 1.96 * final_std, 4))

print(rainbow_option_pricing(S0, var_mat, K, r, T, N_SIM, B, n))
# antithetic variate approach and moment matching method

sim_avg = np.array([])
for sims in range(B):
    ori_sim = np.random.normal(0, 1, (N_SIM, n))
    # antithetic approach
    neg_sim = -ori_sim
    anti_sim = np.vstack((ori_sim, neg_sim))
    # moment matching
    for i in range(n):
        cur_std = np.std(anti_sim[:, i])
        anti_sim[:, i] = anti_sim[:, i] / cur_std
    data = anti_sim.dot(A0)
    E_ST = S0 * np.exp(r * T)
    ST = data + E_ST - K
    zero_column = np.zeros((N_SIM*2, 1))
    ST = np.hstack((ST, zero_column))
    max_values = np.amax(ST, axis=1)
    max_values = max_values.reshape(N_SIM*2, 1)
    avg_val = np.mean(max_values)
    sim_avg = np.append(sim_avg, avg_val)

final_mean = np.mean(sim_avg)
final_std = np.std(sim_avg)
print((round(final_mean - 1.96 * final_std, 4), round(final_mean + 1.96 * final_std, 4)))

# inverted Chloesky method
sim_avg = np.array([])
for sims in range(B):
    ori_sim = np.random.normal(0, 1, (N_SIM, n))
    cov_mat = np.cov(ori_sim)
    cov_A = np.linalg.inv(chlo_dec(cov_mat))
    nor_sim = cov_A.dot(ori_sim)
    data = nor_sim.dot(A0)
    E_ST = S0 * np.exp(r * T)
    ST = data + E_ST - K
    zero_column = np.zeros((N_SIM, 1))
    ST = np.hstack((ST, zero_column))
    max_values = np.amax(ST, axis=1)
    max_values = max_values.reshape(N_SIM, 1)
    avg_val = np.mean(max_values)
    sim_avg = np.append(sim_avg, avg_val)

final_mean = np.mean(sim_avg)
final_std = np.std(sim_avg)
print((round(final_mean - 1.96 * final_std, 4), round(final_mean + 1.96 * final_std, 4)))



