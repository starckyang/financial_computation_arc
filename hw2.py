import numpy as np
import copy
import time
from prettytable import PrettyTable

# inputs
St = 50
K = 50
r = 0.1
q = 0.05
sigma = 0.8
t = 0.25
T = 0.5
M = 100
n = 100
Savg = 50
N_SIM = 1000
B = 10

# simple calculated results
delta_t = T-t
u_ = np.exp(sigma * np.sqrt(delta_t/n))
d_ = np.exp(-sigma * np.sqrt(delta_t/n))
p_ = (np.exp((r-q) * delta_t / n) - d_) / (u_ - d_)
q_ = 1 - p_
previous_portion = int(n * (t / (T-t))) + 1

# constructing a tree with maximum value storage
rows = 2 * n + 1
columns = n + 1
record_tree = np.zeros((rows, columns))
record_tree[n, 0] = St
for i in range(n):
    record_tree[n+1+i, i+1] = record_tree[n+i, i] * d_
    record_tree[:2*n, i+1] += record_tree[1:2*n+1, i] * u_

avg_rec_quick = [[[] for _ in range(2 * n + 1)] for _ in range(n+1)]
avg_rec_quick[0][n] = [Savg]
for i in range(1, n+1): # 0 to i is the possible top lane, and count means the number of possibilities
    count = 0
    for l in range(2*n+1):
        if (record_tree[l, i] != 0) & (i != 0):
            ups = i - count
            downs = count
            inter_sum = 0
            for k in range(1, i+1):
                if k <= ups:
                    inter_sum += u_ ** k
                else:
                    inter_sum += (u_ ** ups) * (d_ ** (k-ups))
            Amax = ((St * inter_sum) + Savg * previous_portion) / (previous_portion + i)
            inter_sum = 0
            for k in range(1, i+1):
                if k <= downs:
                    inter_sum += d_ ** k
                else:
                    inter_sum += (d_ ** downs) * (u_ ** (k-downs))
            Amin = ((St * inter_sum) + Savg * previous_portion) / (previous_portion +i)
            avg_rec_quick[i][l] = [(Amax * (M - k) / M) + (Amin * k / M) for k in range(M+1)]
            count += 1

# print(avg_rec_quick)
# print(record_tree)


def rounder(number):
    if (number % 1) >= 0.5:
        return int(np.ceil(number))
    return int(np.floor(number))


def locator(target_list, value, method):
    if value <= target_list[-1]:
        return [-1, -1]
    if value >= target_list[0]:
        return [0, 0]
    if method == "sequential":
        for i in range(len(target_list)-1):
            if (value <= target_list[i]) & (value >= target_list[i+1]):
                return [i, i+1]
    if method == "binary":
        bot = 0
        top = len(target_list)-1
        for i in range(len(target_list) - 1):
            mid = rounder((bot + top) / 2)
            if (target_list[mid] <= value) & (target_list[mid-1] >= value):
                return [mid-1, mid]
            elif target_list[mid] > value:
                bot = mid
            else:
                top = mid
    if method == "linear":
        cur_max = target_list[0]
        cur_min = target_list[-1]
        if cur_max == cur_min:
            return [0, 0]
        if int(np.floor((cur_max - value) / (cur_max - cur_min) * M)) < 0:
            return [0, int(np.ceil((cur_max - value) / (cur_max - cur_min) * M))]
        elif int(np.ceil((cur_max - value) / (cur_max - cur_min) * M)) > len(target_list) - 1:
            return [int(np.floor((cur_max - value) / (cur_max - cur_min) * M)), -1]
        else:
            return [int(np.floor((cur_max - value) / (cur_max - cur_min) * M)),
                      int(np.ceil((cur_max - value) / (cur_max - cur_min) * M))]


def eu_call(method):
    reward_tree = [[[] for _ in range(2 * n + 1)] for _ in range(n+1)]
    for i in range(n+1):
        if i == 0:
            for k in range(2*n+1):
                reward_tree[-i-1][k] = [max(max_list-K, 0)
                                        for max_list in copy.deepcopy(avg_rec_quick[-i-1][k])]
        else:
            for k in range(2*n+1):
                my_max = copy.deepcopy(avg_rec_quick[-i - 1][k])
                if (my_max!=[]) & (k != 0) & (k!=2*n):
                    upper_comparative_max = copy.deepcopy(avg_rec_quick[-i][k-1])
                    lower_comparative_max = copy.deepcopy(avg_rec_quick[-i][k+1])
                    for c_max in my_max:
                        cur_column = n-i
                        Au = (c_max * (cur_column+previous_portion) + record_tree[k-1, -i]) \
                             / (cur_column+previous_portion+1)
                        Ad = (c_max * (cur_column+previous_portion) + record_tree[k+1, -i]) \
                             / (cur_column+previous_portion+1)
                        UP_pos = locator(upper_comparative_max, Au, method)
                        DOWN_pos = locator(lower_comparative_max, Ad, method)
                        if upper_comparative_max[UP_pos[0]] == upper_comparative_max[UP_pos[1]]:
                            UP_wu = 1
                        else:
                            UP_wu = (upper_comparative_max[UP_pos[0]] - Au) / \
                                    (upper_comparative_max[UP_pos[0]] - upper_comparative_max[UP_pos[1]])
                        UP_val = reward_tree[-i][k-1][UP_pos[0]] * (1-UP_wu) + reward_tree[-i][k-1][UP_pos[1]] * UP_wu
                        if lower_comparative_max[DOWN_pos[0]] == lower_comparative_max[DOWN_pos[1]]:
                            DOWN_wu = 1
                        else:
                            DOWN_wu = (lower_comparative_max[DOWN_pos[0]] - Ad) / \
                                      (lower_comparative_max[DOWN_pos[0]] - lower_comparative_max[DOWN_pos[1]])
                        DOWN_val = reward_tree[-i][k+1][DOWN_pos[0]] * (1-DOWN_wu) \
                                   + reward_tree[-i][k+1][DOWN_pos[1]] * DOWN_wu
                        reward_tree[-i-1][k].append((p_*UP_val+q_*DOWN_val)*np.exp(-r*delta_t/n))
    return reward_tree[0][n][0]


# american
def am_call():
    reward_tree = [[[] for _ in range(2 * n + 1)] for _ in range(n+1)]
    for i in range(n+1):
        if i == 0:
            for k in range(2*n+1):
                reward_tree[-i-1][k] = [max(max_list-K, 0)
                                        for max_list in copy.deepcopy(avg_rec_quick[-i-1][k])]
        else:
            for k in range(2*n+1):
                my_max = copy.deepcopy(avg_rec_quick[-i - 1][k])
                if (my_max!=[]) & (k != 0) & (k!=2*n):
                    upper_comparative_max = copy.deepcopy(avg_rec_quick[-i][k-1])
                    lower_comparative_max = copy.deepcopy(avg_rec_quick[-i][k+1])
                    for c_max in my_max:
                        UP_pos = []
                        DOWN_pos = []
                        a1 = record_tree[k-1, -i]
                        a2 = record_tree[k + 1, -i]
                        cur_column = n-i
                        Au = (c_max * (cur_column+previous_portion) + record_tree[k-1, -i]) / (cur_column+previous_portion+1)
                        Ad = (c_max * (cur_column+previous_portion) + record_tree[k+1, -i]) / (cur_column+previous_portion+1)
                        UP_max = upper_comparative_max[0]
                        UP_min = upper_comparative_max[-1]
                        DOWN_max = lower_comparative_max[0]
                        DOWN_min = lower_comparative_max[-1]
                        if UP_max == UP_min:
                            UP_pos = [0, 0]
                        else:
                            if int(np.floor((UP_max - Au) / (UP_max - UP_min) * M)) < 0:
                                UP_pos = [0, int(np.ceil((UP_max - Au) / (UP_max - UP_min) * M))]
                            elif int(np.ceil((UP_max - Au) / (UP_max - UP_min) * M)) > len(upper_comparative_max)-1:
                                UP_pos = [int(np.floor((UP_max - Au) / (UP_max - UP_min) * M)), -1]
                            else:
                                UP_pos = [int(np.floor((UP_max - Au) / (UP_max - UP_min) * M)),
                                          int(np.ceil((UP_max - Au) / (UP_max - UP_min) * M))]
                        if DOWN_max == DOWN_min:
                            DOWN_pos = [0, 0]
                        else:
                            if int(np.floor((DOWN_max - Ad) / (DOWN_max - DOWN_min) * M)) < 0:
                                DOWN_pos = [0, int(np.ceil((DOWN_max - Ad) / (DOWN_max - DOWN_min) * M))]
                            elif int(np.ceil((DOWN_max - Ad) / (DOWN_max - DOWN_min) * M)) > len(upper_comparative_max)-1:
                                DOWN_pos = [int(np.floor((DOWN_max - Ad) / (DOWN_max - DOWN_min) * M)), -1]
                            else:
                                DOWN_pos = [int(np.floor((DOWN_max - Ad) / (DOWN_max - DOWN_min) * M)),
                                          int(np.ceil((DOWN_max - Ad) / (DOWN_max - DOWN_min) * M))]
                        if upper_comparative_max[UP_pos[0]] == upper_comparative_max[UP_pos[1]]:
                            UP_wu = 1
                        else:
                            UP_wu = (upper_comparative_max[UP_pos[0]] - Au) / \
                                    (upper_comparative_max[UP_pos[0]] - upper_comparative_max[UP_pos[1]])
                        UP_val = reward_tree[-i][k-1][UP_pos[0]] * (1-UP_wu) + reward_tree[-i][k-1][UP_pos[1]] * UP_wu
                        if lower_comparative_max[DOWN_pos[0]] == lower_comparative_max[DOWN_pos[1]]:
                            DOWN_wu = 1
                        else:
                            DOWN_wu = (lower_comparative_max[DOWN_pos[0]] - Ad) / \
                                      (lower_comparative_max[DOWN_pos[0]] - lower_comparative_max[DOWN_pos[1]])
                        DOWN_val = reward_tree[-i][k+1][DOWN_pos[0]] * (1-DOWN_wu) \
                                   + reward_tree[-i][k+1][DOWN_pos[1]] * DOWN_wu
                        total_val = (p_*UP_val+q_*DOWN_val)*np.exp(-r*delta_t/n)
                        exercise_val = c_max - K
                        if exercise_val > total_val:
                            reward_tree[-i - 1][k].append(exercise_val)
                        else:
                            reward_tree[-i - 1][k].append((p_ * UP_val + q_ * DOWN_val) * np.exp(-r * delta_t / n))
    return reward_tree[0][n][0]


def monte_carlo():
    delta_t = (T-t)
    value = np.array([])
    for i in range(B):
        Stt = copy.deepcopy(St)
        avg_record = np.full((N_SIM, 1), Savg)
        for j in range(n):
            raw_sim = np.random.normal(0, 1, (N_SIM, 1))
            Stt = Stt * np.exp((r - q - 0.5*(sigma**2))*(delta_t/n) + raw_sim*sigma*np.sqrt(delta_t/n))
            avg_record = (Stt + avg_record * (previous_portion + j)) / (previous_portion + j + 1)
        zero_column = np.zeros((N_SIM, 1))
        avg_record = np.hstack((avg_record-K, zero_column))
        avg_record = np.amax(avg_record, axis=1)
        value = np.append(value, np.mean(avg_record)*np.exp(-r*delta_t))
    simed_mean = np.mean(value)
    simed_std = np.std(value)
    return (round(simed_mean-1.96*simed_std,4), round(simed_mean+1.96*simed_std, 4))


# eu_call_price = eu_call("linear")
# am_call_price = am_call()
# monte = monte_carlo()
#
# output_table = PrettyTable()
# output_table.field_names = ["Eu", "Am", "Monte"]
# output_table.add_row(
#     [round(eu_call_price,4), round(am_call_price, 4), monte]
# )
# print("Your Outputs are:\n")
# print(output_table)


# bonus 2

def timer(method):
    start = time.time()
    price = eu_call(method)
    end = time.time()
    return end-start, price


time_output_table = PrettyTable()
time_output_table.field_names = ["binary", "linear", "sequential"]
bin_price, bin_time = timer("binary")
lin_price, lin_time = timer("linear")
seq_price, seq_time = timer("sequential")
time_output_table.add_rows(
    [[bin_price, lin_price, seq_price],
    [bin_time, lin_time, seq_time]]
)
print("Your Runtimes are:\n")
print(time_output_table)