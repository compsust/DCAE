"""
    Power Assignation Metric
    Alejandro Rodriguez Silva
"""
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, f1_score
from power_assignation_metric import *
from change_point_filt import *


def test_case(y, y_hat, lmbda=1):
    g_input = y
    f_input = y_hat
    g = []
    f = []

    for j in range(len(g_input)):
        g.append(g_input[j][:])

    for i in range(len(f_input)):
        f.append(f_input[i][:])

    e, alpha, beta, t_time = metric_nilm_sparsity_v4(g, f, lmbda)

    print('\n\n\tThe assignations are:')
    print('Alpha:\n', alpha)

    print('\n\n\tThe assignations are:')
    print('Beta:\n', beta)

    print('\nThe error = ', e)
    print('\nTotal elapsed time = ', t_time)
"""
    Metric parameters
"""

w_size = 1800  # window size for calculating the error metric
ds_factor = 2  # downsampling factor
case = 4

"""
    Load data and hand pick the signals
    that we will use for our test cases
"""
noise_thres = 50
df = pd.read_csv('/Users/alejandro/Documents/Code/UNILM/house1_power_blk1.csv')

mains = df['mains'].values
fridge = df['sub8'].values
cde = df['sub5'].values + df['sub6'].values
hp = df['sub13'].values + df['sub14'].values
furnace = df['sub11'].values
dw = df['sub10'].values

"""
    Introduce the signals
"""
# 2300 samples
t_plot_fridge = np.arange(59400, 61700)
t_plot_cde = np.arange(50000, 54000)
t_plot_dw = np.arange(502000, 504300)

# # 1800 samples
# t_plot_fridge = np.arange(59400, 61500)
# t_plot_dw = np.arange(502000, 504100)

# # 3600 samples
# t_plot_fridge = np.arange(59400, 62700)
# t_plot_dw = np.arange(502000, 505300)

#######
t_state_dw = 1400 # time where the second state starts
dw_state_1 = np.zeros(len(t_plot_dw))
dw_state_1[0:t_state_dw] = dw[t_plot_dw[0:t_state_dw]]
# zero out power less than noise_thres W (for simplicity)
dw_state_1[dw_state_1 < noise_thres] = 0

dw_state_2 = np.zeros(len(t_plot_dw))
dw_state_2[t_state_dw:] = dw[t_plot_dw[t_state_dw:]]
dw_state_2[dw_state_2 < noise_thres] = 0

####### signals that will be used for all test cases
y1 = fridge[t_plot_fridge]
dw = dw[t_plot_dw]
y2 = dw
y2[dw < noise_thres]=0

y2_s1 = dw_state_1
y2_s2 = dw_state_2

##### aggregate
y = y1 + y2 # obtain the aggregate

##### filtered versions of the signals
y_hat = cp_filter(y, 30, 5) # obtain the filtered signal
y1_hat = cp_filter(y1, 30, 5) # obtain the filtered signal
y2_hat = cp_filter(y2, 30, 5)

"""
    Test case 1. Disaggregator is the sum of the inferred signals

"""

# create lists for passing it to the metric

if case == 1:
    # bad disaggregator
    # y_in = list( np.stack((y1,y2)) )
    # y_hat_in = list([y])
    # test_case(y_in, y_hat_in, 1)
    # print('The RMSE = ', mean_squared_error(y_in, y_in, squared=False))

    # good disaggregator
    y_in = list( np.stack((y1,y2)) )
    y_hat_in = list( np.stack((y1,y2)) )
    test_case(y_in, y_hat_in, 1)

"""
    Test Case 2. Symmetry
"""
if case == 2:
    y_in = list( np.stack((y1,y2)) )
    y_hat_in = list( np.stack((y2,y1)) )
    g_input = y_in
    f_input = y_hat_in
    test_case(y_in, y_hat_in)
"""
    Test Case 3. Penalizing splitting into different operational modes
"""
if case == 3:
    lmbda = 1e6
    # perfect disaggregator
    y_in = list( np.stack((y1,y2)) )
    y_hat_in = list( np.stack((y1,y2)) )
    test_case(y_in, y_hat_in, lmbda)

    # split things
    y_in = list( np.stack((y1,y2)) )
    y_hat_in = list( np.stack((y1, y2_s1, y2_s2)) )
    g_input = y_in
    f_input = y_hat_in
    test_case(y_in, y_hat_in, lmbda)
"""
    Test Case 4. Power - scaling the outputs
"""
y_scaled = y*0.5
y1_scaled = y1*0.8
y2_scaled = y2*0.8

if case == 4:
    lmbda = 1e4
    # perfect disaggregator
    y_in = list( np.stack((y1,y2)) )
    y_hat_in = list( np.stack((y1,y2)) )
    test_case(y_in, y_hat_in, lmbda)

    # scaled disaggregator
    y_in = list( np.stack((y1,y2)) )
    y_hat_in = list( np.stack((y1_scaled, y2_scaled)) )
    test_case(y_in, y_hat_in, lmbda)
    # g_input = y_in
    # f_input = y_hat_in

    # for calculating the f1-score - fridge

    on_thr = 20 # above 20 watts we consider the appliance to be ON
    y_true_fridge = [i > on_thr for i in y1]
    y_pred_fridge = [i > on_thr for i in y1_scaled]
    f1_fridge = f1_score(y_true_fridge, y_pred_fridge)
    print('The F1-score for the scaled inferred signal of the fridge = ', f1_fridge)
    y_true_dw = [i > on_thr for i in y2]
    y_pred_dw = [i > on_thr for i in y2_scaled]
    f1_dw = f1_score(y_true_dw, y_pred_dw)
    print('The F1-score for the scaled inferred signal of the furnace = ', f1_dw)


"""
    Error metric calculation
"""
# if case != 0:
#     g = []
#     f = []
#
#     for j in range(len(g_input)):
#         g.append(g_input[j][:])
#
#     for i in range(len(f_input)):
#         f.append(f_input[i][:])
#
#     e, alpha, beta, t_time = metric_nilm_sparsity_v3(g, f)
#
#     print('\n\n\tThe assignations are:')
#     print('Beta:\n', beta)
#
#     print('\nThe error = ', e)
#     print('\nTotal elapsed time = ', t_time)


# if case != 0:
#     E = []
#     ALPHAS = []
#     BETAS = []
#     t_time = 0
#     ener_g_total_window = []
#     ener_f_total_window = []
#     eneg_g_indiv_window = []
#     eneg_f_indiv_window = []
#
#     n_samples = len(y_in[0]) # total number of samples in the signals
#     t_windows = int(n_samples/w_size)
#
#     for w in range(0,t_windows):
#         print('Calculating window number:', w)
#         idx_left = w * w_size
#         idx_right = (w+1) * w_size
#         g = []
#         f = []
#         for j in range(len(g_input)):
#             g.append(g_input[j][idx_left:idx_right])
#         eneg_g_indiv_window.append(np.sum(y_in, axis=1))
#         for i in range(len(f_input)):
#             f.append(f_input[i][idx_left:idx_right])
#         eneg_f_indiv_window.append(np.sum(y_hat_in, axis=1))
#
#         e, alpha, beta, t_time_w = metric_nilm_sparsity_v3(g, f)
#         E.append(e) # append the error at each window
#         ALPHAS.append(alpha) # append alphas
#         BETAS.append(beta)
#         t_time+=t_time_w
#         print('------')
#
#     print('Energy ground truths:', eneg_g_indiv_window)
#     print('Energy output signals:', eneg_f_indiv_window)
#
#     print('Total energy ground truths per window:', np.sum(eneg_g_indiv_window,axis=1))
#     print('Total energy output signals per window:', np.sum(eneg_f_indiv_window,axis=1))
#
#     print('\n\n\tThe assignations are:')
#
#     for i in range(len(BETAS)):
#         print(i)
#         print(BETAS[i])
#         print('*********')
#
#     print('\nThe error = ', np.sum(E)/t_windows)
#
#     print('\nTotal elapsed time = ', t_time)

"""
    Graphs
"""
if case == 0:
    plt.figure()
    plt.plot(y1, label='Fridge')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()
    #############################
    plt.figure()
    plt.plot(y2, label='Dishwasher')
    plt.xlabel('Time (sec)', fontsize=11)
    plt.ylabel('Power (Watts)', fontsize=11)
    plt.legend()

    plt.figure()
    plt.plot(y2_s1, label='Dishwasher - state 1')
    plt.xlabel('Time (sec)', fontsize=11)
    plt.ylabel('Power (Watts)', fontsize=11)
    plt.legend()

    plt.figure()
    plt.plot(y2_s2, label='Dishwasher - state 2')
    plt.xlabel('Time (sec)', fontsize=11)
    plt.ylabel('Power (Watts)', fontsize=11)
    plt.legend()
    plt.show()
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(y2, label='$y_{t}^{(2)}$ - Dishwasher')
    # plt.xlabel('Time (sec)', fontsize=11)
    # plt.ylabel('Power (Watts)', fontsize=11)
    # plt.legend()
    #
    # plt.subplot(312)
    # plt.plot(y2_s1, label='Dishwasher - state 1')
    # plt.xlabel('Time (sec)', fontsize=11)
    # plt.ylabel('Power (Watts)', fontsize=11)
    # plt.legend()
    #
    # plt.subplot(313)
    # plt.plot(y2_s2, label='Dishwasher - state 2')
    # plt.ylabel('Power (Watts)', fontsize=11)
    # plt.xlabel('Time (sec)', fontsize=11)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
#####################################################
if case == 1:
    # aggregate
    plt.figure()
    plt.plot(y, label='Aggregate')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()

    # poor disaggregator
    plt.figure()
    plt.plot(y, label='Disaggregated output')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()

    # good disaggregator
    plt.figure()
    plt.plot(y1, label='Disaggregated output 1')
    plt.plot(y2, label='Disaggregated output 2')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()
    plt.show()

####################################################
if case == 2:
    aggregate
    plt.figure()
    plt.plot(y, label='Aggregate')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()

    # disaggregator 1
    plt.figure()
    plt.plot(y1, label='Disaggregated output 1')
    plt.plot(y2, label='Disaggregated output 2')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()

    # disaggregator 2
    plt.figure()
    plt.plot(y2, label='Disaggregated output 1')
    plt.plot(y1, label='Disaggregated output 2')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()
    plt.show()

####################################################
# if case == 3:
    # aggregate
    plt.figure()
    plt.plot(y, label='Aggregate')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()

    # disaggregator 1
    plt.figure()
    plt.plot(y1, label='Disaggregated output 1')
    plt.plot(y2, label='Disaggregated output 2')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()

    # disaggregator 2
    plt.figure()
    plt.plot(y1, label='Disaggregated output 1')
    plt.plot(y2_s1, label='Disaggregated output 2')
    plt.plot(y2_s2, label='Disaggregated output 3')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()
    plt.show()

if case == 4:
    # aggregate
    plt.figure()
    plt.plot(y, label='Aggregate')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()

    # disaggregator 1
    plt.figure()
    plt.plot(y1, label='Disaggregated output 1')
    plt.plot(y2, label='Disaggregated output 2')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()

    # disaggregator 2
    plt.figure()
    plt.plot(y1_scaled, label='Disaggregated output 1')
    plt.plot(y2_scaled, label='Disaggregated output 2')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Power (Watts)',fontsize=11)
    plt.legend()
    plt.show()
