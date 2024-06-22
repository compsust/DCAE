import cvxpy as cp
import numpy as np
import time

def metric_nilm_sparsity_v4(g, f, lambda_= 1):
    I = len(f) # number of individual appliances
    J = len(g) # number of disaggregated appliances
    K = len(f[0]) # time instant

    # print(I)
    # print(J)
    # print(K)
    #
    # print(g[J-1][:])
    # print(f[I-1][:])
    # normalization term for now

    ener_g = []
    for j in range(J):
        ener_g.append(np.sum([k**2 for k in g[j]]))

    ener_g = np.sum(ener_g)
    norm_term = ener_g

    if norm_term < 1e-3:
        norm_term = 1e-3

    # 1. variables Creation
    var_1 = cp.Variable(I*J)
    ALPHA = np.array(list(iter(var_1))).reshape((I,J))

    var_2 = cp.Variable(I*J)

    BETA = np.array(list(iter(var_2))).reshape((I,J))

    # 2. Objective Formulation

    # Part 1 of the optimization
    # obj_1 = cp.sum([cp.square(g[j][k] - cp.sum([ALPHA[i][j]*f[i][k] for i in range(I)])) for k in range(K) for j in range(J)])/norm_term
    # obj_1 += (lambda_/norm_term)*cp.sum([ALPHA[i][j] for i in range(I) for j in range(J)])
    # objective_1 = cp.Minimize(obj_1)

    obj_1 = cp.sum([cp.square(g[j][k] - cp.sum([ALPHA[i][j]*f[i][k] for i in range(I)])) for k in range(K) for j in range(J)])
    obj_1 += lambda_*cp.sum([ALPHA[i][j] for i in range(I) for j in range(J)])
    objective_1 = cp.Minimize(obj_1/norm_term)

    # obj_1 = cp.sum([cp.square(g[j][k] - cp.sum([ALPHA[i][j]*f[i][k] for i in range(I)])) for k in range(K) for j in range(J)])
    # obj_1 =  (obj_1/norm_term)
    # obj_1 += lambda_*cp.sum([ALPHA[i][j] for i in range(I) for j in range(J)])
    # objective_1 = cp.Minimize(obj_1)


    # Part 2 of the optimization
    # obj_2 = cp.sum([cp.square(g[j][k] - cp.sum([BETA[i][j]*f[i][k] for i in range(I)])) for k in range(K) for j in range(J)])/norm_term
    # objective_2 = cp.Minimize(obj_2)

    obj_2 = cp.sum([cp.square(g[j][k] - cp.sum([BETA[i][j]*f[i][k] for i in range(I)])) for k in range(K) for j in range(J)])
    objective_2 = cp.Minimize(obj_2/norm_term)

    # 3. Constraints Creation

    constraints_1 = [
        var_1 >= 0.,
        var_1 <= 1.,
#         *[cp.sum([ALPHA[i][j] for j in range(J)]) == 1 for i in range(I)] # unpack the list with *
    ]

    constraints_2 = [
        var_2 >= 0.,
        var_2 <= 1.,
        *[cp.sum([BETA[i][j] for j in range(J)]) == 1 for i in range(I)] # unpack the list with *
    ]

    # 4. Problem solution
    prob_1 = cp.Problem(objective_1, constraints_1)
    prob_2 = cp.Problem(objective_2, constraints_2)

    # solve part 1 of the minimization problem

    tstart_obj1 = time.time() # start the clock
    print('Solving objective function 1...')
    error_1 = prob_1.solve(solver=cp.OSQP)
    tend_obj1 = time.time() # stop the clock
    print('Elapsed time:', tend_obj1 - tstart_obj1)

    # solve part 2 of the minimization problem

    tstart_obj2 = time.time() # start the clock
    print('Solving objective function 2...')
    error_2 = prob_2.solve(warm_start=True)
    tend_obj2 = time.time() # stop the clock
    print('Elapsed time: ', tend_obj2 - tstart_obj2)

    # calculate the total error given by the metric
    total_error = error_1 + error_2
    print(f"Metric Value: {total_error}")
    x_sol_1 = np.round(var_1.value.reshape((I,J)), 3)
    x_sol_2 = np.round(var_2.value.reshape((I,J)), 3)

    # calculate the elapsed time for minimizing LHS and RHS of the metric
    t_time = (tend_obj1 - tstart_obj1) + (tend_obj2 - tstart_obj2)
#     print(f"Function assignation: \n{np.round(x_sol,3)}")
    return total_error, x_sol_1, x_sol_2, t_time
