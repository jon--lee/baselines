import tensorflow as tf
import numpy as np
import IPython
from baselines.deepqmax4.simple import featurize, control_params
from scipy.optimize import minimize
import cma
from DIRECT import solve
np.random.seed(1)
tf.set_random_seed(1)



# def original(env, graph_args, xs, iterations):
#     q_values = graph_args['q_values']
#     input_var = graph_args['input_var']
#     train_inputs = graph_args['train_inputs']
#     clipper = graph_args['clipper']
#     eval_inputs = graph_args['eval_inputs']

#     # iterations = 600

#     debug = {'paths': []}


#     try:    
#         d = env.env.num_params
#         bounds = env.env.bounds
#         ac_low, ac_high = env.env.ac_low, env.env.ac_high
#     except: 
#         d = env.num_params
#         bounds = env.bounds
#         ac_low, ac_high = env.ac_low, env.ac_high
#     controls = np.zeros((xs.shape[0], d))

#     for j, x in enumerate(xs):
#         best_u = None
#         best_value = -float("inf")
#         for i in range(5):
#             path = []
#             res = tf.variables_initializer([input_var]).run()
#             # input_var.assign([[.02, .02]]).eval()
#             for _ in range(iterations):
#                 results = train_inputs([x])
#                 clipper.eval()
#                 new_value = results[2]
#                 path.append(new_value[0])
#             fn_value = results[1]
#             if fn_value > best_value:
#                 best_u = new_value
#                 best_value = fn_value
#             debug['paths'].append(np.array(path))

#         # print("Final value: " + str(results[1]))
#         # print("Final params: " + str(results[0]))
#         u = np.clip(best_u, ac_low, ac_high)

#         controls[j, :] = u

#     return controls, debug


"""
Gradient descent controller
"""
def controller1(env, graph_args, xs, iterations):
    q_values = graph_args['q_values']
    input_var = graph_args['input_var']
    train_inputs = graph_args['train_inputs']
    clipper = graph_args['clipper']
    eval_inputs = graph_args['eval_inputs']
    rounds = graph_args['rounds']

    debug = {'paths': [], 'us': []}

    d, bounds, ac_low, ac_high = control_params(env)
    controls = np.zeros((xs.shape[0], d))

    for j, x in enumerate(xs):
        best_u = None
        best_value = -float("inf")

        for i in range(rounds):
            path = []
    
            res = tf.variables_initializer([input_var]).run()

            for _ in range(iterations):
                results = train_inputs([x])
                clipper.eval()
                new_value = results[2]
                path.append(new_value[0])
            fn_value = results[1]
            debug['us'].append(new_value[0])
            debug['paths'].append(np.array(path))

            if fn_value > best_value:
                best_u = new_value
                best_value = fn_value

        controls[j, :] = best_u

    return controls, debug


"""
    Random sample controller
"""
def controller2(env, graph_args, xs, iterations):
    d, bounds, ac_low, ac_high = control_params(env)
    
    q_values = graph_args['q_values']
    rounds = graph_args['rounds']


    controls = np.zeros((xs.shape[0], d))
    low, high = np.tile(ac_low, (rounds, 1)), np.tile(ac_high, (rounds, 1))
    
    for j, x in enumerate(xs):
        rand = np.random.uniform(low, high)
        feat = featurize(env, x * np.ones((rounds, x.shape[0])), rand)


        values = -q_values(feat)
        index = np.argmin(values)

        best_u = rand[index]
        best_value = values[index]

        controls[j, :] = best_u

    debug = { 'us': rand }

    return controls, debug


"""
Scipy optimizer controller
"""
def controller3(env, graph_args, xs, iterations):
    d, bounds, ac_low, ac_high = control_params(env)
    
    q_values = graph_args['q_values']
    rounds = graph_args['rounds']


    controls = np.zeros((xs.shape[0], d))
    low, high = np.tile(ac_low, (rounds, 1)), np.tile(ac_high, (rounds, 1))
    
    for j, x in enumerate(xs):
        def f(u, user_data):
            feat = featurize(env, [x], [u])
            values = -q_values(feat)

            return values.item()

        for i in range(1):
            x0 = np.random.uniform(ac_low, ac_high)
            bounds = list(zip(ac_low, ac_high))
            IPython.embed()

        controls[j, :] = best_u

    debug = {'us': [best_u]}
    return controls, debug


"""
cma-es controller
"""

def controller4(env, graph_args, xs, iterations):
    d, bounds, ac_low, ac_high = control_params(env)
    
    q_values = graph_args['q_values']
    rounds = graph_args['rounds']


    controls = np.zeros((xs.shape[0], d))
    # low, high = np.tile(ac_low, (rounds, 1)), np.tile(ac_high, (rounds, 1))
    
    for j, x in enumerate(xs):
        def f(u):
            feat = featurize(env, [x], [u])
            values = -q_values(feat)

            return values.item()

        us = []
        best_u = None
        best_value = float("inf")
        for i in range(rounds):
            x0 = np.random.uniform(ac_low, ac_high)
            bounds = list(zip(ac_low, ac_high))
            es = cma.CMAEvolutionStrategy(x0, 1.0, {'bounds':[-2.0, 2.0], 'verb_disp':0, 'verbose':-1, 'maxiter':200})
            es.optimize(f)
            u = es.result.xbest
            value = es.result.fbest


            us.append(u)
            if value < best_value:
                best_value = value
                best_u = u

        controls[j, :] = best_u

    debug = {'us': us}
    return controls, debug


controller_map = {'contr1': controller1,
                     'contr2': controller2,
                      'contr3': controller3,
                       'contr4': controller4}

