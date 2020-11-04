import os
import plotly
from plotly.graph_objs import Scatter, Figure
from plotly.graph_objs.scatter import Line
import numpy as np
import tensorflow as tf

def get_eval_results(file, field):
    eval_returns = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == field:
                eval_returns.append(v.simple_value)
    return np.array(eval_returns, dtype=np.float32)

def plot_lines(xs, y_lines, line_names, title, xaxis='Iteration', yaxis='Reward', yrange=None):
    line_colors = ['rgb(60, 180, 75)', 'rgb(145, 30, 180)', 'rgb(230, 25, 75)', 'rgb(0, 172, 237)', 'rgb(170, 110, 40)', 'rgb(240, 50, 230)', 'rgb(255, 255, 25)', 'rgb(128, 0, 0)', 'rgb(128, 128, 128)']
    fill_colors = ['rgba(60, 180, 75, 0.2)', 'rgba(145, 30, 180, 0.2)', 'rgba(230, 25, 75, 0.2)', 'rgba(0, 172, 237, 0.2)', 'rgba(170, 110, 40, 0.2)', 'rgba(240, 50, 230, 0.2)', 'rgba(255, 255, 25, 0.2)', 'rgba(128, 0, 0, 0.2)', 'rgba(128, 128, 128, 0.2)']

    blank = 'rgba(0, 0, 0, 0)'

    # y_lines = np.array(y_lines)#, dtype=np.float32)
    data = []

    for i in range(len(line_names)):
        y = y_lines[i]
        y_mean, y_std = y[:2]

        line_color = line_colors[i]
        fill_color = fill_colors[i]

        # __import__('ipdb').set_trace()
        mean = Scatter(x=xs, y=y_mean, line=Line(color=line_color), name=line_names[i])
        data += [mean]

        if (y_std.shape[0] > 0):
            y_upper, y_lower = y_mean + y_std, y_mean - y_std
            upper_std = Scatter(x=xs, y=y_upper, line=Line(color=blank), showlegend=False)
            lower_std = Scatter(x=xs, y=y_lower, fill='tonexty', fillcolor=fill_color, line=Line(color=blank), showlegend=False)
            data += [upper_std, lower_std]

        if len(y) > 2:
            y_max = y[2]
            max_val = Scatter(x=xs, y=y_max, line=Line(color=line_colors[-(i+1)]), name="max_" + line_names[i])
            data += [max_val]

    fig = Figure()
    for trace in data:
        fig.add_trace(trace)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=1.23,
        xanchor="right",
        # x=0.01
    ))
    fig.update_layout(
        title=title,
        xaxis={'title': xaxis},
        yaxis={'title': yaxis, 'range': yrange},
        autosize=False,
        width=1000,
        height=600
    )
    fig.write_image(f'{title}_plot.png')
    return fig



pwd = "/Users/henktillman/Dropbox/Berkeley/8/cs285/hw/hw3/final_data/"

# Experiment 1

fields = ['Train_AverageReturn', 'Train_StdReturn', 'Train_BestReturn']
folder = "/"

header1 = "hw3_q1_"

# contents = os.listdir(pwd + folder)


# graph1_logs = []
# for item in contents:
#     if header1 in item:
#         logname = pwd+folder+"/"+item+"/"+os.listdir(pwd+folder+item)[0]
#     if header1 in item:
#         graph1_logs.append((item, logname))

# graph1_logs.sort(key=lambda x: x[0])

# graph1_data = [
#     [
#         get_eval_results(log[1], field) for field in fields
#     ] for log in graph1_logs
# ]
# plot_lines(np.arange(len(graph1_data[0][0])), graph1_data, [log[0] for log in graph1_logs], title="Experiment 1: Basic Q-learning Performance")


################################################################################

# Experiment 2


header1 = "hw3_q2_dqn"
header2 = "hw3_q2_doubledqn"

# contents = os.listdir(pwd)
# fields = ['Train_AverageReturn', 'Train_StdReturn']

# graph1_logs = []
# graph2_logs = []
# for item in contents:
#     if header1 in item or header2 in item:
#         if 'gym' in os.listdir(pwd+folder+item)[0]:
#             logname = pwd+folder+"/"+item+"/"+os.listdir(pwd+folder+item)[1]
#         else:
#             logname = pwd+folder+"/"+item+"/"+os.listdir(pwd+folder+item)[0]
#     if header1 in item:
#         graph1_logs.append((item, logname))
#     elif header2 in item:
#         graph2_logs.append((item, logname))

# graph1_logs.sort(key=lambda x: x[0])
# graph2_logs.sort(key=lambda x: x[0])

# graph1_data = [
#     [
#         get_eval_results(log[1], field) for field in fields
#     ] for log in graph1_logs
# ]

# graph2_data = [
#     [
#         get_eval_results(log[1], field) for field in fields
#     ] for log in graph2_logs
# ]

# data = []

# avg = (graph1_data[0][0] + graph1_data[1][0] + graph1_data[2][0]) / 3
# data += [[avg, np.array([])]]
# avg = (graph2_data[0][0] + graph2_data[1][0]) / 2
# data += [[avg, np.array([])]]
# # __import__('ipdb').set_trace()

# plot_lines(np.arange(len(graph1_data[0][0])), data, ['DQN', 'DoubleDQN'], title="Experiment 2 DQN vs Double DQN")


################################################################################

# Experiment 3


# folder = "/exp4a_data_final/"

header1 = "hw3_q3"
# contents = os.listdir(pwd)

# fields = ['Train_AverageReturn', 'Train_StdReturn']
# graph1_logs = []
# for item in contents:
#     if header1 in item:
#         if 'gym' in os.listdir(pwd+folder+item)[0]:
#             logname = pwd+folder+"/"+item+"/"+os.listdir(pwd+folder+item)[1]
#         else:
#             logname = pwd+folder+"/"+item+"/"+os.listdir(pwd+folder+item)[0]
#     if header1 in item:
#         graph1_logs.append((item, logname))

# graph1_logs.sort(key=lambda x: x[0])

# graph1_data = [
#     [
#         get_eval_results(log[1], field) for field in fields
#     ] for log in graph1_logs
# ]
# plot_lines(np.arange(len(graph1_data[0][0])), graph1_data, ["Original (lr=1e-3)", "lr=1e-5", "lr=1e-4", "lr=5e-3"], title="Experiment 3: Q-learning Hyperparameter Tuning")

################################################################################

# Experiment 4



header1 = "hw3_q4_"
# contents = os.listdir(pwd)

# fields = ['Eval_AverageReturn', 'Eval_StdReturn']
# graph1_logs = []
# for item in contents:
#     if header1 in item:
#         if 'gym' in os.listdir(pwd+folder+item)[0]:
#             logname = pwd+folder+"/"+item+"/"+os.listdir(pwd+folder+item)[1]
#         else:
#             logname = pwd+folder+"/"+item+"/"+os.listdir(pwd+folder+item)[0]
#     if header1 in item:
#         graph1_logs.append((item, logname))

# graph1_logs.sort(key=lambda x: x[0])

# graph1_data = [
#     [
#         get_eval_results(log[1], field) for field in fields
#     ] for log in graph1_logs
# ]
# plot_lines(np.arange(len(graph1_data[0][0])), graph1_data, [log[0] for log in graph1_logs], title="Experiment 4:  Sanity check with Cartpole")

################################################################################


# Experiment 5a

# header1 = "hw3_q5_10_10_HalfCheetah"
# contents = os.listdir(pwd)

# fields = ['Eval_AverageReturn', 'Eval_StdReturn']
# graph1_logs = []
# for item in contents:
#     if header1 in item:
#         if 'gym' in os.listdir(pwd+folder+item)[0]:
#             logname = pwd+folder+"/"+item+"/"+os.listdir(pwd+folder+item)[1]
#         else:
#             logname = pwd+folder+"/"+item+"/"+os.listdir(pwd+folder+item)[0]
#     if header1 in item:
#         graph1_logs.append((item, logname))

# graph1_logs.sort(key=lambda x: x[0])

# graph1_data = [
#     [
#         get_eval_results(log[1], field) for field in fields
#     ] for log in graph1_logs
# ]
# plot_lines(np.arange(len(graph1_data[0][0])), graph1_data, [log[0] for log in graph1_logs], title="Experiment 5a: Half Cheetah")

################################################################################


# Experiment 5a

header1 = "hw3_q5_10_10_InvertedPendulum"
contents = os.listdir(pwd)

fields = ['Eval_AverageReturn', 'Eval_StdReturn']
graph1_logs = []
for item in contents:
    if header1 in item:
        if 'gym' in os.listdir(pwd+folder+item)[0]:
            logname = pwd+folder+"/"+item+"/"+os.listdir(pwd+folder+item)[1]
        else:
            logname = pwd+folder+"/"+item+"/"+os.listdir(pwd+folder+item)[0]
    if header1 in item:
        graph1_logs.append((item, logname))

graph1_logs.sort(key=lambda x: x[0])

graph1_data = [
    [
        get_eval_results(log[1], field) for field in fields
    ] for log in graph1_logs
]
plot_lines(np.arange(len(graph1_data[0][0])), graph1_data, [log[0] for log in graph1_logs], title="Experiment 5b: Inverted Pendulum")