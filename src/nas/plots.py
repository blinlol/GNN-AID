# %%
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt

from seaborn import lineplot



def read_log(fname: str) -> pd.DataFrame:
    # прочитать логи в пандас
    val_accs = []
    with open(fname) as f:
        for line in f:
            if match := re.match(r"INFO:nas.trainer:eval.*(\[.*\]), (0\.\d*)", line):
                # sampled_gnn = match.group(1)
                val_acc = match.group(2)
                val_accs.append(float(val_acc))
    return pd.DataFrame(val_accs, columns=['val_acc'])


def plot_of_count(vals: pd.DataFrame, min_val: float = 0.8, title: str = None):
    x = vals.index
    y = []
    cnt = 0
    for val in vals.val_acc:
        if min_val <= val:
            cnt += 1
        y.append(cnt)
    df = pd.DataFrame({"step": x, "count": y})
    lineplot(df)

# %%

datasets = ["cora", "bzr", "pubmed", "citeseer", "mutag", "cox2", "aids", "proteins", 
            "computers", "mnistsuperpixels", "photo"]
logs_dir = "/home/ubuntu/GNN-AID/src/nas/logs/many_logs/"
logs = {d:[] for d in datasets}
for dir_name, _, fnames in os.walk(logs_dir):
    for f in fnames:
        for d in datasets:
            if d in f:
                logs[d].append(dir_name + f)
                break
    break

# %%
# вывести картинку количества архитектур преодолевших порог
min_val = 0.82
df = pd.DataFrame()
for log in logs["bzr"]:
    vals = read_log(log)
    x = vals.index
    if df.empty:
        df = pd.DataFrame(index=x)
    y = []
    cnt = 0
    for val in vals.val_acc:
        if min_val <= val:
            cnt += 1
        y.append(cnt)
    name = log.split('.')[0].split('/')[-1]
    if len(y) < len(df.index):
        y.extend([None for _ in range(len(df.index) - len(y))])
    elif len(y) > len(df.index):
        y = y[:len(df.index)]

    df['count ' + name] = y
lineplot(df).set_title("threshold " + str(min_val))

# %%
# вывести картинку максимальной точности
df = pd.DataFrame()
for log in logs["photo"]:
    vals = read_log(log)
    x = vals.index
    if df.empty:
        df = pd.DataFrame(index=x)
    y = []
    cnt = 0
    max_ = -1
    for val in vals.val_acc:
        if max_ < val:
            max_ = val
        y.append(max_)
    name = log.split('.')[0].split('/')[-1]
    if len(y) < len(df.index):
        y.extend([None for _ in range(len(df.index) - len(y))])
    elif len(y) > len(df.index):
        y = y[:len(df.index)]

    df[name] = y
plt.figure(figsize=(10, 8))
lineplot(df.loc[50:, :]).set_title("max accuracy")

# %%
# создать csv таблицу с данными о результатах
# dataset 1st_method_place ...

df_data = {
    'dataset': [],
}
datasets = ["cora", "bzr", "pubmed", "citeseer", "mutag", "cox2", 
            "computers", "photo"]
fname = 'results_argmax_epoch.csv'

for dataset in datasets:
    places = []
    for log in logs[dataset]:
        vals = read_log(log)
        name = log.split('.')[0].split('/')[-1]
        name = name[name.index('_') + 1:]
        # if name == "random":
        #     continue
        argmax = np.argmax(vals['val_acc'])
        places.append((vals['val_acc'][argmax], name, argmax))

    df_data['dataset'].append(dataset)
    for max_acc, name, argmax in places:
        if name not in df_data:
            df_data[name] = []
        df_data[name].append(argmax)
df = pd.DataFrame(df_data)
df.to_csv("/home/ubuntu/GNN-AID/src/nas/" + fname)

#%%
# создать csv с данными о количестве архитектур, преодолевших порог
# dataset threshold name_1 cnt_1

fname = 'results_cnt.csv'
methods_names = ["basic no-combinations",
                "basic combinations",
                "fixed no-combinations",
                "fixed combinations",
                "dynamic-probsonly no-combinations",
                "dynamic-probsonly combinations",
                "dynamic-both no-combinations",
                "dynamic-both combinations",
                "random"]

dataset_thresholds = {
    'cora': [0.87, 0.88, 0.89, 0.9],
    'citeseer': [0.74, 0.75, 0.76, 0.77],
    'pubmed': [0.84, 0.85, 0.86, 0.87],
    'photo': [0.91, 0.92, 0.93, 0.94],
    'computers': [0.86, 0.87, 0.88, 0.89],
    'mutag': [0.86, 0.89, 0.92, 0.95],
    'cox2': [0.84, 0.86, 0.88, 0.9],
    'bzr': [0.82, 0.84, 0.86, 0.88],
}

min_val = 0.93
data = {
    'dataset': [],
    'threshold': [],
    'method': [],
    'count': [],
}
# data.update(((name, []) for name in methods_names))
for dataset, thresholds in dataset_thresholds.items():
    name_th_cnt = []
    for log in logs[dataset]:
        vals = read_log(log)
        name = log.split('.')[0].split('/')[-1]
        name = name[name.index('_') + 1:]
        name = re.sub('_', ' ', name)
        for th in thresholds:
            cnt = 0
            for val in vals.val_acc:
                if th <= val:
                    cnt += 1
            name_th_cnt.append((name, th, cnt))
        
    for name in methods_names:
        for _, th, cnt in sorted(filter(lambda e: e[0] == name, name_th_cnt)):
            data["dataset"].append(dataset)
            data["threshold"].append(th)
            data["method"].append(name)
            data["count"].append(cnt)

df = pd.DataFrame(data)
df.to_csv(fname)

#%%

import seaborn as sn

sn.barplot(df, x='dataset', y='count', hue='method', errorbar=None)


#%%

# index = list(logs.keys())
# d = {

# }
# for dataset, fnames in logs:
#     for log in fnames:


# # field = "test_accuracy"
# field = "training_time"
# dataset = "BZR"
# t_optims = optimizers
# t_layers = ["GCNConv"]
# filename = f"tables/optim_{field}_{dataset}.txt"

# l_nums_by_dataset = {"Cora": [1, 2, 3], "BZR": [2, 3, 4]}
# l_nums = l_nums_by_dataset[dataset]

# num_cols = len(l_nums) * len(t_layers) + 1

# t_header = ("\\begin{table}[h!]\n"
#             "\\centering\n"
#             f"\\begin{{tabular}}{{{'c' * num_cols}}}\n")
# t_footer = ("\\end{tabular}\n"
#            "\\caption{Максимум полученных архитектур}\n"
#            "\\label{tab:max}\n"
#            "\\end{table}\n")

# with open(filename, "w") as f:
#     f.write(t_header)
#     f.write("Оптимайзер\t&\t")
#     for l in t_layers:
#         for n in l_nums:
#             f.write(f"{n}x{l}\t")
#             if l == t_layers[-1] and n == l_nums[-1]:
#                 f.write("\t\\\\\n")
#             else:
#                 f.write("&\t")
    
#     for o in t_optims:
#         f.write(f"{o}\t&\t")
#         for l in t_layers:
#             for n in l_nums:
#                 mean_val = mean.query(f"optimizer == '{o}' & layer == '{l}' & layers_number == {n} & dataset == '{dataset}'")[field].values
#                 if len(mean_val) != 1:
#                     continue
#                 mean_val = mean_val[0]

#                 var_val = var.query(f"optimizer == '{o}' & layer == '{l}' & layers_number == {n} & dataset == '{dataset}'")[field].values
#                 if len(var_val) != 1:
#                     continue
#                 var_val = var_val[0]

#                 f.write(f"{mean_val:.2f}±{var_val ** 0.5:.2f}\t")
#                 if l == t_layers[-1] and n == l_nums[-1]:
#                     f.write("\t\\\\\n")
#                 else:
#                     f.write("&\t")

#     f.write(t_footer)