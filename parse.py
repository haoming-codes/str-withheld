import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams["mathtext.fontset"] = "cm"
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
from operator import itemgetter
import paramiko
from scp import SCPClient
import ast
import re
import seaborn as sns

def createSSHClient(server, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, username=user, password=password)
    return client

def fileToList(file_dict):
    files = {name: [] for name in file_dict.keys()}
    for name, fnames in file_dict.items():
        run = 0
        for fname in fnames:
            print(fname, "opened")
            with open(fname, 'r') as file:
                content = file.read()
                if len(content) < 10: continue
                content = content[:-1]+"}"
                content = content.replace("nan", "np.nan")
                content = content.replace("array", "np.array")
                content = content.replace(", dtype=int64", "")
                content = content.replace("...,", "")
                content = re.sub("\[([\s\S]*?)\]", "[]", content)
                try: data = eval(content)
                except SyntaxError:
                    print("error reading", fname)
                    continue
            files[name] += list(data.values())
            print(fname, "processed, with", len(data), "runs")
        print(name, "processed, with", len(files[name]), "runs in total\n")
    return files

def fourBoxplots(dataset_exp_list, metric, bal):
    if bal:
        dataset_exp_list = {dataset: dataset_exp_list[dataset] for dataset in dataset_exp_list if dataset.endswith("_bal")}
    else:
        dataset_exp_list = {dataset: dataset_exp_list[dataset] for dataset in dataset_exp_list if not dataset.endswith("_bal")}
    
    print("\n",list(dataset_exp_list.keys()))
    if metric == "AUC":
        y_name = "AUC"
        order = plot_order_no_mincut
    elif metric == "PCC":
        y_name = "Accuracy"
        order = plot_order

    res = pd.DataFrame(columns=(y_name, "Dataset", "$\\epsilon$", "Classifier"))
    for file, content in dataset_exp_list.items():
        file_name_split = file.split("_")
        dataset = file_name_split[0].capitalize()
        epsilon = int(file_name_split[1])/10
        for exp in content:
            for test, metrics in exp.items():
                if test not in order: continue
                row = {y_name: metrics[metric][metric][1],
                       "Dataset": dataset,
                       "$\\epsilon$": epsilon,
                       "Classifier": order[test]}
                res = res.append(row, ignore_index=True)

    accuracy_table = res
    print(accuracy_table)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ds = ["Australia", "Germany", "Poland", "Taiwan"]
    count=0
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            a = sns.boxplot(x="$\\epsilon$", y=y_name, hue="Classifier",
                     data=accuracy_table[accuracy_table["Dataset"] == ds[count]], 
                     hue_order=list(order.values()), ax=col, showfliers=False, linewidth=0.7)
            a.get_legend().remove()
            a.title.set_text(ds[count])
            if i == 0: 
                a.get_xaxis().set_ticks([])
                a.get_xaxis().set_label_text("")
            if j == 1: 
            #     a.get_yaxis().set_ticks([])
                a.get_yaxis().set_label_text("")
            # a.set_ylim([0.5, 1.0])
            count+=1
    handles, labels = a.get_legend_handles_labels()
    # print(handles, labels)
    fig.legend(handles, labels, loc="lower center", ncol=len(order), prop={'size': 9})
    plt.subplots_adjust(bottom=0.15)
    plt.subplot_tool()
    plt.show()
    return

def printAcc(dataset_exp_list, met, ending=False):
    dataset_exp_list = {dataset: dataset_exp_list[dataset] for dataset in dataset_exp_list if dataset.endswith(ending)}
    # print("\n",list(dataset_exp_list.keys()))
    ending_split = ending.split("_")
    frac = ending_split[1]
    if len(ending_split) == 3:
        bal = "balanced"
    else:
        bal = "unbalanced"
    caption_ending = "$\\epsilon=0."+frac+"$, "+bal+" datasets"
    if met == "PCC":
        metric_full = "accuracy"
    elif met == "AUC":
        metric_full = "AUC"

    table = {}
    for test in test_dict.keys():
        # if test not in list(dataset_exp_list.values())[0][0].keys(): continue
        if test_dict[test][0] not in table:
            table[test_dict[test][0]] = {d: ["-", "-"] for d in file_dict.keys()}
        for name, data in dataset_exp_list.items():
            trains = []
            tests = []
            cv_accs = []
            cv_accs_training = []
            for exp in data:
                if test not in exp.keys(): continue
                if exp[test][met][met] == None: continue
                cv_accs.append(exp[test][met][met][1])
                cv_accs_training.append(exp[test][met][met][0])
            # print(test, name, "avg over", len(cv_accs), "runs")
            if len(cv_accs)==0 or np.isnan(cv_accs[0]) or cv_accs[0]==None: continue
            cell = ("%.3f" % np.mean(cv_accs))[1:]
            if test_dict[test][1] == "S":
                table[test_dict[test][0]][name][1] = cell
            elif test_dict[test][1] == "N":
                table[test_dict[test][0]][name][0] = cell
            elif test_dict[test][1] == "SN":
                table[test_dict[test][0]][name][0] = cell
                table[test_dict[test][0]][name][1] = cell

    for name in list(dataset_exp_list.keys()):
        best0 = 0
        best1 = 0
        for clf in test_order:
            if table[clf][name][0] == "-": continue
            if float(table[clf][name][0]) > best0: best0 = float(table[clf][name][0])
            if float(table[clf][name][1]) > best1: best1 = float(table[clf][name][1])
        for clf in test_order:
            if table[clf][name][0] == "-": continue
            if float(table[clf][name][0]) == best0: table[clf][name][0] = "{\\bf "+table[clf][name][0]+"}"
            if float(table[clf][name][1]) == best1: table[clf][name][1] = "{\\bf "+table[clf][name][1]+"}"


    print("\
\\begin{table*}\\centering\n\
\\caption{Our methods vs. the rest: mean classifier %s for %s (\"w/ disc.\" stands for \"with discretization of features\", \"Tru.\" for \"truthful reporting\", and \"Str.\" for \"strategic reporting\").}\n\
\\small\n\
\\begin{tabular}{@{}lccccccccccc@{}}\\toprule\n\
\\multirow{2}{*}[-3pt]{\makecell[l]{Classifier}} & \\multicolumn{2}{c}{Australia} & \\phantom{}& \\multicolumn{2}{c}{Germany} & \\phantom{} & \\multicolumn{2}{c}{Poland} & \\phantom{} & \\multicolumn{2}{c}{Taiwan}\\\\ \n\
\\cmidrule{2-3} \\cmidrule{5-6} \\cmidrule{8-9} \\cmidrule{11-12} \n\
& \\multicolumn{1}{c}{Tru.} & \\multicolumn{1}{c}{Str.} && \\multicolumn{1}{c}{Tru.} & \\multicolumn{1}{c}{Str.} && \\multicolumn{1}{c}{Tru.} & \\multicolumn{1}{c}{Str.} && \\multicolumn{1}{c}{Tru.} & \\multicolumn{1}{c}{Str.}\\\\ \\midrule"
           % (metric_full, caption_ending))

    for clf in test_order:
        if clf not in table.keys(): continue
        row = table[clf]
        print(clf+" & ", end="")
        for name in list(dataset_exp_list.keys()):
            if name == list(dataset_exp_list.keys())[-1]:
                print(row[name][0]+" & "+row[name][1]+"\\\\", end="")
            else:
                print(row[name][0]+" & "+row[name][1]+" && ", end="")

            if name == list(dataset_exp_list.keys())[-1] and clf == "{\\sc Mincut} w/ disc.":
                print("\\hline")
            elif name == list(dataset_exp_list.keys())[-1] and clf == "{\\sc Maj} w/ disc.":
                print("\\hdashline")
            elif name == list(dataset_exp_list.keys())[-1] and clf == "kNN (Imputation) w/ disc.":
                print("\\hdashline")
        print()

    print("\
\\bottomrule\n\
\\end{tabular}\n\
\\label{tab:%s,%s}\n\
\\end{table*}\n"
           % (metric_full, ending))
    print()

def printBase(dataset_exp_list, met, ending):
    dataset_exp_list = {dataset: dataset_exp_list[dataset] for dataset in dataset_exp_list if dataset.endswith(ending)}

    table = {}
    for test in base_dict.keys():
        if test not in base_dict.keys(): continue
        if base_dict[test][0] not in table:
            table[base_dict[test][0]] = {d: ["-", "-"] for d in file_dict.keys()}
        for name, data in dataset_exp_list.items():
            trains = []
            tests = []
            cv_accs = []
            cv_accs_training = []
            for exp in data:
                if test not in exp.keys(): continue
                if exp[test][met][met] == None: continue
                cv_accs.append(exp[test][met][met][1])
                cv_accs_training.append(exp[test][met][met][0])

            if len(cv_accs)==0 or np.isnan(cv_accs[0]) or cv_accs[0]==None: continue
            cell = ("%.3f" % np.mean(cv_accs))[1:]
            if base_dict[test][1] == "0":
                table[base_dict[test][0]][name][0] = cell
            elif base_dict[test][1] == "2":
                table[base_dict[test][0]][name][1] = cell

    print(table)
    for clf in base_order:
        row = table[clf]
        print(clf+" & ", end="")
        for name in file_dict_order:
            if name == file_dict_order[-1]:
                print(row[name][0]+" & "+row[name][1]+"\\\\", end="")
            else:
                print(row[name][0]+" & "+row[name][1]+" && ", end="")
        print()
        
file_dict = {
             "australia_0":     ["_australia_0.0.txt"],
             "australia_1":     ["_australia_0.1.txt"],
             "australia_2":     ["_australia_0.2.txt"],
             "australia_3":     ["_australia_0.3.txt"],
             "australia_4":     ["_australia_0.4.txt"],
             "australia_5":     ["_australia_0.5.txt"],
             "australia_0_bal": ["_australia_0.0_bal.txt"],
             "australia_1_bal": ["_australia_0.1_bal.txt"],
             "australia_2_bal": ["_australia_0.2_bal.txt"],
             "australia_3_bal": ["_australia_0.3_bal.txt"],
             "australia_4_bal": ["_australia_0.4_bal.txt"],
             "australia_5_bal": ["_australia_0.5_bal.txt"],
             "germany_0":       ["_germany_0.0.txt"],
             "germany_1":       ["_germany_0.1.txt"],
             "germany_2":       ["_germany_0.2.txt"],
             "germany_3":       ["_germany_0.3.txt"],
             "germany_4":       ["_germany_0.4.txt"],
             "germany_5":       ["_germany_0.5.txt"],
             "germany_0_bal":   ["_germany_0.0_bal.txt"],
             "germany_1_bal":   ["_germany_0.1_bal.txt"],
             "germany_2_bal":   ["_germany_0.2_bal.txt"],
             "germany_3_bal":   ["_germany_0.3_bal.txt"],
             "germany_4_bal":   ["_germany_0.4_bal.txt"],
             "germany_5_bal":   ["_germany_0.5_bal.txt"],
             "poland_0":        ["_poland_0.0.txt"],
             "poland_1":        ["_poland_0.1.txt"],
             "poland_2":        ["_poland_0.2.txt"],
             "poland_3":        ["_poland_0.3.txt"],
             "poland_4":        ["_poland_0.4.txt"],
             "poland_5":        ["_poland_0.5.txt"], 
             "poland_0_bal":    ["_poland_0.0_bal.txt"],
             "poland_1_bal":    ["_poland_0.1_bal.txt"],
             "poland_2_bal":    ["_poland_0.2_bal.txt"],
             "poland_3_bal":    ["_poland_0.3_bal.txt"],
             "poland_4_bal":    ["_poland_0.4_bal.txt"],
             "poland_5_bal":    ["_poland_0.5_bal.txt"],
             "taiwan_0":        ["_taiwan_0.0.txt"],
             "taiwan_1":        ["_taiwan_0.1.txt"],
             "taiwan_2":        ["_taiwan_0.2.txt"],
             "taiwan_3":        ["_taiwan_0.3.txt"],
             "taiwan_4":        ["_taiwan_0.4.txt"],
             "taiwan_5":        ["_taiwan_0.5.txt"],
             "taiwan_0_bal":    ["_taiwan_0.0_bal.txt"],
             "taiwan_1_bal":    ["_taiwan_0.1_bal.txt"],
             "taiwan_2_bal":    ["_taiwan_0.2_bal.txt"],
             "taiwan_3_bal":    ["_taiwan_0.3_bal.txt"],
             "taiwan_4_bal":    ["_taiwan_0.4_bal.txt"],
             "taiwan_5_bal":    ["_taiwan_0.5_bal.txt"]
             }

file_dict_order = [file for file in file_dict.keys()]

METRIC = ["PCC", "AUC", "F1", "Brier"]

plot_order = {'LR, greedy': "Hɪʟʟ-Cʟɪᴍʙɪɴɢ (LR)", 
              'LR, greedy, discretized': "Hɪʟʟ-Cʟɪᴍʙɪɴɢ (LR) w/ disc.", 
              'min-cut': "Mɪɴᴄᴜᴛ", 
              'min-cut, discretized': "Mɪɴᴄᴜᴛ w/ disc.", 
              'LR, imputation, strategic': "LR (Imp.)", 
              'LR, reduced-feature, strategic': "LR (R-F)"
              }
plot_order_no_mincut = {'LR, greedy': "Hɪʟʟ-Cʟɪᴍʙɪɴɢ (LR)", 
              'LR, greedy, discretized': "Hɪʟʟ-Cʟɪᴍʙɪɴɢ (LR) w/ disc.", 
              'LR, imputation, strategic': "LR (Imp.)", 
              'LR, reduced-feature, strategic': "LR (R-F)"
              }

test_dict = {
             'clustering, strategic': ("{\\sc Maj}", "S"),
             'clustering, discretized, strategic': ("{\\sc Maj} w/ disc.", "S"),
             'clustering, non-strategic': ("{\\sc Maj}", "N"),
             'clustering, discretized, non-strategic': ("{\\sc Maj} w/ disc.", "N"),
             'KN, imputation, strategic': ("kNN (Imputation)", "S"),
             'KN, imputation, non-strategic': ("kNN (Imputation)", "N"),
             'KN, imputation, discretzed, strategic': ("kNN (Imputation) w/ disc.", "S"),
             'KN, imputation, discretzed, non-strategic': ("kNN (Imputation) w/ disc.", "N"),
             'KN, reduced-feature, strategic': ("kNN (Reduced Feature)", "S"),
             'KN, reduced-feature, non-strategic': ("kNN (Reduced Feature)", "N"),
             'KN, reduced-feature, discretzed, strategic': ("kNN (Reduced Feature) w/ disc.", "S"),
             'KN, reduced-feature, discretzed, non-strategic': ("kNN (Reduced Feature) w/ disc.", "N"),
             'NN, imputation, strategic': ("ANN (Imputation)", "S"),
             'NN, imputation, non-strategic': ("ANN (Imputation)", "N"),
             'NN, imputation, discretzed, strategic': ("ANN (Imputation) w/ disc.", "S"),
             'NN, imputation, discretzed, non-strategic': ("ANN (Imputation) w/ disc.", "N"),
             'NN, reduced-feature, strategic': ("ANN (Reduced Feature)", "S"),
             'NN, reduced-feature, non-strategic': ("ANN (Reduced Feature)", "N"),
             'NN, reduced-feature, discretzed, strategic': ("ANN (Reduced Feature) w/ disc.", "S"),
             'NN, reduced-feature, discretzed, non-strategic': ("ANN (Reduced Feature) w/ disc.", "N"),
             'LR, imputation, strategic': ("LR (Imputation)", "S"),
             'LR, imputation, non-strategic': ("LR (Imputation)", "N"),
             'LR, imputation, discretzed, strategic': ("LR (Imputation) w/ disc.", "S"),
             'LR, imputation, discretzed, non-strategic': ("LR (Imputation) w/ disc.", "N"),
             'LR, reduced-feature, strategic': ("LR (Reduced Feature)", "S"),
             'LR, reduced-feature, non-strategic': ("LR (Reduced Feature)", "N"),
             'LR, reduced-feature, discretzed, strategic': ("LR (Reduced Feature) w/ disc.", "S"),
             'LR, reduced-feature, discretzed, non-strategic': ("LR (Reduced Feature) w/ disc.", "N"),
             'RF, imputation, strategic': ("RF (Imputation)", "S"),
             'RF, imputation, non-strategic': ("RF (Imputation)", "N"),
             'RF, imputation, discretzed, strategic': ("RF (Imputation) w/ disc.", "S"),
             'RF, imputation, discretzed, non-strategic': ("RF (Imputation) w/ disc.", "N"),
             'RF, reduced-feature, strategic': ("RF (Reduced Feature)", "S"),
             'RF, reduced-feature, non-strategic': ("RF (Reduced Feature)", "N"),
             'RF, reduced-feature, discretzed, strategic': ("RF (Reduced Feature) w/ disc.", "S"),
             'RF, reduced-feature, discretzed, non-strategic': ("RF (Reduced Feature) w/ disc.", "N"),
             'min-cut': ("{\\sc Mincut}", "SN"),
             'min-cut, discretized': ("{\\sc Mincut} w/ disc.", "SN"),
             'NN, greedy': ("HC (ANN)", "SN"),
             'NN, greedy, discretized': ("HC (ANN) w/ disc.", "SN"),
             'LR, greedy': ("HC (LR)", "SN"),
             'LR, greedy, discretized': ("HC (LR) w/ disc.", "SN")}
base_dict = {"KN, imputation, non-strategic, no fs": ("kNN (Imputation)", "2"),
             "KN, imputation, non-strategic, no fs, no dropping": ("kNN (Imputation)", "0"),
             "NN, imputation, non-strategic, no fs": ("ANN (Imputation)", "2"),
             "NN, imputation, non-strategic, no fs, no dropping": ("ANN (Imputation)", "0"),
             "LR, imputation, non-strategic, no fs": ("LR (Imputation)", "2"),
             "LR, imputation, non-strategic, no fs, no dropping": ("LR (Imputation)", "0"),
             "RF, imputation, non-strategic, no fs": ("RF (Imputation)", "2"),
             "RF, imputation, non-strategic, no fs, no dropping": ("RF (Imputation)", "0")
              }
test_order = ["HC (LR)",
              "HC (LR) w/ disc.",
              "HC (ANN)",
              "HC (ANN) w/ disc.",
              "{\\sc Mincut}",
              "{\\sc Mincut} w/ disc.",
              "{\\sc Maj}",
              "{\\sc Maj} w/ disc.",

              "LR (Imputation)",
              "LR (Imputation) w/ disc.",
              "ANN (Imputation)",
              "ANN (Imputation) w/ disc.",
              "RF (Imputation)",
              "RF (Imputation) w/ disc.",
              "kNN (Imputation)",
              "kNN (Imputation) w/ disc.",

              "LR (Reduced Feature)",
              "LR (Reduced Feature) w/ disc.",
              "ANN (Reduced Feature)",
              "ANN (Reduced Feature) w/ disc.",
              "RF (Reduced Feature)",
              "RF (Reduced Feature) w/ disc.",
              "kNN (Reduced Feature)",
              "kNN (Reduced Feature) w/ disc.",
              ]
base_order = ["LR (Imputation)",
              "ANN (Imputation)",
              "RF (Imputation)",
              "kNN (Imputation)"
              ]   

dataset_exp_list = fileToList(file_dict)

# Main result tables
for frac in ["0","1","2","3","4","5"]:
    printAcc(dataset_exp_list, "PCC", "_"+frac+"_bal")
for frac in ["0","1","2","3","4","5"]:
    printAcc(dataset_exp_list, "PCC", "_"+frac)
for frac in ["0","1","2","3","4","5"]:
    printAcc(dataset_exp_list, "AUC", "_"+frac+"_bal")
for frac in ["0","1","2","3","4","5"]:
    printAcc(dataset_exp_list, "AUC", "_"+frac)

# Boxplots
fourBoxplots(dataset_exp_list, "PCC", bal=True)
fourBoxplots(dataset_exp_list, "PCC", bal=False)
fourBoxplots(dataset_exp_list, "AUC", bal=True)
fourBoxplots(dataset_exp_list, "AUC", bal=False)

# use only the _2_bal files in file_dict
printBase(dataset_exp_list, "PCC", "_2_bal")
