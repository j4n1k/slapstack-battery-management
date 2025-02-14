from dotenv import load_dotenv
import os
import wandb
import tarfile
import os
import shutil

import pandas as pd
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from itertools import cycle

from tqdm.notebook import tqdm

from collections import namedtuple

# Load .env file
load_dotenv()

# Access the API key
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key)

api = wandb.Api()

# 20cs no_charge_in_breaks: 5ci1627k
# 4cs: smng8nfs
# 20cs charge_in_breaks (eval only): 5qdjkxob
# 20cs charge_in_breaks (train): q7y9vy3q
# Specify your project and run ID
entity = "j4b"        # Your WandB username or team name
project = "rl-battery-management" # Your WandB project name
run_id = "q7y9vy3q" # "f8l4u7x4" # "waj0fjc3"         # ID of the specific run

# Retrieve the run object
run = api.run(f"{entity}/{project}/runs/{run_id}")

n_cs = 20
num_partitions = 14
tar_name = 'experiment_data3'
root_dir = f'./result_data_charging_wepa/{n_cs}cs/cib'
root_dir_extraction = f'{root_dir}/result_data_remote3'

sns.set_style("whitegrid")
sns.set_context("talk")

hex_colors = ['#144246',
              #'#69657e',
              '#338470',
              '#a6874e',
              #'#FFFF33', '#FFD801', '#FFDF00',
              '#f2be25', '#e8dcb9']
pal = sns.color_palette(hex_colors, desat=1)
sns.palplot(pal)

shortnames = {
    'COL': 'COL',
    ' COL': 'COL',
    ' CTD': 'CTD',
    ' CTNR': 'CTNR',
    ' SL': 'SL',
    ' SLO': 'SLO',
    'RND':'RND',
    'allOrdersPopularity_future_z2': 'AOPF2',
    'allOrdersPopularity_future_z3': 'AOPF3',
    'allOrdersPopularity_future_z5': 'AOPF5',
    'allOrdersPopularity_past_z2': 'AOPP2',
    'allOrdersPopularity_past_z3': 'AOPP3',
    'allOrdersPopularity_past_z5': 'AOPP5',
    'classBasedCycleTime_z2': 'CBCT2',
    'classBasedCycleTime_z3': 'CBCT3',
    'classBasedCycleTime_z5': 'CBCT5',
    'retrievalPopularity_future_z2': 'ROPF2',
    'retrievalPopularity_future_z3': 'ROPF3',
    'retrievalPopularity_future_z5': 'ROPF5',
    'retrievalPopularity_past_z2': 'ROPP2',
    'retrievalPopularity_past_z3': 'ROPP3',
    'retrievalPopularity_past_z5': 'ROPP5',
    'VeryGreedy COL': 'GCOL',
    'DQN': 'COL',
    'SAC': 'SAC'
}


def order_files(src_dir):
    files = os.listdir(src_dir)
    pbar = tqdm(total=len(files))
    for i in range(len(files)):
        f_name = files[i]
        print(f_name)
        if not f_name.endswith('.csv') or 'actions' in f_name:
            continue
        segs = f_name.split('_')
        pt = segs[1]
        charging_policy = segs[3]
        th = segs[4]
        name = segs[2]
        new_name = shortnames[name]
        orders = f'{int(segs[-1].split(".")[0]):06}'
        tgt_dir = f'{src_dir}/partition_{pt}/{th}/{new_name}/'
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)
        shutil.move(f'{src_dir}/{f_name}',
                    f'{tgt_dir}/{orders}.csv')
        pbar.update(1)


order_files(root_dir)

def load_storage_strategy_dataframe(data_root):
    n_zones = 3
    try:
        strategy_name = data_root.split('/')[3]
        # n_zones = int(strategy_name[-1]) if strategy_name[-1].isdigit() else 3
    except:
        strategy_name = "DQN"
    if not os.path.exists(data_root):
        print(f"did not find path {data_root}; skipping...")
        return
    dfs = []
    csv_f_names = os.listdir(data_root)
    # pbar = tqdm(total=len(csv_f_names))
    print(f'Loading result files into dataframes for the '
          f'{strategy_name} simulation run...')
    for f_name in csv_f_names:
        if os.path.isdir(f'{data_root}/{f_name}') or f_name == '.DS_Store':
            #print(f'{data_root}/{f_name}')
            continue
        df_result_part = pd.read_csv(f'{data_root}/{f_name}', index_col=0)
        n_rows = df_result_part.shape[0]
        df_result_part['strategy_name'] = [strategy_name] * n_rows
        df_result_part['n_zones'] = [n_zones] * n_rows
        dfs.append(df_result_part)
        # print(strategy_name, n_zones, order_set_nr)
        # pbar.update(1)
    strategy_df = pd.concat(dfs).reset_index(drop=True)
    strategy_df.name = strategy_name
    return strategy_df

# dfs_d = dict({})
# n_agvs = 40
# for n_cs in range(1,9):
#     dfs_d[n_cs] = []
#     for shortname in shortnames.values():
#         df = load_storage_strategy_dataframe(
#             f'{root_dir}/n_agvs__{n_agvs}/n_cs__{n_cs}/{shortname}')
#         if df is not None:
#             dfs_d[n_cs].append(df)
dfs_d = dict({})
partitions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
thresholds = [30, 40, 50, 60, 70, 80]
for pt in partitions:
    print(f"Loading partition {pt}")
    dfs_d[pt] = []
    for th in thresholds:
        path = f'{root_dir}/partition_{pt}/th{th}/COL'
        df = load_storage_strategy_dataframe(path)
        if df is not None:
            dfs_d[pt].append(df)

_, ax = plt.subplots(figsize=(8, 4.5))
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

palette = cycle(pal)
for df in dfs_d[0]:
    ax = sns.lineplot(ax=ax, x='kpi__makespan', y='kpi__average_service_time',
                      label=df.name, color=next(palette), data=df)

    ax.legend(title='Partition')
    # ax.set_xlim((-5000, xlim + 20000))
    ax.set_xlabel('Time (in Seconds)')
    ax.set_ylabel('Average Service Time\n(in Seconds)')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.plot()

def gen_result_table(dfs_d, value, mode):
    list_of_dfs = []
    for idx in partitions:
        print("Loading partition", idx)
        # print(idx)
        df_results_pt = pd.DataFrame(columns=[f"{idx}"])
        for df in dfs_d[idx]:
            # print(df.name)
            # print(df["kpi__average_service_time"].mean())
            df = df.sort_values(by="kpi__makespan")
            if mode == "mean":
                row_data = pd.DataFrame(data={f"{idx}": [df[value].mean()]})  # df["kpi__makespan"].iloc[-1]]
            elif mode == "last":
                row_data = pd.DataFrame(
                    data={f"{idx}": [float(df[value].iloc[-1].round(decimals=2))]})  # df["kpi__makespan"].iloc[-1]]
            elif mode == "max":
                row_data = pd.DataFrame(data={f"{idx}": [df[value].max()]})
            # print(df["kpi__makespan"].iloc[-1])
            df_results_pt = pd.concat([df_results_pt, row_data])
        list_of_dfs.append(df_results_pt)
        print(len(df_results_pt))
    strategy_col = pd.DataFrame(columns=["Strategy"])
    for th in thresholds:
        row = pd.DataFrame(data={"Strategy": [th]})
        strategy_col = pd.concat([strategy_col, row])
    print(strategy_col)
    result_df = pd.DataFrame()
    result_df = pd.concat([result_df, strategy_col], axis=1)
    for df in list_of_dfs:
        result_df = pd.concat([result_df, df], axis=1)
    result_df = result_df.transpose().reset_index()
    new_header = result_df.iloc[0]
    result_df.drop(index=0, inplace=True)
    result_df.columns = new_header
    # result_df["Avg. overall"] = result_df.loc[:, [40, 50, 60, 70, 80, 90, 100, "no_battery_constraints"]].mean(axis = 1)
    # result_df["Std"] = result_df.iloc[:, 1:5].std(axis=1)
    # fill_lvl = [68, 87, 90, 77, 74, 86, 78, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # result_df.insert(11, "Fill Level", fill_lvl, True)
    return result_df


def gen_latex_tabel(subset_min, th_comp_df):
    th_comp_df.rename(columns={"lower_bound": "Lower Bound",
                               "Strategy": "Partition"}, inplace=True)

    th_comp_df = th_comp_df.set_index("Partition")

    round = th_comp_df.columns

    if "Std" in th_comp_df.columns:
        th_comp_view = th_comp_df.style.highlight_min(subset=subset_min, axis=1, props="font-weight:bold;").format({
            (val): '{:.2f}' for val in round}).text_gradient(cmap="rainbow", subset="Std", vmin=0, vmax=10)
    else:
        th_comp_view = th_comp_df.style.highlight_min(subset=subset_min, axis=1, props="font-weight:bold;").format({
            (val): '{:.2f}' for val in round})
    latex_string = th_comp_view.to_latex(convert_css=True, hrules=True)
    return latex_string


thresholds = [30, 40, 50, 60, 70, 80]
result_df = gen_result_table(dfs_d, "kpi__average_service_time", "last")

# Initialize lists to store the results
avg_service_times = []
max_service_times = []
ppo_results = {i: 0 for i in range(num_partitions)}
# Loop through each partition to retrieve and compute statistics
for i in range(num_partitions):
    # Construct the key for the specific partition
    key = f"logs/Evaluation/{i}/Servicetime"

    # Retrieve history for the current partition
    history_df = run.history(keys=[key])

    # Check if the data is available for this partition
    if key in history_df:
        # Calculate the average and max service time for this partition
        avg_service_time = history_df[key].iloc[-1]
        max_service_time = history_df[key].max()
        ppo_results[i] = avg_service_time

        # Append to the result lists
        avg_service_times.append((i, avg_service_time))
        max_service_times.append((i, max_service_time))

        # Print the results for this partition
        # print(f"Partition {i}:")
        # print(f"  Average Service Time: {avg_service_time}")
        # print(f"  Maximum Service Time: {max_service_time}")
    else:
        print(f"Data for Partition {i} not found.")
new_col = pd.DataFrame(data=[ppo_results]).transpose()
result_df.reset_index(inplace=True, drop=True)
result_df = pd.concat([result_df, new_col], axis=1)
result_df = result_df.rename(columns={0: "PPO"})
print(result_df)

result_melt = pd.melt(result_df, id_vars=["Strategy"], value_vars=[
    30, 40, 50, 60, 70, 80, "PPO"], var_name="Threshold")

th_comp_df = result_df[["Strategy", 30, 40, 50, 60, 70, 80, "PPO"]]
th_comp_df.rename(columns={"Strategy": "Partition"}, inplace=True)

th_comp_df = th_comp_df.set_index("Partition")
subset_min = [
    30, 40, 50, 60, 70, 80, "PPO"
]

round = [
    30, 40, 50, 60, 70, 80, "PPO"
]

th_comp_view = th_comp_df.style.highlight_min(subset=subset_min, axis=1, props="font-weight:bold;").format({
 (val): '{:.2f}' for val in round}).text_gradient(cmap="rainbow", subset="Std", vmin=0, vmax=10)
# latex_string = th_comp_view.to_latex(convert_css=True, hrules=True)

# subset_no_rl = [30, 40, 50, 60, 70, 80, 90, 100, "random"]
# subset_rl = [30, 40, 50, 60, 70, 80, 90, 100, "random", "PPO"]
# latex_string_no_rl = gen_latex_tabel(subset_no_rl, result_df[["Strategy", 30, 40, 50, 60, 70, 80, 90, 100, "random", "PPO"]])
# latex_string_rl = gen_latex_tabel(subset_rl, result_df[["Strategy", 30, 40, 50, 60, 70, 80, 90, 100, "random", "PPO"]])

# result_mean = result_df[[40, 50, 60, 70, 80, "random", "PPO"]].mean()
# result_mean.rename("Mean", inplace=True)
# result_std = result_df[[40, 50, 60, 70, 80, "random", "PPO"]].std()
# result_std.rename("Std", inplace=True)
# result_m_s = pd.concat([result_mean, result_std], axis=1)
# result_m_s.reset_index(inplace=True)
# # result_std
# result_m_s.rename(columns={0: "Strategy"}, inplace=True)
# result_m_s = result_m_s[["Strategy", "Mean"]]
# result_m_s = result_m_s.set_index("Strategy")
#
# subset_min = ["Mean"]
# round = ["Mean"]
# result_m_s_view = result_m_s.style.highlight_min(subset=subset_min, axis=1, props="font-weight:bold;").format({
#  (val): '{:.2f}' for val in round})
# latex_string = result_m_s_view.to_latex(convert_css=True, hrules=True)

Result = namedtuple('Result', [
    'total_distance',
    #     'average_distance',
    #     'travel_time_retrieval_ave',
    'total_shift_distance',
    'distance_retrieval_ave',
    'utilization_time',
    'makespan',
    'cycle_time',
    'entropy',
    'average_service_time',
    'throughput', 'max_delivery_buffer', 'max_retrieval_buffer', 'mean_retrieval_buffer',
    'max_agv_depleted', 'mean_agv_depleted',
    'index', 'name'])


def get_best_storage_strategies(experiment_dfs, n_best, exclude, scoring='average_service_time'):
    scores = []
    idx = 0
    for df in experiment_dfs:
        if df.name in exclude:
            print(df.name)
            continue
        df_sorted = df[[
            'total_distance',
            # 'average_distance',
            # 'travel_time_retrieval_ave',
            'total_shift_distance',
            'distance_retrieval_ave',
            'utilization_time',
            'kpi__makespan',
            'kpi__cycle_time',
            'entropy',
            'kpi__average_service_time',
            'kpi__throughput', 'n_finished_orders',
            'n_queued_delivery_orders', 'n_queued_retrieval_orders', 'n_agv_depleted']
        ].sort_values('kpi__makespan')
        if df.name == "th_no_battery_constraints":
            df_sorted.name = "lower bound"
        if df.name == "dqn" or df.name == "sac":
            df_sorted.name = df.name
        else:
            df_sorted.name = df.name  # .split("_")[1]

        end_row = df_sorted.iloc[-1, :]
        res = Result(
            end_row['total_distance'],
            # end_row['average_distance'],
            # end_row['travel_time_retrieval_ave'],
            end_row['total_shift_distance'],
            end_row['distance_retrieval_ave'],
            end_row['utilization_time'],
            end_row['kpi__makespan'],
            end_row['kpi__cycle_time'],
            end_row['entropy'],
            end_row['kpi__average_service_time'],
            end_row['kpi__throughput'],
            df_sorted['n_queued_delivery_orders'].max(),
            df_sorted['n_queued_retrieval_orders'].max(),
            df_sorted['n_queued_retrieval_orders'].mean(),
            df_sorted['n_agv_depleted'].max(),
            df_sorted['n_agv_depleted'].mean(),
            idx, df_sorted.name)
        scores.append(res)
        idx += 1
    scores_sorted = sorted(scores, key=lambda x: getattr(x, scoring))
    df_selection = []
    n_best = min(n_best, len(scores_sorted))
    print(n_best)
    for i in range(n_best):
        res = scores_sorted[i]
        print(res.average_service_time, i, experiment_dfs[scores_sorted[i].index].name)
        df_selection.append(experiment_dfs[scores_sorted[i].index])
    print("done")
    res_df = pd.DataFrame(data=scores_sorted)
    return df_selection, res_df


best_dfs_d = dict({})
res_overview_df_d = dict({})
for pt, dfs_s in dfs_d.items():
    print(f"Heuristic Service Time Ranks for partition {pt}:")
    best_dfs_d[pt], res_overview_df_d[pt] = get_best_storage_strategies(
        dfs_s, 5, [], 'average_service_time')

result_df.describe()

result_df_copy = result_df.copy()
res_paper_df = result_df.copy()
# result_df_copy = result_df_copy.drop([30, 100], axis=1)

idx = 11
res_paper_df["Min"] = result_df_copy.iloc[:, 1:idx].min(axis=1)
res_paper_df["Min Strat"] = result_df_copy.iloc[:, 1:idx].astype("float").idxmin(axis=1)
res_paper_df["Max"] = result_df_copy.iloc[:, 1:idx].max(axis=1)
res_paper_df["Max Strat"] = result_df_copy.iloc[:, 1:idx].astype("float").idxmax(axis=1)

res_paper_df = res_paper_df[["Strategy", "Min", "Min Strat", "Max", "Max Strat", "PPO"]]

print(res_paper_df)