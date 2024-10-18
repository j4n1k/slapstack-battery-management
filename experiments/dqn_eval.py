import os
import pandas as pd


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
# thresholds = [40, 50, 60, 70, 80, "dqn_best_model", "random", "no_battery_constraints", "sac"]
#thresholds = [40, 50, 60, 70, 80, 90, 100, "random", "no_battery_constraints", "sac", "dqn"]
thresholds = [30, 40, 50, 60, 70, 80, 90, 100, "random", "DQN", "SAC"] #"lower_bound", "SAC", "DQN"
for pt in partitions:
    print(f"Loading partition {pt}")
    dfs_d[pt] = []
    for th in thresholds:
        if not th == "SAC":
            path = f'{root_dir}/partition_{pt}/th{th}/COL'
        else:
            path = f'{root_dir}/partition_{pt}/th{th}/SAC'
        df = load_storage_strategy_dataframe(path)
        if df is not None:
            dfs_d[pt].append(df)
