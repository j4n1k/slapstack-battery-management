import pandas as pd
from experiment_commons import gen_charging_stations, gen_charging_stations_left

use_case = "wepastacks"
n_cs = 1
layout = pd.read_csv(f"../1_environment/slapstack/slapstack/use_cases/{use_case}/1_layout.csv", header=None, delimiter=",")
path_out = f"../1_environment/slapstack/slapstack/use_cases/{use_case}_bm/1_layout.csv"
layout.dropna(axis=1, how='all', inplace=True)
layout_new = pd.DataFrame()
if use_case == "wepastacks":
    layout_new = gen_charging_stations(layout, n_cs)
elif use_case == "crossstacks":
    layout_new = gen_charging_stations_left(layout, n_cs)

layout_new.to_csv(path_out, header=None, index=False)
