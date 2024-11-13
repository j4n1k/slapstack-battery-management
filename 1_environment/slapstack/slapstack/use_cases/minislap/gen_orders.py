import json
import os
import random
from operator import itemgetter

import numpy as np

# start_range = 0
# end_range = 86400
# num_delivery_order = 1000
# num_random = 1000
#
#
# def get_rand_orders(start, end, num):
#     random_numbers = []
#
#     while len(random_numbers) < num:
#         random_num = random.randint(start, end)
#         if random_num not in random_numbers:
#             random_numbers.append(random_num)
#
#     # print(random_numbers)
#     return random_numbers
#
#
# sample_orders = []
# delivery_order = []
# retrieval_order = []
# sorted_retrieval_order = []
#
#
# def get_sample_data(data):
#     # Process your data here
#
#     # print(len(data))
#     random_index = get_rand_orders(start_range, end_range, num_random)
#
#     for index, order in enumerate(data):
#         # order = item[1:]
#         if order[0] == 'delivery':
#             if index in random_index:
#                 order[3] = 1
#                 order[5] = 1
#                 delivery_order.append(order)
#         if len(delivery_order) == num_delivery_order:
#             break
#
#         # else:
#         #    retrieval_order.append(item)
#
#     delivery_order.sort(key=itemgetter(2, 1))
#
#     list_sku = check_inventory(delivery_order)
#
#     for order in data:
#         # order = item[1:]
#         if order[0] == 'retrieval':
#             if order[1] in list_sku:
#                 order[3] = 1
#                 order[5] = 1
#                 retrieval_order.append(order)
#
#     retrieval_order.sort(key=itemgetter(2, 1))
#
#     # sample_orders.extend(sorted(sample_orders, key=lambda x: x[1]))
#
#     print(delivery_order)
#     # print(sorted_retrieval_order)
#     print(retrieval_order)
#
#     for del_order in delivery_order:
#         for ret_order in retrieval_order:
#             if ret_order[1] == del_order[1] and ret_order[2] > del_order[2]:
#                 sorted_retrieval_order.append(ret_order)
#                 break
#
#     sorted_retrieval_order.sort(key=itemgetter(2, 1))
#     # sample_orders = delivery_order + sorted_retrieval_order
#     sample_orders = sorted_retrieval_order + delivery_order
#
#     print(len(sample_orders))
#     return sample_orders
#
#
# # check there are enough items for each sku to get retrieved by the retrieval order
# def check_inventory(data):
#     count_dict = {}
#
#     for order in data:
#         sku = order[1]
#         count_dict[sku] = count_dict.get(sku, 0) + 1
#     print(count_dict)
#     print(list(count_dict.keys()))
#     return list(count_dict.keys())
#
#
# # Get the directory path of the current script
# current_directory = os.path.dirname(os.path.abspath(__file__))
#
# # Construct the file path by joining the directory and the JSON file name
# json_file_path = os.path.join(current_directory, "2_orders.json")
#
# print(json_file_path)
#
# # Specify the path to your JSON file
# # json_file_path = "./2_orders.json"
#
# # Open the JSON file and load the data
# with open(json_file_path) as file:
#     order_data = json.load(file)
#
# # Process the data
# # sample_orders = get_sample_data(order_data)
# sample_orders = get_sample_data(order_data)
# check_inventory(delivery_order)
# check_inventory(sorted_retrieval_order)
#
# # Construct the file path by joining the directory and the JSON file name
# output_file_path = os.path.join(current_directory, "sample_orders.json")
#
# # Open the file in write mode and write the list as JSON
# print(sample_orders)
# with open(output_file_path, "w") as file:
#     json.dump(sample_orders, file, indent=3)


rng: np.random.default_rng = np.random.default_rng(seed=1)


def create_delivery_order(sku_counts: dict, i: int, random_sku,
                          sim_time: int, source: int):
    """get random sku, create delivery order, push to running event heap,
    and update sku_counts dictionary
    """
    sku = random_sku()
    sku_counts[sku] += 1
    # Convert all numpy integers to Python integers
    return ["delivery", int(sku), int(sim_time), int(source), 1, 1]


def create_retrieval_order(sku_counts: dict, i: int, sim_time: int,
                           sink: int, n_SKUs: int):
    """get random feasible sku, create retrieval order, push to running
    event heap, and update sku_counts dictionary
    """
    possible_skus = [sku for sku in range(1, n_SKUs + 1)]
    for j in possible_skus[:]:  # Create a copy to iterate over
        if sku_counts[j] == 0:
            possible_skus.remove(j)
    sku = rng.choice(possible_skus)
    sku_counts[sku] -= 1
    # Convert all numpy integers to Python integers
    return ["retrieval", int(sku), int(sim_time), int(sink), 1, 1]


def get_order_type(total_pallets: int, average_n_pallets) -> str:
    order_type = rng.choice(["delivery", "retrieval"])
    if total_pallets > 1.2 * average_n_pallets:
        order_type = "retrieval"
    if total_pallets < 0.8 * average_n_pallets:
        order_type = "delivery"
    return order_type


def get_order_time(n_storage_locs) -> int:
    mean_order_time = n_storage_locs * 0.4
    std_order_time = n_storage_locs * 0.1
    order_time = rng.normal(mean_order_time, std_order_time, 1)[0]
    return int(order_time)  # Convert numpy.float64 to Python int


def create_orders_from_distribution(n_sinks: int, n_sources: int, n_SKUs: int, n_orders: int, n_storage_locs: int,
                                    desired_fill_level: float):
    sim_time = 0
    orders = []

    def random_sku():
        return int(rng.integers(1, n_SKUs + 1))  # Convert numpy.int32 to Python int

    average_n_pallets = int(desired_fill_level * n_storage_locs)
    available_source_tiles = [i for i in range(0, n_sources)]
    available_sink_tiles = [i for i in range(0, n_sinks)]
    SKU_counts = {i: 0 for i in range(1, n_SKUs + 1)}

    for i in range(1, n_orders):
        total_pallets = sum(SKU_counts.values())
        order_time = get_order_time(n_storage_locs)
        order_type = get_order_type(total_pallets, average_n_pallets)

        if order_type == "delivery":
            source_index = int(rng.choice(available_source_tiles))  # Convert numpy.int32 to Python int
            order = create_delivery_order(SKU_counts, i, random_sku,
                                          sim_time, source_index)
        else:  # retrieval
            sink_index = int(rng.choice(available_sink_tiles))  # Convert numpy.int32 to Python int
            order = create_retrieval_order(SKU_counts, i,
                                           sim_time, sink_index, n_SKUs)
        sim_time += order_time
        orders.append(order)

    n_initial_pallets = average_n_pallets

    return orders


# Generate orders
orders = create_orders_from_distribution(
    n_sinks=2,
    n_sources=2,
    n_orders=300,
    n_SKUs=4,
    n_storage_locs=64,
    desired_fill_level=0.75
)

# Save to JSON file
current_directory = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(current_directory, "2_orders.json")

with open(output_file_path, "w") as file:
    json.dump(orders, file, indent=3)