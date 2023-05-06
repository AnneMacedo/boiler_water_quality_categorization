import random
import pandas as pd
import numpy as np


def get_list_random_param(value_max, value_min, k_choices=50):
    if value_max == value_min:
        step = value_max / 100 if value_max != 0 else 0.005
        sigma = value_max / 4 if value_max != 0 else 0.005

    elif value_max < 1 and value_max > -1:
        step = value_min / 10 if value_min != 0 else value_max / 10
        sigma = value_min / 2 if value_min != 0 else value_max / 2
    else:
        step = 0.13
        sigma = np.std([value_max, value_min])

    new_value_min = value_min - sigma
    new_value_max = value_max + sigma

    if new_value_min < 0:
        new_value_min = 0

    # print(f"    value_min={value_min}, new_value_min={new_value_min}")
    # print(f"    value_max={value_max}, new_value_max={new_value_max}")
    # print(f"    step={step}, sigma={sigma}")
    # print(f"")

    number_list = np.arange(new_value_min, new_value_max, step)
    return random.choices(number_list, k=k_choices)


param_data = pd.read_csv("..\data\param_data.csv")

random.seed(6523)

random_pressure = []
random_column = []
data = {}

for row in range(0, param_data.shape[0]):
    min_pressure = param_data.min_pressure[row]
    max_pressure = param_data.max_pressure[row]
    random_pressure_list = get_list_random_param(max_pressure, min_pressure)

    random_pressure.extend(random_pressure_list)
    data["pressure"] = random_pressure

    for column in param_data:
        if column not in ["min_pressure", "max_pressure"]:
            column_data = param_data[column][row:]
            min_value = column_data.min()
            max_value = column_data.max()

            # print(
            #     f"row_number={row}, column={column}, min={min_value}, max={max_value}"
            # )

            random_column_list = get_list_random_param(max_value, min_value)

            random_column = data.get(column, [])
            random_column.extend(random_column_list)

            data[column] = random_column
    # print("-----------------------------------")

synthetic_data = pd.DataFrame(data=data)
for column in synthetic_data:
    print(
        f"column={column}, min={synthetic_data[column].min()}, max={synthetic_data[column].max()}"
    )

synthetic_data.to_csv("..\data\synthetic_data.csv", index=False)
