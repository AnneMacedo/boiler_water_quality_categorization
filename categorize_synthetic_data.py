import pandas as pd


def pressure_class(pressure):
    if pressure <= 300:
        return 0
    elif pressure <= 450:
        return 1
    elif pressure <= 600:
        return 2
    elif pressure <= 750:
        return 3
    elif pressure <= 900:
        return 4
    elif pressure <= 1000:
        return 5
    elif pressure <= 1500:
        return 6
    else:
        return 7


def categorize_synthetic_data(row, param_data):
    pressure = row.pressure
    class_number = pressure_class(pressure)

    non_compliance = False

    for column in synthetic_data:
        if column != "pressure":
            if not row[column] <= param_data[column][class_number]:
                non_compliance = True
                break

    if not non_compliance:
        return "conforming"
    else:
        return "nonconforming"


param_data = pd.read_csv("param_data.csv")
synthetic_data = pd.read_csv("synthetic_data.csv")

synthetic_data["conformity_class"] = synthetic_data.apply(
    lambda row: categorize_synthetic_data(row, param_data), axis=1
)
# print(synthetic_data.iloc[0:20])

synthetic_data.to_csv("synthetic_conformity_data.csv", index=False)
