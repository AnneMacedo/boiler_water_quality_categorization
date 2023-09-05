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
            row[f"{column}_class"] = (
                0 if not row[column] <= param_data[column][class_number] else 1
            )

    non_compliance = False

    for column in synthetic_data:
        if column != "pressure":
            if not row[column] <= param_data[column][class_number]:
                non_compliance = True
                break
    row["conformity_class"] = 0 if non_compliance else 1

    return row


synthetic_data = pd.read_csv("..\data\synthetic_data.csv")
param_data = pd.read_csv("..\data\param_data.csv")

synthetic_data = synthetic_data.apply(
    lambda row: categorize_synthetic_data(row, param_data), axis=1
)

synthetic_data.to_csv("..\data\synthetic_multiple_label_data.csv", index=False)
