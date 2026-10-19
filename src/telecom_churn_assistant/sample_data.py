import json
from importlib.resources import files


with files(__package__).joinpath("sample_data.json").open(
    encoding="utf-8"
) as data_file:
    CUSTOMERS = {
        customer["customerId"]: customer
        for customer in json.load(data_file)["customers"]
    }
