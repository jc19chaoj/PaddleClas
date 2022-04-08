import os
import json
import pandas as pd

model_metrics = "model_metrics.json"

ROOT_PATH = "/paddle/chao/experiments/PaddleClas"
model_summary = ROOT_PATH+'/model_summary.json'

def json2df(json_file, top_key):
    with open(json_file, 'r') as json_f:
        json_data = json.load(json_f)[top_key]
        flat_data = [x for x in json_data]
        df_data = pd.json_normalize(flat_data)

    return df_data


metrics_df = json2df(model_metrics, "layers")
summary_df = json2df(model_summary, "layers")

merge_df = pd.merge(summary_df, metrics_df, on=["op_name","kernel.name","kernel.duration"])
merge_df.to_csv(model_metrics.split(".")[0]+".csv")

print("Done!")
