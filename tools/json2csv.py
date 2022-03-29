import json
import pandas as pd

def json2csv(model_summary_file):
    print(f"[INFO] Converting {model_summary_file} to csv format...")
    with open(model_summary_file, 'r') as input_f:
        json_data = json.load(input_f)['layers']
        flat_data = [x for x in json_data]
        df = pd.json_normalize(flat_data)

    df.to_csv(model_summary_file.split(".")[0]+".csv")

    print("Done!")

if __name__=="__main__":
    json2csv("model_summary.json")