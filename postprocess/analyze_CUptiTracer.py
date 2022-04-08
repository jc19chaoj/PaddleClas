import ast
import json
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p","--path",required=True,help="Path to CUptiTracer results, e.g., 33838_output/CUptiTracer_PID33983.")
parser.add_argument("-f","--fix",action="store_true")
args = parser.parse_args()

# file path
ROOT_PATH = "/paddle/chao/experiments-1/PaddleClas"
TRACER_PATH = ROOT_PATH+'/'+args.path
tracer_metrics = TRACER_PATH+'/'+"METRICS.json"
tracer_activity = TRACER_PATH+'/'+"ACTIVITY.json"

model_summary = ROOT_PATH+'/model_summary.json'


# metrics:
## json2dict
with open(tracer_metrics, 'r') as metrics_f:
    metrics_dict = ast.literal_eval(metrics_f.readline())
    #meta_info_dict = ast.literal_eval(metrics_f.readline())

metrics_list = metrics_dict['MetricNames']
#tracer_output = f"{TRACER_PATH}/{meta_info_dict['Address']}.json"
tracer_output = f"{TRACER_PATH}/{metrics_dict['Address'][0]}.json"


# activity:
## fix format
if args.fix:
    with open(tracer_activity, 'r+b') as act_f:
        # find the last ','
        act_f.seek(-1, os.SEEK_END)
        while act_f.read(1) != b',':
            act_f.seek(-2, os.SEEK_CUR)
        # replace it with a space
        act_f.seek(-1, os.SEEK_CUR)
        act_f.write(b' ')


## json2df
with open(tracer_activity, 'r') as act_f:
    act_json = json.load(act_f)['traceEvents']
    act_data = [x for x in act_json if x['cat'] == 'Kernel']
    act_df = pd.json_normalize(act_data)


# tracer output:
## json2dict
with open(tracer_output, 'r') as tracer_f:
    tracer_json = json.load(tracer_f)

tracer_dict = tracer_json['ProfilingData']


# model summary:
## json2df
with open(model_summary, 'r') as summary_f:
    json_data = json.load(summary_f)['layers']
    flat_data = [x for x in json_data]
    model_df = pd.json_normalize(flat_data)

print("[DEBUG]:",model_df.columns)
# correlate layers, kernels, and metrics:
all_layers = model_df['op_name']
all_kernels = model_df['kernel.name']
all_kernels_dur = model_df['kernel.duration']
all_ids = []

for kernel in all_kernels:
    for id, act in act_df[act_df['name']==kernel].iterrows():
        all_ids.append(id)
        act_df = act_df.drop(id)
        break

model_metrics_dict = {'layers': []}

# sanity check
assert(len(all_ids)==len(all_layers)==len(all_kernels)==len(all_kernels_dur))

for id, layer, kernel, dur in zip(all_ids, all_layers, all_kernels, all_kernels_dur):
    metrics = tracer_dict[str(id)]['Metrics']
    metrics_dict = dict(zip(metrics_list, metrics))
    metrics_dict['op_name'] = layer
    metrics_dict['kernel.name'] = kernel
    metrics_dict['kernel.duration'] = dur

    model_metrics_dict['layers'].append(metrics_dict)


# write output:
with open("model_metrics.json", 'w') as json_f:
    json_f.write(json.dumps(model_metrics_dict, indent=2))

print("Done!")
