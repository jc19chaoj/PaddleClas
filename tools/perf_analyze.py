# "op_names","info","data","kernel_name","kernel_shape","kernel_duration"
# "info": "input_shape","output_shape","params","flops"
# type2name = {
#         nn.Conv1D: "",
#         nn.Conv2D: "conv2d",
#         nn.Conv3D: "",
#         nn.Conv1DTranspose: "",
#         nn.Conv2DTranspose: "",
#         nn.Conv3DTranspose: "",
#         nn.layer.norm.BatchNorm2D: "",
#         nn.BatchNorm: "batch_norm",
#         nn.ReLU: "relu",
#         nn.ReLU6: "",
#         nn.LeakyReLU: "",
#         nn.Linear: "matmul_v2",
#         nn.Dropout: "",
#         nn.AvgPool1D: "",
#         nn.AvgPool2D: "",
#         nn.AvgPool3D: "",
#         nn.AdaptiveAvgPool1D: "",
#         nn.AdaptiveAvgPool2D: "pool2d",
#         nn.AdaptiveAvgPool3D: ""
#     }
import ast
import json
from typing import OrderedDict
import numpy as np
import pandas as pd


# convert Python layer name to profiler op name
def type2name(layer_name):
    if layer_name.startswith("conv2d"):
        return "conv2d"
    elif layer_name.startswith("batch_norm"):
        return "batch_norm"
    elif layer_name.startswith("re_lu"):
        return "relu"
    elif layer_name.startswith("linear"):
        return "matmul_v2"
    elif layer_name.startswith("max_pool2d"):
        return "pool2d"
    elif layer_name.startswith("adaptive_avg_pool2d"):
        return "pool2d"
    elif layer_name.startswith("flatten"):
        return "flatten_contiguous_range"
    else:
        print(f"[WARNING] Layer {layer_name} not defined!")
        return 


# parse layer
def parse_layer(l):
    info_dict = {}
    
    # parse from paddle.model.flops() output
    input_shape = ast.literal_eval(l[1].strip())
    output_shape = ast.literal_eval(l[2].strip())
    params = int(l[3].strip())
    flops = int(l[4].strip())
    info_dict = {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'params': params,
        'flops': flops
    }

    # calculated
    data = np.prod(np.array(input_shape)) \
            + np.prod(np.array(output_shape)) \
            + params

    return info_dict, int(data)


# load correlation files to dataframes
def open_files(source_dir):
    pid = source_dir.split('/')[-1]
    root_path = f"{source_dir}"

    correlation_csv = f"{root_path}/correlation.csv"
    device_span = f"{root_path}/device_span_{pid}.json"
    op_span = f"{root_path}/op_span_{pid}.json"

    # correlation df
    header_list = ['op_id', 'runtime_id', 'kernel_id']
    correlation_df = pd.read_csv(correlation_csv, names=header_list)

    # op df
    with open(op_span, 'r') as op_f:
        op_json = json.load(op_f)['traceEvents']
        kernel_data = [x for x in op_json if x['cat'] == 'Operator']
        op_df = pd.json_normalize(kernel_data)

    # device df
    with open(device_span, 'r') as device_f:
        device_json = json.load(device_f)['traceEvents']
        kernel_data = [x for x in device_json if 'dur' in x and x['cat'] == 'Kernel']
        kernel_df = pd.json_normalize(kernel_data)

    return op_df, kernel_df, correlation_df


# parse kernel
def parse_kernel(layer_type, op_df, kernel_df, correlation_df):
    # find op idx
    op_name = type2name(layer_type)
    try:
        op_layer_df = op_df[op_df['name']==op_name].iloc[0]
        op_id = op_layer_df['id']

        # find correlated kernels
        kernel_op_df = correlation_df[correlation_df['op_id']==op_id]
        kernel_ids = kernel_op_df['kernel_id'][kernel_op_df['kernel_id'].notna()]
        kernel_correlated_df = kernel_df[kernel_df['id'].isin(kernel_ids)]

        return op_id, kernel_correlated_df
    except:
        print("[WARNING] {op_name} does not exist")
        return None, None
    

# Combine parse_layer and parse_kernel
def parse_model(summary_file, profile_output="13184_output/13187"):

    op_df, kernel_df, correlation_df = open_files(profile_output)

    model_dict = OrderedDict()
    model_dict['layers'] = []

    with open(summary_file, "r") as flops_f:
        line = flops_f.readline()
        start_parse = False
            
        while line:
            if line.startswith("+-"):
                start_parse = True
                break
            line = flops_f.readline()
            
        if start_parse:
            lines = flops_f.readlines()

        first_line = True
        for l in lines:
            if l.startswith("+-"):
                continue
            
            l = l.split("|")[1:-1]

            if first_line:
                first_line = False
            else:                
                layer_dict = OrderedDict()
                if l:
                    # build dict
                    # op
                    layer_info, data = parse_layer(l)
                    layer_type = l[0].strip()
                    layer_dict['op_name'] = layer_type
                    layer_dict['info'] = layer_info
                    layer_dict['data'] = data

                    # kernels
                    layer_dict['kernels'] = []
                    op_id, kernel_correlated_df = parse_kernel(layer_type, op_df, kernel_df, correlation_df)
                    
                    if op_id == None:
                        continue
                    else:
                        op_df = op_df.drop(op_id)

                        #print(kernel_df)
                        total_duration = 0
                        for _, df in kernel_correlated_df.iterrows():
                            kernel_dict = {
                                'name': df['name'],
                                'grid': df['args.grid'],
                                'block': df['args.block'],
                                'duration': df['dur']
                            }
                            layer_dict['kernels'].append(kernel_dict)
                            total_duration += df['dur']

                        layer_dict['total_kernel_duration'] = total_duration
                        model_dict['layers'].append(layer_dict)

    return model_dict


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path")
    args = parser.parse_args()
    #path = "29506_output/29509"

    model_info = parse_model(summary_file="model_summary.txt", 
                             profile_output=args.path)

    print("[INFO] Writing into model_summary.json...")
    with open("model_summary.json", "w") as output_f:
        output_f.write(json.dumps(model_info, indent=2))

    print("Done!")