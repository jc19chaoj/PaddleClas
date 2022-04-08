import shutil
import pandas as pd
import ast

# read all options
# with open('CUptiTracer_options.csv', newline='') as f:
#     all_options = csv.read(f)
option_df = pd.read_csv('CUptiTracer_options.csv', delimiter=',')
all_options = [item for item in option_df.columns]

all_necessary_options = [ 
                        "dram__bytes_read.sum.per_second", 
                        "dram__bytes_write.sum.per_second",
                        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                        "lts__t_sectors.sum.per_second",
                        "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
                        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
                        "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
                        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
                        "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
                        "lts__t_sector_hit_rate.pct",
                        "lts__t_sectors_lookup_hit.sum",
                        "lts__t_sectors_lookup_hit.sum",
                        "lts__t_sectors_lookup_miss.sum",
                        "lts__t_sectors_srcunit_tex_lookup_hit.sum",
                        "lts__t_sectors_srcunit_tex_lookup_hit.sum",
                        "lts__t_sectors_srcunit_tex_lookup_miss.sum",
                        "lts__t_sectors.sum",
                        "dram__bytes_write.sum",
                        "dram__bytes_read.sum",
                        "lts__t_sectors_srcunit_tex_op_read.sum",
                        "lts__t_sectors_srcunit_tex_op_write.sum",
                        "lts__t_sectors_srcunit_tex.sum",
                        "lts__t_sectors_srcunit_tex_aperture_device_lookup_miss.sum"
                        ]

# option groups
option_groups = {
    # DRAM throughput
    "dram_throughput":  [
                         "dram__bytes_read.sum.per_second",
                         "dram__bytes_write.sum.per_second"
                        ],

    # DRAM utilization
    "dram_utilization":  "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_utilization1": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_utilization2": [
                          "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
                          "dram__bytes_write.sum.pct_of_peak_sustained_elapsed"
                         ],
    
    # L2 throughput
    "l2_throughput":    "lts__t_sectors.sum.per_second",
    
    # L2 utilization
    "l2_utilization":   "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "l2_utilization1":  "lts__throughput.avg.pct_of_peak_sustained_elapsed",

    # sm__pipe
    "tensor_cycles":    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "alu_cycles":       "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    "fma_cycles":       "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "fp64_cycles":      "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",

    # L2 hit rate
    "L2_hit_rate":      "lts__t_sector_hit_rate.pct",

    # GPU hit rate
    "GPU_hit_rate":     [
                        "lts__t_sectors_lookup_hit.sum",
                        "lts__t_sectors_lookup_miss.sum"
                        ],

    # L1/TEXtotal hit rate
    "L1_hit_rate":      [
                        "lts__t_sectors_srcunit_tex_lookup_hit.sum",
                        "lts__t_sectors_srcunit_tex_lookup_miss.sum"
                        ]
}


# helper
def add_options(options):
    added_options = []
    for o in options:
        cupti_option = option_groups[o]
        
        if isinstance(cupti_option, list):
            if all(option in all_options for option in cupti_option):
                added_options += cupti_option
            else:
                print(f"[ERROR] option {cupti_option} is not supported!")
        else:
            if cupti_option in all_options:
                added_options.append(cupti_option)
            else:
                print(f"[ERROR] option {cupti_option} is not supported!")

    return added_options


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--list", action='store', type=str)
    parser.add_argument("-m","--metric_group", nargs="*", action='store', type=str)
    parser.add_argument("-n","--new", action='store', type=str)
    parser.add_argument("-a","--all", action='store_true')
    parser.add_argument("-p","--append", action='store_true')
    parser.add_argument("-y","--yes", action='store_true')                            
    args = parser.parse_args()

    if args.all:
        added_options = all_options
    
    if args.new:
        added_options = args.new.strip().split(",")

    if args.metric_group:
        added_options = add_options(args.metric_group)
    
    if args.list:
        added_options = ast.literal_eval(args.list)

    if args.append:
        with open("libCUptiTracer.conf", "r+") as conf_f:
            l = conf_f.readline()
            while not l.startswith("METRICS="):
                l = conf_f.readline()
            added_options += l.strip("METRICS=\n").split(",")
    
    added_options = list(set(added_options))
    
    print(added_options)
    
    with open("libCUptiTracer.conf", "w") as conf_f:
        conf_f.write("ACTIVITY=ENABLE\nPROFILING=ENABLE\n")
        conf_f.write("METRICS="+",".join(added_options))
        conf_f.write("\nKERNEL_CNT=100\nNUM_SESSION=12\n")
    
    # copy to /etc
    if args.yes:
        shutil.copy2("libCUptiTracer.conf", "/etc")
    else:
        ans = input("Copy to /etc/libCUptiTracer.conf? [Y/n] ") or "y"

        if ans[0].lower() == "y":
            print("Copying...")
            shutil.copy2("libCUptiTracer.conf", "/etc")


