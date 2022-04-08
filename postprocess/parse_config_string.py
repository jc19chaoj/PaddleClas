s_default = "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_ld_lookup_hit.sum+l1tex__t_sectors_pipe_tex_mem_texture_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_st_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_red_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_atom_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_ld_lookup_hit.sum+l1tex__t_sectors_pipe_tex_mem_texture_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_st_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_red_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_atom_lookup_hit.sum+l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_ld_lookup_miss.sum+l1tex__t_sectors_pipe_tex_mem_texture_lookup_miss.sum+l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_miss.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_st_lookup_miss.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_miss.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_red_lookup_miss.sum+l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_miss.sum+l1tex__t_sectors_pipe_tex_mem_surface_op_atom_lookup_miss.sum"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s","--str", type=str)
                        
args = parser.parse_args()

if args.str:
    s = args.str
    sp = ','
else:
    s = s_default
    sp = '+'

# print(s)
config_list = s.split(sp)
#print(config_list)
config_list = list(set(config_list))
print(config_list)

# sanity check
import pandas as pd
option_df = pd.read_csv('CUptiTracer_options.csv', delimiter=',')
all_options = [item for item in option_df.columns]

if all(option in all_options for option in config_list):
    print("Done!")
else:
    print("[Error] something is wrong!")