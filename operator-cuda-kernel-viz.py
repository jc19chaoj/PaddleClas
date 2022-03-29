import dash
from dash import dcc
from dash import html
from dash import dash_table
import pandas as pd
import numpy as np
import json
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import swifter
from plotly.subplots import make_subplots
from plotly import offline
from flask_caching import Cache
from dash.dependencies import Input, Output, State
import plotly.figure_factory as ff
import argparse


TRACE_EVENT_KEY = 'traceEvents'
OPERATOR_KEY = 'cat'
OPERATOR_VAL = 'Operator'
DUR_KEY = 'dur'
START_KEY = 'ts'

app = dash.Dash(__name__)
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

def assign_correlation_id(df, cor_df):
    """
    merge kernel/operator dataframe with correlation dataframe
    """
    return pd.merge(df, cor_df, how='left', on=['id'])

def loadOperatorDF(json_file):
    """
    load json file and filter out Kernel, Runtime and Operator records
    """
    f = open(json_file)
    trace_data = json.load(f)[TRACE_EVENT_KEY]
    operator_data = [x for x in trace_data if DUR_KEY in x and
                     (x[OPERATOR_KEY] == 'Kernel' or x[OPERATOR_KEY] == 'Runtime' or x[OPERATOR_KEY] == 'Operator')]
    return pd.json_normalize(operator_data)

def cal_end_time(df):
    """
    calculate end time for each record
    """
    soted_df = df.sort_values(by=[START_KEY])
    soted_df['Start'] = soted_df[START_KEY] - soted_df.iloc[0][START_KEY]
    return soted_df

def update_grid(row, key):
    """
    Convert grid type from list to string to appear on plot
    """
    if row[key] == ' ':
        return row[key]
    else:
        return '[' + ", ".join([str(a) for a in row[key]]) + ']'

def calculate_grid(row):
    """
    calculate grid size for height display
    """
    grid = row['Grid']
    if grid == ' ':
        return 1
    else:
        ans = 1
        grid_list = row['args.grid']
        for i in grid_list:
            ans *= i
        return ans

def show_ops_timeline(df):
    """
    given a processed dataframe, plot interactive timeline
    """
    fig = px.bar(df,
                 x='duration',
                 y='Profiler',
                 color='operator',
                 base='ts',
                 orientation='h',
            #     width='grid_size',
                 category_orders={'Profiler':['GPU Kernel', 'CUDA API', 'Operator']},
                 hover_data=['cat','Grid'],
                 text='name')

    fig.update_traces(textangle=0)
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="linear"), yaxis=dict(autorange='reversed'), legend_title_text='Operator ID')

  #  fig.update_traces(mode="lines")
    fig.update_yaxes(categoryarray=['Operator','CUDA API','GPU Kernel'])
    fig.update_xaxes(
        showspikes=True,
        title='Duration',
        spikecolor="black",
        spikesnap="cursor",
        spikemode="across",
        spikedash="dot",
        spikethickness=1.,

    )
    #fig.write_html("./timeline.html")
    return fig

def get_correlation_df(corr_file):
    """
    load correlation result and join with both runtime records and operator records
    """
    header_list = ['OperatorId', 'RuntimeId', 'KernelId']
    sep_df_name = ['id', 'index']
    correlation_df = pd.read_csv(corr_file, names=header_list)
 #   correlation_df['index'] = correlation_df.index
    op_cor_df = correlation_df[['OperatorId']]
    op_cor_df.rename(columns={'OperatorId':'id'}, inplace=True)
    runtime_cor_df = correlation_df[['RuntimeId', 'OperatorId']]
    runtime_cor_df.rename(columns={'RuntimeId':'id'}, inplace=True)
    kernel_cor_df = correlation_df[['KernelId', 'OperatorId']]
    kernel_cor_df.rename(columns={'KernelId':'id'}, inplace=True)
  #  print(op_cor_df)
    runtime_df = pd.concat([runtime_cor_df, kernel_cor_df])
    return op_cor_df, runtime_df

@cache.memoize(timeout=600)
def get_data(OPERATOR_FILE, DEVICE_FILE, CORR_FILE):
    """
    load both device span and operator span to different dataframes
    separate operator span to CUDA API and GPU kernel dataframes
    find the correlation betweem device dataframe (operator), CUDA API and Kernel by OperatorID
    """
    columns_needed = ['id', 'cat', 'name', START_KEY, 'dur', 'Profiler', 'args.grid', 'OperatorId']
    device_df = loadOperatorDF(DEVICE_FILE)
    op_cor_df, runtime_df = get_correlation_df(CORR_FILE)
    device_df.loc[device_df['cat'] == 'Kernel', 'Profiler'] = 'GPU Kernel'
    device_df.loc[device_df['cat'] != 'Kernel', 'Profiler'] = 'CUDA API'
    # new_frame = device_df
    #    new_frame['OperatorId'] = new_frame['args.correlation']
    device_cor_op_df = pd.merge(device_df, runtime_df, how='left', on=['id'])
    new_device_df = device_cor_op_df[columns_needed]
    ops_df = loadOperatorDF(OPERATOR_FILE)
    top_ops_df = pd.merge(op_cor_df, ops_df, how='left', on=['id'])
    top_ops_df['Profiler'] = 'Operator'
    top_ops_df['args.grid'] = ' '
    top_ops_df['OperatorId'] = top_ops_df['id']
    new_ops_df = top_ops_df[columns_needed]
    new_frame = pd.concat([new_device_df, new_ops_df], ignore_index=True)
    new_frame['args.grid'] = new_frame['args.grid'].fillna(' ')
    new_frame['Grid'] = new_frame.swifter.apply(lambda x: update_grid(x, 'args.grid'), axis=1)
    print(new_frame)
    new_frame['grid_size'] = new_frame.swifter.apply(lambda x: calculate_grid(x), axis=1)
    kernel_df = new_frame.loc[new_frame['Profiler'] == 'GPU Kernel']
    other_df = new_frame.loc[new_frame['Profiler'] != 'GPU Kernel']
    kernel_df_max_scaled = kernel_df.copy()
    # apply normalization techniques on Column 1
    column = 'grid_size'
    kernel_df_max_scaled[column] = kernel_df_max_scaled[column] / kernel_df_max_scaled[column].abs().max()
    new_norm_frame = pd.concat([kernel_df_max_scaled, other_df], ignore_index=True)
    new_norm_frame['ts'] = new_norm_frame[START_KEY]
    new_norm_frame['duration'] = new_norm_frame['dur']
    new_norm_frame['operator'] = new_norm_frame['OperatorId'].swifter.apply(str)
    return new_norm_frame.sort_values(by=['OperatorId'])

def main():
    """
    establish a dashboard, which contains a timeline display and related table
    """
    parser = argparse.ArgumentParser(description='Operator, CUDA API and GPU kernel visualization')
    parser.add_argument('-p', '--pid', help='the pid of profiling result')
    args = parser.parse_args()

    PID = str(args.pid)
    OPERATOR_FILE = './' + PID + '/op_span_' + PID + '.json'
    DEVICE_FILE = './' + PID + '/device_span_' + PID + '.json'
    CORR_FILE = './' + PID + '/correlation.csv'

    columns_needed = ['id', 'cat', 'name', START_KEY, 'duration']
    columns_order = ['OperatorId', 'name', START_KEY, 'duration', 'kernel launch time', 'kernel launch duration', 'kernel start time', 'kernel duration']
    df = get_data(OPERATOR_FILE, DEVICE_FILE, CORR_FILE)
    dedupe_df = df[df['Profiler'] == 'Operator'][columns_needed].drop_duplicates()
    kernel = df[df['cat'] == 'Kernel'][['OperatorId', START_KEY, 'dur']]
    kernel.rename(columns={'OperatorId':'id', START_KEY:'kernel start time', 'dur': 'kernel duration'}, inplace=True)
    cuda = df[df['name'] == 'cudaLaunchKernel'][['OperatorId', START_KEY, 'dur']]
    cuda.rename(columns={'OperatorId':'id', START_KEY:'kernel launch time', 'dur': 'kernel launch duration'}, inplace=True)
    merge_tmp = pd.merge(dedupe_df, cuda, on='id', how='left')
    merge = pd.merge(merge_tmp, kernel, on='id', how='left')
    merge.rename(columns={'id':'OperatorId'}, inplace=True)
    merge = merge[columns_order]

    app.layout = html.Div(
        className="row",
        children=[
            html.Div(
                id='table-paging-with-graph-container',
                className="five columns"
            ),
            html.Div(
                dash_table.DataTable(
                    id='table-paging-with-graph',
                    columns=[
                        {"name": i, "id": i} for i in merge.columns
                    ],
                    page_current=0,
                    page_size=50,
                    page_action='custom',
                 #   page_action="native",
                    filter_action='custom',
                    filter_query='',

                    sort_action='custom',
                    sort_mode='multi',
                    sort_by=[]
                ),
                style={'height': 350, 'overflowY': 'scroll'},
                className='six columns'
            ),
        ]
    )

    operators = [['ge ', '>='],
                 ['le ', '<='],
                 ['lt ', '<'],
                 ['gt ', '>'],
                 ['ne ', '!='],
                 ['eq ', '='],
                 ['contains '],
                 ['datestartswith ']]

    def split_filter_part(filter_part):
        for operator_type in operators:
            for operator in operator_type:
                if operator in filter_part:
                    name_part, value_part = filter_part.split(operator, 1)
                    name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                    value_part = value_part.strip()
                    v0 = value_part[0]
                    if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                        value = value_part[1: -1].replace('\\' + v0, v0)
                    else:
                        try:
                            value = float(value_part)
                        except ValueError:
                            value = value_part

                    # word operators need spaces after them in the filter string,
                    # but we don't want these later
                    return name, operator_type[0].strip(), value

        return [None] * 3

    @app.callback(
        Output('table-paging-with-graph', "data"),
        Input('table-paging-with-graph', "page_current"),
        Input('table-paging-with-graph', "page_size"),
        Input('table-paging-with-graph', "sort_by"),
        Input('table-paging-with-graph', "filter_query"))
    def update_table(page_current, page_size, sort_by, filter):
        """
        update table based on filters
        """
        filtering_expressions = filter.split(' && ')
        dff = merge
        for filter_part in filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)

            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                dff = dff.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                # this is a simplification of the front-end filtering logic,
                # only works with complete fields in standard format
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]

        if len(sort_by):
            dff = dff.sort_values(
                [col['column_id'] for col in sort_by],
                ascending=[
                    col['direction'] == 'asc'
                    for col in sort_by
                ],
                inplace=False
            )

        return dff.iloc[
               page_current * page_size: (page_current + 1) * page_size
               ].to_dict('records')

    @app.callback(
        Output('table-paging-with-graph-container', "children"),
        Input('table-paging-with-graph', "data"))
    def update_graph(rows):
        """
        update graph based on records in the table
        """
        dff = pd.DataFrame(rows)
        op_list = dff['OperatorId'].to_list()
        selected_df = df[df['OperatorId'].isin(op_list)]
        fig = show_ops_timeline(selected_df)
        return html.Div(
            [
                dcc.Graph(
                    id='graph',
                    figure=fig
                )
            ]
        )

    app.run_server(debug=False)

if __name__ == '__main__':
    main()
