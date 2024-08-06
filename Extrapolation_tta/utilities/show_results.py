import os
from os.path import join
import numbers
from copy import copy
import json

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib.lines import Line2D

import wandb


# ------- loggers ----------------

class WANDBLogger(object):
    def __init__(self, checkpoint_path, wandb_run) -> None:
        self.wandb_run = wandb_run
        self.config_path = join(checkpoint_path, "config.json")
        self.log_path = join(checkpoint_path, "log.jsonl")

        config = dict(self.wandb_run.config)
        for key in ["tags", "notes", "group", "name"]:
            config[key] = getattr(self.wandb_run, key)
        with open(self.config_path, "w") as config_file:
            json.dump(config, config_file)

    def __call__(self, entry):
        # Log to wandb
        self.wandb_run.log(entry)

        # Also log to local history file
        with open(self.log_path, "a") as history_file:
            history_file.write(json.dumps(entry) + "\n")


def filtering(logs):
    # get the indices
    for key, vals in logs.items():
        if any(np.isnan(vals)):
            logs[key] = [val for val in vals if not np.isnan(val)]
        else:
            logs[key] = sorted(list(set(vals)))

# ------------ loading -----------------------------------
def get_raw_records(root_path_local, root_path_wandb, experiments, n_samples=500):
    # load the raw records from specified experiment text files that contain run paths
            
    # load the records from wandb
    records = []
    missing_local_records = []
    for exp in experiments:
        exp_folder_path = join(root_path_local, exp)
        if os.path.exists(exp_folder_path):
            # import ipdb; ipdb.set_trace()
            for run_folder_name in os.listdir(exp_folder_path):
                config_file_path = join(exp_folder_path, run_folder_name, "config.json")
                history_file_path = join(exp_folder_path, run_folder_name, "log.jsonl")
                if os.path.exists(config_file_path) and os.path.exists(history_file_path):
                    with open(config_file_path, "r") as config_file:
                        config = json.load(config_file)
                    _history = []
                    with open(history_file_path, "r") as history_file:
                        for line in history_file:
                            _history.append(json.loads(line))
                    history = {}
                    for step in _history:
                        for key, value in step.items():
                            history.setdefault(key, []).append(value)
                    records.append({"arguments": config, "runtime": history})
                else:
                    missing_local_records.append(run_folder_name)
        else:
            print(f"{exp_folder_path} does not exist!!!")
        
        # get the run paths
        run_ids = [
            run_folder_name.split("_")[-1] for run_folder_name in missing_local_records
        ]
        # for exp in experiments:
        #     exp_folder_path = join(root_path_local, exp)
        #     if os.path.exists(exp_folder_path):
        #         for run_folder_name in os.listdir(exp_folder_path):
        #             run_ids.append(run_folder_name.split("_")[-1]) # folder name: "*_runid"
        #     else:
        #         print(f"{exp_folder_path} does not exist!!!")

        api = wandb.Api()
        for run_id in run_ids:
            try:
                run = api.run(join(root_path_wandb, run_id))
            except:
                print(f"{run_id} is not found online -- skipped!")
                continue
            config = copy(run.config)
            for key in ["tags", "notes", "group", "name"]:
                config[key] = getattr(run, key)
            records.append({"arguments": config, "runtime": run.history(samples=n_samples).to_dict(orient="list")})
        
    return records

# ------------ filtering -----------------------------------

def filter_records(list_of_records, conditions):
    # load and filter data.
    records = []

    for raw_record in list_of_records:
        # check conditions.
        if not is_meet_conditions(raw_record["arguments"], conditions):
            continue

        # get parsed records
        records += [(raw_record["arguments"], raw_record["runtime"])]

    print("we have {}/{} records.".format(len(records), len(list_of_records)))
    return records


# def reorganize_record(record):
#     # collect each runtime item into a list
#     organized_record = defaultdict(list)
#     for stepstamp in record:
#         for key, value in stepstamp.items():
#             organized_record[key].append(value)
    
#     return organized_record

def _is_same(items):
    return len(set(items)) == 1


def is_meet_conditions(args, conditions, threshold=1e-8):
    if len(conditions) == 0:
        return True

    # get condition values and have a safety check.
    condition_names = list(conditions.keys())
    condition_values = list(conditions.values())
    assert _is_same([len(values) for values in condition_values]) is True

    # re-build conditions.
    num_condition = len(condition_values)
    num_condition_value = len(condition_values[0])
    condition_values = [
        [condition_values[ind_cond][ind_value] for ind_cond in range(num_condition)]
        for ind_value in range(num_condition_value)
    ]

    # check re-built condition.
    g_flag = False
    try:
        for cond_values in condition_values:
            l_flag = True
            for ind, cond_value in enumerate(cond_values):
                _cond = cond_value == (
                    args[condition_names[ind]] if condition_names[ind] in args else None
                )

                if isinstance(cond_value, numbers.Number):
                    _cond = (
                        _cond
                        or abs(cond_value - args[condition_names[ind]]) <= threshold
                    )

                l_flag = l_flag and _cond
            g_flag = g_flag or l_flag
        return g_flag
    except:
        return False

# --------------- to tables ------------------

"""summary the results."""

def summarize_info(
    records, arg_names, be_groupby="te_top1", larger_is_better=True, get_final=False
):
    # note that 'get_final' has higher priority than 'larger_is_better'.
    # define header.
    headers = arg_names + [be_groupby]
    # extract test records
    test_records = [
        _summarize_info(
            record, arg_names, be_groupby, larger_is_better, get_final=get_final
        )
        for record in records
    ]
    # aggregate test records
    aggregated_records = pd.DataFrame(test_records, columns=headers)
    # average test records
    averaged_records = (
        aggregated_records.fillna(-1)
        .groupby(headers[:-1], as_index=False)
        .agg({be_groupby: ["mean", "std", "max", "min", "count"]})
        .sort_values((be_groupby, "mean"), ascending=not larger_is_better)
    )
    return averaged_records

def _summarize_info(
    record, arg_names, be_groupby, larger_is_better, get_final=False
):
    args, info = record

    # if "test" in be_groupby or "train" in be_groupby:
    if not get_final:
        sorted_info = sorted(info[be_groupby], reverse=False)
        performance = sorted_info[-1] if larger_is_better else sorted_info[0]
    else:
        performance = info[be_groupby][-1]
    # else:
        # performance = args[be_groupby] if be_groupby in args else -1
        
    return [args[arg_name] if arg_name in args else None for arg_name in arg_names] + [performance]


# --------------- to figures ------------------

def reorder_records(records, based_on):
    # records is in the form of <args, info>
    conditions = based_on.split(",")
    list_of_args = [
        (ind, [args[condition] for condition in conditions])
        for ind, (args, info) in enumerate(records)
    ]
    sorted_list_of_args = sorted(list_of_args, key=lambda x: x[1:])
    return [records[ind] for ind, args in sorted_list_of_args]


def plot_curve_wrt_time(
    ax,
    records,
    x_wrt_sth,
    y_wrt_sth,
    xlabel,
    ylabel,
    title=None,
    markevery_list=None,
    is_smooth=True,
    smooth_space=100,
    l_subset=0.0,
    r_subset=1.0,
    reorder_record_item=None,
    remove_duplicate=True,
    legend=None,
    legend_loc="lower right",
    legend_ncol=2,
    bbox_to_anchor=[0, 0],
    ylimit_bottom=None,
    ylimit_top=None,
    use_log=False,
    num_cols=3,
):
    """Each info is a dictionary of lists {"epoch": [], "test_accuracy": []..}
    """
    # parse a list of records.
    num_records = len(records)
    distinct_conf_set = set()

    # re-order the records.
    if reorder_record_item is not None:
        records = reorder_records(records, based_on=reorder_record_item)

    count = 0
    for ind, (args, info) in enumerate(records):
        # build legend.
        _legend = build_legend(args, legend)
        if _legend in distinct_conf_set and remove_duplicate:
            continue
        else:
            distinct_conf_set.add(_legend)

        # split the y_wrt_sth if it can be splitted.
        if ";" in y_wrt_sth:
            has_multiple_y = True
            list_of_y_wrt_sth = y_wrt_sth.split(";")
        else:
            has_multiple_y = False
            list_of_y_wrt_sth = [y_wrt_sth]

        for _y_wrt_sth in list_of_y_wrt_sth:
            # determine the style of line, color and marker.
            line_style, color_style, mark_style = determine_color_and_lines(
                num_rows=num_records // num_cols, num_cols=num_cols, ind=count
            )
            if markevery_list is not None:
                mark_every = markevery_list[count]
            else:
                mark_style = None
                mark_every = None

            # update the counter.
            count += 1
        
            # get the data
            y = info[_y_wrt_sth]
            x = info[x_wrt_sth] if x_wrt_sth is not None else np.arange(len(y))

            if is_smooth:
                x, y = smoothing_func(x, y, smooth_space)

            # only plot subtsets.
            _l_subset, _r_subset = int(len(x) * l_subset), int(len(x) * r_subset)
            _x = x[_l_subset:_r_subset]
            _y = y[_l_subset:_r_subset]

            # use log scale for y
            if use_log:
                _y = np.log(_y)

            # plot
            ax = plot_one_case(
                ax,
                x=_x,
                y=_y,
                label=_legend if not has_multiple_y else _legend + f", {_y_wrt_sth}",
                line_style=line_style,
                color_style=color_style,
                mark_style=mark_style,
                mark_every=mark_every,
                remove_duplicate=remove_duplicate,
            )

    ax.set_ylim(bottom=ylimit_bottom, top=ylimit_top)
    ax = configure_figure(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        has_legend=legend is not None,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        bbox_to_anchor=bbox_to_anchor,
    )
    return ax


def smoothing_func(x, y, smooth_length=10):
    def smoothing(end_index):
        # print(end_index)
        if end_index - smooth_length < 0:
            start_index = 0
        else:
            start_index = end_index - smooth_length

        data = y[start_index:end_index]
        if len(data) == 0:
            return y[start_index]
        else:
            return 1.0 * sum(data) / len(data)

    if smooth_length == 0:
        _min_length = min(len(x), len(y))
        return x[:_min_length], y[:_min_length]

    # smooth curve
    x_, y_ = [], []

    for end_ind in range(0, len(x)):
        x_.append(x[end_ind])
        y_.append(smoothing(end_ind))
    return x_, y_



"""plot style related."""


def determine_color_and_lines(num_rows, num_cols, ind):
    line_styles = ["-", "--", "-.", ":"]
    color_styles = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]

    num_line_styles = len(line_styles)
    num_color_styles = len(color_styles)
    total_num_combs = num_line_styles * num_color_styles

    assert total_num_combs > num_rows * num_cols

    if max(num_rows, num_cols) > max(num_line_styles, num_color_styles):
        row = ind // num_line_styles
        col = ind % num_line_styles
        # print('plot {}. case 1, row: {}, col: {}'.format(ind, row, col))
        return line_styles[row], color_styles[col], Line2D.filled_markers[ind]

    denominator = max(num_rows, num_cols)
    row = ind // denominator
    col = ind % denominator
    # print('plot {}. case 2, row: {}, col: {}'.format(ind, row, col))
    return line_styles[row], color_styles[col], Line2D.filled_markers[ind]


def configure_figure(
    ax,
    xlabel,
    ylabel,
    title=None,
    has_legend=True,
    legend_loc="lower right",
    legend_ncol=2,
    bbox_to_anchor=[0, 0],
):
    if has_legend:
        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=legend_ncol,
            shadow=True,
            fancybox=True,
            fontsize=20,
        )

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel(ylabel, fontsize=24, labelpad=18)

    if title is not None:
        ax.set_title(title, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)
    return ax


def plot_one_case(
    ax,
    label,
    line_style,
    color_style,
    mark_style,
    line_width=2.0,
    mark_every=5000,
    x=None,
    y=None,
    sns_plot=None,
    remove_duplicate=False,
):
    if sns_plot is not None and not remove_duplicate:
        ax = sns.lineplot(
            x="x",
            y="y",
            data=sns_plot,
            label=label,
            linewidth=line_width,
            linestyle=line_style,
            color=color_style,
            marker=mark_style,
            markevery=mark_every,
            markersize=16,
            ax=ax,
        )
    elif sns_plot is not None and remove_duplicate:
        ax = sns.lineplot(
            x="x",
            y="y",
            data=sns_plot,
            label=label,
            linewidth=line_width,
            linestyle=line_style,
            color=color_style,
            marker=mark_style,
            markevery=mark_every,
            markersize=16,
            ax=ax,
            estimator=None,
        )
    else:
        ax.plot(
            x,
            y,
            label=label,
            linewidth=line_width,
            linestyle=line_style,
            color=color_style,
            marker=mark_style,
            markevery=mark_every,
            markersize=16,
        )
    return ax


def build_legend(args, legend):
    legend = legend.split(",")

    my_legend = []
    for _legend in legend:
        _legend_content = args[_legend] if _legend in args else -1
        my_legend += [
            "{}={}".format(
                _legend,
                list(_legend_content)[0]
                if "pandas" in str(type(_legend_content))
                else _legend_content,
            )
        ]
    return ", ".join(my_legend)