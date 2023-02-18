import pandas as pd
from glob import glob
import argparse
import os
import re
import numpy as np
import yaml


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gather results and extract best hyperparameters for each strategy')   # noqa: E501
    parser.add_argument('--output-folder', "-o", type=str,  help="Path to directory containing validation results.", default=".")   # noqa: E501
    parser.add_argument('--dataset', "-d", type=str,  help="The FLamby dataset on which to test.", default="Fed-TCGA-BRCA")   # noqa: E501
    parser.add_argument('--seed', "-s", type=int, help="The seed for the dataset", default=42)   # noqa: E501

    args = parser.parse_args()

    # Find all results parquet files
    pq_files_list = glob(os.path.join(args.output_folder, "outputs", "*.parquet"))   # noqa: E501

    # Establish list of strategies
    strategies = {}
    new_dfs_list = []
    for i, pq in enumerate(pq_files_list):
        new_dfs_list.append(pd.read_parquet(pq))

    # Gather all results
    new_df = pd.concat(new_dfs_list, ignore_index=True)

    # For the right dataset
    new_df = new_df[new_df["data_name"] == (args.dataset + f"[seed={args.seed},test=val,train=fl]")]   # noqa: E501

    # Create startegy column
    new_df["strategy"] = [re.findall("[a-zA-Z].+(?=\[)", e)[0] for e in new_df["solver_name"].tolist()]   # noqa: E501 W605
    found_strategies = new_df["strategy"].unique()

    # Create all params columns by extracting all params for all strategies
    pnames = []
    for strat in found_strategies:
        strat_df = new_df[new_df["strategy"] == strat]
        # We can take any row to extract params
        first_row = strat_df.iloc[0]
        full_name = first_row["solver_name"]
        params_value_couples = re.findall("(?<=\[).+(?=\])", full_name)[0].split(",")    # noqa: E501 W605
        for pv in params_value_couples:
            pname, value = pv.split("=")
            if pname not in pnames:
                pnames.append(pname)

    # Create params columns using previous names (can probably be done
    # in one sweep with a one-liner)

    # helper function used just below
    def number_or_nan_filler(cell_value, name):
        param_value_list = re.findall("(?<=" + name + "=)([a-zA-Z-0-9]+[.]?[0-9]*)", cell_value)   # noqa: E501  W605
        if len(param_value_list) > 0:
            assert len(param_value_list) == 1
            # We currently can match all hyperparams either floats or boolean
            try:
                return float(param_value_list[0])
            except ValueError:
                try:
                    if not (param_value_list[0].lower() in ["true", "false"]):
                        raise ValueError
                    else:
                        return (param_value_list[0].lower() == "true")
                except ValueError:
                    raise ValueError(f"{param_value_list[0]} cannot be converted to float or boolean")   # noqa: E501
        else:
            return np.nan

    for pn in pnames:
        new_df[pn] = [number_or_nan_filler(e, pn) for e in new_df["solver_name"].tolist()]   # noqa: E501

    # We look for a specific hpset (note that we could also use solver_name)
    groupby_cols = ["strategy"] + pnames
    # We look at final performance to choose best hyperparameters
    idx_max_time = new_df.groupby(groupby_cols)['time'].transform(max) == new_df['time']   # noqa: E501
    assert (idx_max_time != (new_df.groupby(["solver_name"])["time"].transform(max) == new_df["time"])).sum() == 0   # noqa: E501
    # This way we keep only the final-value which we will be able to use
    # to filter the dataframe
    new_df_final_values = new_df[idx_max_time]

    cfg = {}
    cfg["n-repetitions"] = 1
    cfg["max-runs"] = 12
    # We will be testing on Test Now
    cfg["dataset"] = [args.dataset + "[seed=42,test=test,train=fl]"]

    for s in found_strategies:
        # We use the objective_value as target objective to minimize
        sorted_results = new_df_final_values[new_df_final_values["strategy"] == s].sort_values(by=['objective_value'])   # noqa: E501
        best_row = sorted_results.iloc[0]
        obj_value = best_row["objective_value"]
        avg_val_metric = best_row["objective_average_val_metric"]
        best_hparams = best_row[pnames]
        print(f"For strategy={s}, the best hparams: \n")
        print(f"{best_hparams} \n")
        print(f"give a final objective_value of {obj_value:.4f} with an average metric on val: {avg_val_metric:.4f}")   # noqa: E501
        if "solver" in cfg:
            cfg["solver"].append(best_row["solver_name"])
        else:
            cfg["solver"] = [best_row["solver_name"]]

    with open(f'best_config_test_{args.dataset}.yml', 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
