import pandas as pd
from glob import glob
import argparse
import os
import re
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gather results and extract best hyperparameters for each strategy')
    parser.add_argument('--output-folder', "-o", type=str,  help="Path to directory containing validation results.", default=".")
    args = parser.parse_args()

    # Find all results parquet files
    pq_files_list = glob(os.path.join(args.output_folder, "outputs", "*.parquet"))

    # Establish list of strategies
    strategies = {}
    new_dfs_list = []
    for i, pq in enumerate(pq_files_list):
        new_dfs_list.append(pd.read_parquet(pq))

    # Gather results
    new_df = pd.concat(new_dfs_list, ignore_index=True)

    # Create startegy column
    new_df["strategy"] = [re.findall("[a-zA-Z].+(?=\[)", e)[0] for e in new_df["solver_name"].tolist()]
    found_strategies = new_df["strategy"].unique()

    # Create all params columns by extracting all params for all strategies
    pnames = []
    for strat in found_strategies:
        strat_df = new_df[new_df["strategy"] == strat]
        # We can take any row to extract params
        first_row = strat_df.iloc[0]
        full_name = first_row["solver_name"]
        params_value_couples = re.findall("(?<=\[).+(?=\])", full_name)[0].split(",")
        for pv in params_value_couples:
            pname, value = pv.split("=")
            if pname not in pnames:
                pnames.append(pname)

    # Create params columns using previous names (can probably be done in one sweep with a one-liner)

    # helper function used just below
    def number_or_nan_filler(cell_value, name):
        param_value_list = re.findall("(?<=" + name + "=)([a-zA-Z-0-9]+[.]?[0-9]*)", cell_value)
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
                        return (param_value_list[0] == "true")
                except ValueError:
                    raise ValueError(f"{param_value_list[0]} cannot be converted to float or boolean")
        else:
            return np.nan

    for pn in pnames:
        new_df[pn] = [number_or_nan_filler(e, pn) for e in new_df["solver_name"].tolist()]

    # We look for a specific hpset note that we could also use solver_name
    groupby_cols = ["strategy"] + pnames
    # We look at final performance to choose best hyperparameters
    idx_max_time = new_df.groupby(groupby_cols)['time'].transform(max) == new_df['time']
    # This way we keep only the final-value which we will be able to use to filter the dataframe
    new_df_final_values = new_df[idx_max_time]

    for s in found_strategies:
        # We use the performance
        best_row = new_df_final_values[new_df_final_values["strategy"] == s].sort_values(by=['objective_value'])










