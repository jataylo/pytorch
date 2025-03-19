import os
import glob
import pandas as pd

def process_csv(csv_path: str) -> pd.DataFrame:
    # --- 1) Read the CSV (adjust if your format differs) ---
    df = pd.read_csv(csv_path)
    
    # Identify columns
    all_cols = df.columns.tolist()
    input_col = 'input'     # known from problem statement
    timing_col = 'timing'   # known from problem statement
    
    # --- 2) Group by "input" to assign top‐N indicators within each group ---
    subframes = []
    for inp_val, group in df.groupby(input_col):
        # Sort ascending by timing
        group_sorted = group.sort_values(by=timing_col, ascending=True).reset_index(drop=True)
        
        # Add top‐N columns
        group_sorted['top_1'] = 0
        group_sorted['top_3'] = 0
        group_sorted['top_5'] = 0
        group_sorted['top_10'] = 0
        
        # Mark them according to sorted position
        for i in group_sorted.index:
            rank = i + 1
            if rank <= 10:
                group_sorted.at[i, 'top_10'] = 1
                if rank <= 5:
                    group_sorted.at[i, 'top_5'] = 1
                    if rank <= 3:
                        group_sorted.at[i, 'top_3'] = 1
                        if rank == 1:
                            group_sorted.at[i, 'top_1'] = 1
        
        # --- 3) Drop timing column ---
        group_sorted.drop(columns=[timing_col], inplace=True)
        
        # Collect in list
        subframes.append(group_sorted)
    
    # Combine all subframes
    combined_df = pd.concat(subframes, ignore_index=True)
    
    # --- 4) Count "occurrences" ignoring `input` ---
    # That means: for each row, how many times does its config appear
    # if we *ignore* the input column?
    top_cols = ['top_1', 'top_3', 'top_5', 'top_10']
    
    # All config columns ignoring input and the new top_N columns:
    config_cols_ignoring_input = [
        c for c in combined_df.columns 
        if c not in [input_col] + top_cols
    ]
    
    # For each row, "occurrences" = # of rows that share the same config_cols_ignoring_input
    combined_df['occurrences'] = combined_df.groupby(config_cols_ignoring_input)['input'].transform('size')
    
    # --- 5) Drop the `input` column now ---
    combined_df.drop(columns=[input_col], inplace=True)
    
    # --- 6) De‐duplicate ignoring top‐N columns and 'occurrences' ---
    # The columns that define a config are everything except top_N columns + 'occurrences'.
    dedup_key_cols = [
        c for c in combined_df.columns 
        if c not in (top_cols + ['occurrences'])
    ]
    
    # We'll group by those dedup keys, summing the top_N columns and
    # taking the "max" (or "first") of occurrences (they should be identical).
    agg_dict = {tc: 'sum' for tc in top_cols}
    agg_dict['occurrences'] = 'max'
    
    final_df = combined_df.groupby(dedup_key_cols, as_index=False).agg(agg_dict)
    
    # --- 7) Compute "score" with a heavier weighting for top_1 and lighter for top_10 ---
    # Example weighting: 10 for top_1, 5 for top_3, 3 for top_5, 1 for top_10
    final_df['score'] = (
          10 * final_df['top_1']
        +  5 * final_df['top_3']
        +  3 * final_df['top_5']
        +  1 * final_df['top_10']
    )
    
    # --- 8) Compute "averaged_score" = score / occurrences ---
    final_df['averaged_score'] = final_df['score'] / final_df['occurrences']
    
    return final_df


def main():
    input_dir = "autotune_ops"
    output_dir = os.path.join(input_dir, "top")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each CSV in autotune_ops
    for csv_file in glob.glob(os.path.join(input_dir, "*.csv")):
        print(f"Processing {csv_file}...")
        
        result_df = process_csv(csv_file)
        
        # Save to autotune_ops/top/<same filename>.csv
        out_path = os.path.join(output_dir, os.path.basename(csv_file))
        result_df.to_csv(out_path, index=False)
        
        print(f"  -> Wrote {out_path}")


if __name__ == "__main__":
    main()

