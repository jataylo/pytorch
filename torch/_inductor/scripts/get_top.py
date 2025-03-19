import os
import glob
import pandas as pd

def process_csv(csv_path: str) -> pd.DataFrame:
    # Read CSV
    df = pd.read_csv(csv_path)
    
    input_col = "input"
    timing_col = "timing"
    
    # 1) Group by 'input' to mark top-n flags
    subframes = []
    for _, group in df.groupby(input_col):
        # Sort ascending by timing
        group_sorted = group.sort_values(by=timing_col, ascending=True).reset_index(drop=True)
        
        # Create top-n flags
        group_sorted["top_1"] = 0
        group_sorted["top_3"] = 0
        group_sorted["top_5"] = 0
        group_sorted["top_10"] = 0
        
        for i in group_sorted.index:
            rank = i + 1  # 1-based
            if rank <= 10:
                group_sorted.at[i, "top_10"] = 1
                if rank <= 5:
                    group_sorted.at[i, "top_5"] = 1
                    if rank <= 3:
                        group_sorted.at[i, "top_3"] = 1
                        if rank == 1:
                            group_sorted.at[i, "top_1"] = 1
        
        subframes.append(group_sorted)
    
    # Combine per-input subframes
    combined_df = pd.concat(subframes, ignore_index=True)
    
    # 2) Drop timing and input
    combined_df.drop(columns=[timing_col, input_col], inplace=True)
    
    # 3) De-duplicate ignoring top-n columns by summing them
    top_cols = ["top_1", "top_3", "top_5", "top_10"]
    dedup_cols = [c for c in combined_df.columns if c not in top_cols]
    
    agg_dict = {tc: "sum" for tc in top_cols}
    final_df = combined_df.groupby(dedup_cols, as_index=False).agg(agg_dict)
    
    # 4) Compute a "score" with heavier weight on top_1, etc.
    # Example weighting: top_1 * 10, top_3 * 5, top_5 * 3, top_10 * 1
    final_df["score"] = (
        10 * final_df["top_1"]
        +  5 * final_df["top_3"]
        +  3 * final_df["top_5"]
        +  1 * final_df["top_10"]
    )
    
    # 5) Sort by score descending
    final_df = final_df.sort_values(by="score", ascending=False).reset_index(drop=True)
    
    return final_df


def main():
    input_dir = "autotune_ops"
    output_dir = os.path.join(input_dir, "top")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print("No CSVs found in:", input_dir)
    
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        result_df = process_csv(csv_file)
        
        out_path = os.path.join(output_dir, os.path.basename(csv_file))
        result_df.to_csv(out_path, index=False)
        print(f"  -> Wrote {out_path}")


if __name__ == "__main__":
    main()

