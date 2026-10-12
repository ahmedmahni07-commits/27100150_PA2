import pandas as pd

def extract_dpo_summary():
    # Load the summary results from the current directory
    df = pd.read_csv("beta_summary.csv")

    # Select the specific columns required for the Task 1 evidence
    report_cols = [
        "condition", "budget", "beta", "heldout_dpo_loss", 
        "heldout_pref_acc", "kl_token_mean", "rm_score_mean", 
        "len_mean", "len_std"
    ]
    summary_table = df[report_cols].copy()

    # Format the metrics to match standard reporting precision
    summary_table["heldout_dpo_loss"] = summary_table["heldout_dpo_loss"].round(3)
    summary_table["heldout_pref_acc"] = (summary_table["heldout_pref_acc"] * 100).round(1).astype(str) + "%"
    summary_table["kl_token_mean"] = summary_table["kl_token_mean"].round(4)
    summary_table["rm_score_mean"] = summary_table["rm_score_mean"].round(3)
    summary_table["len_mean"] = summary_table["len_mean"].round(2)
    summary_table["len_std"] = summary_table["len_std"].round(2)

    # Combine Length Mean and Std into a single column for the report
    summary_table["Length (Mean ± Std)"] = summary_table["len_mean"].astype(str) + " ± " + summary_table["len_std"].astype(str)
    summary_table = summary_table.drop(columns=["len_mean", "len_std"])

    # Rename columns to be human-readable
    summary_table = summary_table.rename(columns={
        "condition": "Condition",
        "budget": "Budget",
        "beta": "Beta",
        "heldout_dpo_loss": "DPO Loss",
        "heldout_pref_acc": "Pref. Acc.",
        "kl_token_mean": "KL",
        "rm_score_mean": "RM Score"
    })

    # Print the clean console output
    print("="*80)
    print("TASK 1: DPO SUMMARY RESULTS (CONSOLE VIEW)")
    print("="*80)
    print(summary_table.to_string(index=False))
    print("\n" + "="*80)
    
    # Generate the LaTeX code programmatically for the NeurIPS report
    print("TASK 1: NEURIPS LATEX TABLE CODE")
    print("="*80)
    
    latex_str = r"""\begin{table}[h]
  \centering
  \caption{Task 1 DPO Summary Results. Note that the standard 1-epoch DPO run utilizes a full training budget, whereas the three short $\beta$-ablation forks are trained on a restricted budget to isolate the effect of regularization strength.}
  \label{tab:dpo_summary}
  \begin{tabular}{llcccccc}
    \toprule
    \textbf{Condition} & \textbf{Budget} & \textbf{$\beta$} & \textbf{DPO Loss} & \textbf{Pref. Acc.} & \textbf{KL} & \textbf{RM Score} & \textbf{Length (Mean $\pm$ Std)} \\
    \midrule"""
    
    for _, row in summary_table.iterrows():
        # Add a midrule after the standard DPO run to separate the budgets
        if row["Condition"] == "beta=0.03 (short)":
            latex_str += "\n    \\midrule"
            
        latex_str += f"\n    {row['Condition']} & {row['Budget']} & {row['Beta']} & {row['DPO Loss']} & {row['Pref. Acc.']} & {row['KL']} & {row['RM Score']} & {row['Length (Mean ± Std)']} \\\\"
        
    latex_str += r"""
    \bottomrule
  \end{tabular}
\end{table}"""

    print(latex_str)

if __name__ == "__main__":
    extract_dpo_summary()