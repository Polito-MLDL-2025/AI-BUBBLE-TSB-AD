#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute average scores for each hyperparameter configuration from HP tuning results.
"""

import pandas as pd
import ast
import argparse
from pathlib import Path


def parse_hp_string(hp_str):
    """Parse HP string dict to actual dict."""
    try:
        return ast.literal_eval(hp_str)
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description='Compute average scores per HP config')
    parser.add_argument('--input', type=str, 
                        default='Datasets/TSB-AD-Datasets/hp-tuning-u/Chronos2Ada_252.csv',
                        help='Path to HP tuning CSV file')
    parser.add_argument('--metric', type=str, default='VUS-PR',
                        help='Metric to use for ranking (default: VUS-PR)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Show top K configurations')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Columns: {list(df.columns)}")
    
    # Get score columns (exclude file and HP)
    score_cols = [c for c in df.columns if c not in ['file', 'HP']]
    print(f"\nScore columns: {score_cols}")
    
    # Group by HP and compute mean scores
    hp_stats = df.groupby('HP')[score_cols].agg(['mean', 'std', 'count'])
    
    # Flatten column names
    hp_stats.columns = ['_'.join(col).strip() for col in hp_stats.columns.values]
    hp_stats = hp_stats.reset_index()
    
    # Sort by specified metric (mean)
    sort_col = f'{args.metric}_mean'
    if sort_col not in hp_stats.columns:
        print(f"Warning: {sort_col} not found. Available: {list(hp_stats.columns)}")
        sort_col = hp_stats.columns[1]  # Use first score column
    
    hp_stats_sorted = hp_stats.sort_values(sort_col, ascending=False)
    
    # Print top K configurations
    print(f"\n{'='*80}")
    print(f"Top {args.top_k} HP configurations by {args.metric}:")
    print(f"{'='*80}")
    
    for i, row in hp_stats_sorted.head(args.top_k).iterrows():
        hp_dict = parse_hp_string(row['HP'])
        print(f"\nRank {hp_stats_sorted.index.get_loc(i) + 1}:")
        print(f"  HP: {hp_dict}")
        print(f"  {args.metric}: {row[sort_col]:.4f} (±{row[f'{args.metric}_std']:.4f})")
        print(f"  Count: {int(row[f'{args.metric}_count'])}")
        
        # Print other metrics
        print("  Other metrics (mean):")
        for col in score_cols:
            if col != args.metric:
                print(f"    {col}: {row[f'{col}_mean']:.4f}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("Summary across all HP configurations:")
    print(f"{'='*80}")
    print(f"Total unique HP configs: {len(hp_stats)}")
    print(f"Total evaluations: {len(df)}")
    print(f"Files per config: {df.groupby('HP').size().mean():.1f}")
    
    # Best HP config
    best_hp = hp_stats_sorted.iloc[0]['HP']
    best_hp_dict = parse_hp_string(best_hp)
    print(f"\nBest HP config for {args.metric}:")
    print(f"  {best_hp_dict}")
    
    # Save results
    output_path = Path(args.input).parent / f"{Path(args.input).stem}_hp_avg.csv"
    hp_stats_sorted.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
