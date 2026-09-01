#!/usr/bin/env python3
"""
analyze_per_node_latency.py
Uses min, mean, max — shows bimodal read-retry signature clearly.

Usage:
    python3 analyze_per_node_latency.py per_node_latency.tsv
"""
import sys
import numpy as np
import pandas as pd

tsv = sys.argv[1] if len(sys.argv) > 1 else "per_node_latency.tsv"

print(f"Loading {tsv}...")
df = pd.read_csv(tsv, sep='\t')
df['read_us'] = pd.to_numeric(df['read_us'], errors='coerce')
df = df.dropna()

print(f"  Total read records : {len(df):,}")
print(f"  Unique nodes       : {df['node_id'].nunique():,}")

agg = df.groupby('node_id')['read_us'].agg(
    visits='count', min='min', mean='mean', max='max',
).reset_index()

global_mean = df['read_us'].mean()
global_max  = df['read_us'].max()

print(f"\nGlobal mean read latency : {global_mean:.1f} µs")
print(f"Global max  read latency : {global_max:.1f} µs")

# Hub nodes: high visits, max never goes slow
common_nodes = (
    agg[(agg['visits'] >= 100) & (agg['max'] < 2 * global_mean)]
    .nlargest(5, 'visits').reset_index(drop=True)
)

# Tail nodes: at least 5 visits, max is catastrophically slow
tail_nodes = (
    agg[(agg['visits'] >= 5) & (agg['max'] > 5 * global_mean)]
    .nlargest(5, 'max').reset_index(drop=True)
)

SEP = "─" * 72

print(f"\n{SEP}")
print("  HUB NODES  (high visits, fast reads — BFS caches these)")
print(SEP)
print(f"  {'Node ID':>10}  {'Visits':>7}  {'Min(µs)':>8}  {'Mean(µs)':>9}  {'Max(µs)':>9}")
print(f"  {'─'*68}")
for _, r in common_nodes.iterrows():
    print(f"  {int(r.node_id):>10}  {int(r.visits):>7,}  "
          f"{r['min']:>8.1f}  {r['mean']:>9.1f}  {r['max']:>9.1f}")

print(f"\n{SEP}")
print("  TAIL-CRITICAL NODES  (slow max read — left cold by BFS)")
print(SEP)
print(f"  {'Node ID':>10}  {'Visits':>7}  {'Min(µs)':>8}  {'Mean(µs)':>9}  {'Max(µs)':>9}  ratio")
print(f"  {'─'*68}")
for _, r in tail_nodes.iterrows():
    ratio = r['max'] / global_mean
    print(f"  {int(r.node_id):>10}  {int(r.visits):>7,}  "
          f"{r['min']:>8.1f}  {r['mean']:>9.1f}  {r['max']:>9.1f}  ({ratio:.1f}x global mean)")

print(f"\n{SEP}")
print("  KEY NUMBERS FOR PAPER")
print(SEP)
if len(common_nodes) and len(tail_nodes):
    avg_hub_max  = common_nodes['max'].mean()
    max_tail_max = tail_nodes['max'].max()
    print(f"  Hub node avg max read    : {avg_hub_max:.1f} µs")
    print(f"  Tail node worst max read : {max_tail_max:.1f} µs")
    print(f"  Ratio                    : {max_tail_max / avg_hub_max:.1f}x slower")
    print(f"  Global mean              : {global_mean:.1f} µs")
print(SEP)
