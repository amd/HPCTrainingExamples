#!/usr/bin/env python3

import pandas

df = pandas.read_csv('results.stats.csv')

df.rename(columns={'TotalDurationNs': 'TotalDurationMs'}, inplace=True)
df.rename(columns={'AverageNs': 'AverageMs'}, inplace=True)


df['TotalDurationMs'] = df['TotalDurationMs'].div(1000000)
df['AverageMs']       = df['AverageMs'].div(1000000)

print(df)
