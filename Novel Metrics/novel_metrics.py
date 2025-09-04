# -*- coding: utf-8 -*-
!pip install -q textblob


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob

"""STEP 2: Load uploaded CSV"""

df = pd.read_csv("/content/cobal.csv")

df['Start_sec'] = pd.to_timedelta(df['Start']).dt.total_seconds()
df['End_sec']   = pd.to_timedelta(df['End']).dt.total_seconds()
df['Duration']  = df['End_sec'] - df['Start_sec']

"""STEP 3: Feature extraction"""

df['WordCount'] = df['Text'].apply(lambda x: len(str(x).split()))
fillers = {"um","uh","like","you know"}
df['FillerCount'] = df['Text'].apply(lambda t: sum(1 for w in str(t).lower().split() if w in fillers))
df['Sentiment'] = df['Text'].apply(lambda t: TextBlob(str(t)).sentiment.polarity)

"""STEP 4: Speaker-level metrics"""

speaker_stats = df.groupby('Speaker').agg(
    TalkTime_sec = ('Duration', 'sum'),
    TotalWords   = ('WordCount', 'sum'),
    FillerWords  = ('FillerCount', 'sum'),
    AvgSentiment = ('Sentiment', 'mean')
).reset_index()


total_time = speaker_stats['TalkTime_sec'].sum()

if total_time > 0:
    speaker_stats['TalkTime(%)'] = (speaker_stats['TalkTime_sec'] / total_time * 100).round(2)
else:
    speaker_stats['TalkTime(%)'] = 0.0

def safe_wpm(row):
    if row['TalkTime_sec'] > 0:
        minutes = row['TalkTime_sec'] / 60.0
        return round(row['TotalWords'] / minutes, 2)
    else:
        return 0.0

speaker_stats['WPM'] = speaker_stats.apply(safe_wpm, axis=1)

speaker_stats['FillerRate'] = speaker_stats.apply(
    lambda r: round(r['FillerWords'] / r['TotalWords'], 3) if r['TotalWords'] > 0 else 0.0,
    axis=1
)

speaker_stats['DominanceRatio'] = speaker_stats.apply(
    lambda r: round(r['TalkTime_sec'] / total_time, 3) if total_time>0 else 0.0,
    axis=1
)

display_cols = ['Speaker','TalkTime(%)','WPM','AvgSentiment','FillerRate','DominanceRatio']
print("=== Novel Speaker Metrics ===")
display(speaker_stats[display_cols])

"""Sentiment Divergence (variance)"""

var_val = speaker_stats['AvgSentiment'].var(ddof=0)
if pd.isna(var_val):
    sentiment_divergence = 0.0
else:
    sentiment_divergence = round(float(var_val), 3)

print(f"\nSentiment Divergence (variance across speakers): {sentiment_divergence}")

"""STEP 5 : Visualizations"""

plt.figure(figsize=(6,6))
plt.pie(speaker_stats['TalkTime_sec'], labels=speaker_stats['Speaker'], autopct='%1.1f%%')
plt.title("Talk Time Distribution")
plt.show()

plt.figure(figsize=(6,4))
plt.bar(speaker_stats['Speaker'], speaker_stats['AvgSentiment'])
plt.title("Average Sentiment per Speaker")
plt.ylabel("Polarity (-1=Negative, +1=Positive)")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.show()

plt.figure(figsize=(6,4))
plt.bar(speaker_stats['Speaker'], speaker_stats['WPM'])
plt.title("Words Per Minute (WPM) per Speaker")
plt.ylabel("WPM")
plt.show()

# Sentiment trajectory over time (per utterance)
plt.figure(figsize=(10,5))
for speaker in df['Speaker'].unique():
    subset = df[df['Speaker'] == speaker].sort_values('Start_sec')
    plt.plot(subset['Start_sec'], subset['Sentiment'], marker='o', label=speaker)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title("Sentiment Trajectory Over Time (per Speaker)")
plt.xlabel("Time (seconds)")
plt.ylabel("Sentiment Polarity")
plt.legend()
plt.show()