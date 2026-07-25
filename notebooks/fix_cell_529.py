import json

# Read notebook
with open('project_new2.ipynb', 'r') as f:
    nb = json.load(f)

# New cell source
new_source = """# Calculate Business Impact Score for each topic
# Higher score = Higher priority to address

for idx, row in analysis_df.iterrows():
    # Normalize each component to 0-100 scale

    # 1. Frequency score (more mentions = higher priority)
    freq_score = min((row['mentions'] / 500) * 100, 100)

    # 2. Sentiment severity (more negative ASPECT sentiment = higher priority)
    # Using aspect sentiment now!
    sentiment_score = abs(row['sentiment_severity']) * 100

    # 3. Rating impact (bigger negative impact = higher priority)
    # Invert so negative impact = high score
    # DUAL SCORING: Problems vs Strengths
    negative_impact = max(0, -row['rating_impact']) * 100
    positive_impact = max(0, row['rating_impact']) * 100

    # 4. Reach score (higher % = higher priority)
    reach_score = min((row['reach_pct'] / 50) * 100, 100)

    # Calculate separate scores for problems vs strengths
    # Handle NaN sentiment_severity values
    import pandas as pd
    sentiment_val = row['sentiment_severity'] if not pd.isna(row['sentiment_severity']) else 0

    # Problem Score: What needs fixing?
    problem_score = (
        negative_impact * 0.50 +                    # 50% - Negative rating impact
        (row['reach_pct'] / 100) * 0.30 +          # 30% - Reach
        max(0, -sentiment_val) * 0.20              # 20% - Negative sentiment severity
    )

    # Strength Score: What to protect?
    strength_score = (
        positive_impact * 0.50 +                    # 50% - Positive rating impact
        (row['reach_pct'] / 100) * 0.30 +          # 30% - Reach
        max(0, sentiment_val) * 0.20               # 20% - Positive sentiment severity
    )

    # Legacy scores for compatibility
    impact_score = max(negative_impact, positive_impact)
    business_impact_score = max(problem_score, strength_score)

    # Store in dataframe
    analysis_df.loc[idx, 'freq_score'] = freq_score
    analysis_df.loc[idx, 'sentiment_score'] = sentiment_score
    analysis_df.loc[idx, 'impact_score'] = impact_score
    analysis_df.loc[idx, 'reach_score'] = reach_score
    analysis_df.loc[idx, 'problem_score'] = problem_score
    analysis_df.loc[idx, 'strength_score'] = strength_score
    analysis_df.loc[idx, 'business_impact_score'] = business_impact_score

# Sort by Business Impact Score (highest first)
analysis_df = analysis_df.sort_values('business_impact_score', ascending=False)

print("Business Impact Score calculated for all topics (using aspect sentiment)!")"""

# Find and replace cell 39
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '# Calculate Business Impact Score for each topic' in source and 'negative_impact = max(0, -row' in source:
            # Replace with new source (split into lines for notebook format)
            cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]
            print(f"✓ Replaced cell {i}")
            break

# Write back
with open('project_new2.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Notebook updated!")
