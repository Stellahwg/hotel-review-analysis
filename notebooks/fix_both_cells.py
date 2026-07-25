import json

with open('project_new2.ipynb', 'r') as f:
    nb = json.load(f)

fixed_count = 0

# Fix ALL cells with problem_score issue
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Cell 24 pattern (first Business Impact Score)
        if ("business_impact_score" in source and
            "'problem_score': problem_score" in source and
            "lda_analysis.append" in source):

            print(f"Fixing Cell {i} (first Business Impact Score)...")

            # Find the calculation section and replace
            new_source_lines = []
            inside_calc = False

            for line in cell['source']:
                # Keep lines until we hit the calculation
                if 'freq_score = min((mentions / 500)' in line:
                    # Start replacement
                    new_source_lines.append(line)
                    new_source_lines.append('\n')
                    new_source_lines.append('        # DUAL SCORING: Problems vs Strengths\n')
                    new_source_lines.append('        negative_impact = max(0, -rating_impact) * 100\n')
                    new_source_lines.append('        positive_impact = max(0, rating_impact) * 100\n')
                    new_source_lines.append('\n')
                    new_source_lines.append('        impact_score = max(negative_impact, positive_impact)\n')
                    new_source_lines.append('        reach_score = min((reach_pct / 50) * 100, 100)\n')
                    new_source_lines.append('        sentiment_score = abs(avg_sentiment - overall_avg_sentiment) * 100\n')
                    new_source_lines.append('\n')
                    new_source_lines.append('        # Problem Score: What needs fixing?\n')
                    new_source_lines.append('        problem_score = (\n')
                    new_source_lines.append('            negative_impact * 0.50 +\n')
                    new_source_lines.append('            (reach_pct / 100) * 0.30 +\n')
                    new_source_lines.append('            max(0, -(avg_sentiment - overall_avg_sentiment)) * 0.20\n')
                    new_source_lines.append('        )\n')
                    new_source_lines.append('\n')
                    new_source_lines.append('        # Strength Score: What to protect?\n')
                    new_source_lines.append('        strength_score = (\n')
                    new_source_lines.append('            positive_impact * 0.50 +\n')
                    new_source_lines.append('            (reach_pct / 100) * 0.30 +\n')
                    new_source_lines.append('            max(0, (avg_sentiment - overall_avg_sentiment)) * 0.20\n')
                    new_source_lines.append('        )\n')
                    new_source_lines.append('\n')
                    new_source_lines.append('        business_impact_score = max(problem_score, strength_score)\n')
                    inside_calc = True
                    continue

                # Skip old calculation lines
                if inside_calc:
                    if 'lda_analysis.append' in line:
                        inside_calc = False
                        new_source_lines.append('\n')
                        new_source_lines.append(line)
                    continue

                new_source_lines.append(line)

            cell['source'] = new_source_lines
            fixed_count += 1
            print(f"  ✓ Fixed Cell {i}")

print(f"\n✓ Fixed {fixed_count} cell(s)")

# Write back
with open('project_new2.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Notebook saved!")
