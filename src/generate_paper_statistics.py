"""
Generate summary statistics for paper
"""

import pandas as pd
import json

def generate_paper_statistics():
    """Generate all statistics needed for paper"""
    
    # Load processed data
    tang = pd.read_csv('data/tang_processed_mvp.csv')
    
    stats = {
        'corpus_overview': {
            'tang_total': len(tang),
            'tang_date_range': 'Tang Dynasty (618-907 CE)',
            'source': 'chinese-poetry GitHub repository'
        },
        'line_length_stats': {
            'tang_mean': round(tang['avg_line_length'].mean(), 2),
            'tang_std': round(tang['avg_line_length'].std(), 2),
            'tang_min': int(tang['avg_line_length'].min()),
            'tang_max': int(tang['avg_line_length'].max())
        },
        'form_distribution': tang['poem_form'].value_counts().to_dict(),
        'structural_features': {
            'uniform_length_pct': round(tang['is_uniform_length'].sum() / len(tang) * 100, 1),
            'avg_lines_per_poem': round(tang['line_count'].mean(), 2)
        }
    }
    
    # Save for easy reference in paper
    with open('output/paper_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("Paper statistics saved to output/paper_statistics.json")
    return stats

if __name__ == '__main__':
    stats = generate_paper_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
