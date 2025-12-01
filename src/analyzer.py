"""
Main analysis pipeline for poetry corpus
"""

import pandas as pd
import numpy as np
from collections import Counter
from data_loader import PoetryLoader
from feature_extractor import PoetryFeatureExtractor

class PoetryAnalyzer:
    """Analyze poetry corpus features"""
    
    def __init__(self):
        self.loader = PoetryLoader()
        self.extractor = PoetryFeatureExtractor()
        
    def process_corpus(self, df):
        """
        Extract features for entire corpus
        
        Args:
            df (pd.DataFrame): Poetry corpus with 'lines' column
            
        Returns:
            pd.DataFrame: Original data with added feature columns
        """
        # Extract features for each poem
        features_list = []
        
        for idx, row in df.iterrows():
            features = self.extractor.extract_all_features(row['lines'])
            features_list.append(features)
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Combine with original data
        result = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        return result
    
    def analyze_line_length_distribution(self, df):
        """
        Analyze line length patterns across corpus
        
        Args:
            df (pd.DataFrame): Processed corpus
            
        Returns:
            dict: Distribution statistics
        """
        # Count poems by average line length
        length_counts = df['avg_line_length'].round().value_counts().sort_index()
        
        # Count by poem form
        form_counts = df['poem_form'].value_counts()
        
        return {
            'length_distribution': length_counts.to_dict(),
            'form_distribution': form_counts.to_dict(),
            'mean_length': df['avg_line_length'].mean(),
            'std_length': df['avg_line_length'].std()
        }
    
    def analyze_end_characters(self, df, top_n=20):
        """
        Analyze end-of-line character frequencies
        
        Args:
            df (pd.DataFrame): Processed corpus
            top_n (int): Number of top characters to return
            
        Returns:
            Counter: Most common end characters
        """
        all_end_chars = []
        
        for chars in df['end_characters']:
            all_end_chars.extend(chars)
        
        return Counter(all_end_chars).most_common(top_n)
    
    def compare_dynasties(self, tang_df, song_df):
        """
        Compare features between Tang and Song poetry
        
        Args:
            tang_df (pd.DataFrame): Tang poems
            song_df (pd.DataFrame): Song poems
            
        Returns:
            dict: Comparison statistics
        """
        comparison = {
            'tang': {
                'total_poems': len(tang_df),
                'avg_line_length': tang_df['avg_line_length'].mean(),
                'avg_line_count': tang_df['line_count'].mean(),
                'form_distribution': tang_df['poem_form'].value_counts().to_dict()
            },
            'song': {
                'total_poems': len(song_df),
                'avg_line_length': song_df['avg_line_length'].mean(),
                'avg_line_count': song_df['line_count'].mean(),
                'form_distribution': song_df['poem_form'].value_counts().to_dict()
            }
        }
        
        return comparison
    
    def filter_by_form(self, df, form):
        """
        Filter corpus by poem form
        
        Args:
            df (pd.DataFrame): Processed corpus
            form (str): Poem form to filter (e.g., 'wujue', 'qijue')
            
        Returns:
            pd.DataFrame: Filtered corpus
        """
        return df[df['poem_form'] == form].copy()


# Main execution
if __name__ == '__main__':
    analyzer = PoetryAnalyzer()
    
    print("="*60)
    print("LOADING TANG POETRY CORPUS")
    print("="*60)
    
    # Load Tang poems 
    tang_raw = analyzer.loader.load_tang_poems(max_poems=1000)
    print(f"\nLoaded {len(tang_raw)} Tang poems")
    
    # Process features
    print("\nExtracting features...")
    tang_processed = analyzer.process_corpus(tang_raw)
    
    # Save processed data
    tang_processed.to_csv('data/tang_processed.csv', index=False)
    print("Saved to data/tang_processed.csv")
    
    print("\n" + "="*60)
    print("LINE LENGTH ANALYSIS")
    print("="*60)
    
    line_analysis = analyzer.analyze_line_length_distribution(tang_processed)
    print(f"\nForm distribution:")
    for form, count in line_analysis['form_distribution'].items():
        print(f"  {form}: {count}")
    
    print(f"\nAverage line length: {line_analysis['mean_length']:.2f}")
    print(f"Std deviation: {line_analysis['std_length']:.2f}")
    
    print("\n" + "="*60)
    print("END CHARACTER ANALYSIS")
    print("="*60)
    
    end_chars = analyzer.analyze_end_characters(tang_processed, top_n=15)
    print("\nTop 15 most common end-line characters:")
    for char, count in end_chars:
        print(f"  {char}: {count}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
