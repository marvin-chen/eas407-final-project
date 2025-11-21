"""
Visualization utilities for poetry analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PoetryVisualizer:
    """Create visualizations for poetry analysis"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        
        # Set matplotlib to support Chinese characters
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_form_distribution(self, df, title='Poem Form Distribution', save_name=None):
        """
        Bar chart of poem form frequencies
        
        Args:
            df (pd.DataFrame): Processed corpus
            title (str): Chart title
            save_name (str): Filename to save (optional)
        """
        form_counts = df['poem_form'].value_counts()
        
        plt.figure(figsize=(10, 6))
        form_counts.plot(kind='bar', color='steelblue', edgecolor='black')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Poem Form', fontsize=12)
        plt.ylabel('Number of Poems', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.output_dir}/{save_name}', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_line_length_distribution(self, df, title='Line Length Distribution', save_name=None):
        """
        Histogram of average line lengths
        
        Args:
            df (pd.DataFrame): Processed corpus
            title (str): Chart title
            save_name (str): Filename to save
        """
        plt.figure(figsize=(10, 6))
        plt.hist(df['avg_line_length'], bins=20, color='coral', edgecolor='black', alpha=0.7)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Average Line Length (characters)', fontsize=12)
        plt.ylabel('Number of Poems', fontsize=12)
        plt.axvline(df['avg_line_length'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["avg_line_length"].mean():.2f}')
        plt.legend()
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.output_dir}/{save_name}', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_dynasty_comparison(self, tang_df, song_df, save_name=None):
        """
        Compare line length distributions between dynasties
        
        Args:
            tang_df (pd.DataFrame): Tang poems
            song_df (pd.DataFrame): Song poems
            save_name (str): Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Tang distribution
        axes[0].hist(tang_df['avg_line_length'], bins=15, color='steelblue', 
                    edgecolor='black', alpha=0.7)
        axes[0].set_title('Tang Dynasty', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Average Line Length', fontsize=11)
        axes[0].set_ylabel('Number of Poems', fontsize=11)
        axes[0].axvline(tang_df['avg_line_length'].mean(), color='red', 
                       linestyle='--', linewidth=2)
        
        # Song distribution
        axes[1].hist(song_df['avg_line_length'], bins=15, color='coral', 
                    edgecolor='black', alpha=0.7)
        axes[1].set_title('Song Dynasty', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Average Line Length', fontsize=11)
        axes[1].set_ylabel('Number of Poems', fontsize=11)
        axes[1].axvline(song_df['avg_line_length'].mean(), color='red', 
                       linestyle='--', linewidth=2)
        
        plt.suptitle('Line Length Distribution Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.output_dir}/{save_name}', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_end_character_frequency(self, end_char_counter, top_n=20, save_name=None):
        """
        Bar chart of most frequent end characters
        
        Args:
            end_char_counter (Counter): Counter of end characters
            top_n (int): Number of top characters to show
            save_name (str): Filename to save
        """
        top_chars = end_char_counter.most_common(top_n)
        chars, counts = zip(*top_chars)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(chars)), counts, color='seagreen', edgecolor='black')
        plt.title(f'Top {top_n} Most Common End-Line Characters', fontsize=16, fontweight='bold')
        plt.xlabel('Character', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(range(len(chars)), chars, fontsize=14)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.output_dir}/{save_name}', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_form_comparison_grouped(self, tang_df, song_df, save_name=None):
        """
        Grouped bar chart comparing form distributions
        
        Args:
            tang_df (pd.DataFrame): Tang corpus
            song_df (pd.DataFrame): Song corpus
            save_name (str): Filename to save
        """
        tang_forms = tang_df['poem_form'].value_counts()
        song_forms = song_df['poem_form'].value_counts()
        
        # Get all unique forms
        all_forms = set(tang_forms.index) | set(song_forms.index)
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Tang': [tang_forms.get(form, 0) for form in all_forms],
            'Song': [song_forms.get(form, 0) for form in all_forms]
        }, index=list(all_forms))
        
        comparison.plot(kind='bar', figsize=(12, 6), width=0.8, edgecolor='black')
        plt.title('Poem Form Distribution by Dynasty', fontsize=16, fontweight='bold')
        plt.xlabel('Poem Form', fontsize=12)
        plt.ylabel('Number of Poems', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Dynasty', fontsize=11)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.output_dir}/{save_name}', dpi=300, bbox_inches='tight')
        
        plt.show()


# Test visualizations
if __name__ == '__main__':
    # Load processed data
    print("Loading data...")
    tang_df = pd.read_csv('data/tang_processed.csv')
    
    print(f"Loaded {len(tang_df)} poems")
    
    # Create visualizer
    viz = PoetryVisualizer()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    viz.plot_form_distribution(tang_df, save_name='tang_form_distribution.png')
    viz.plot_line_length_distribution(tang_df, save_name='tang_line_length_dist.png')
    
    # End character analysis
    all_end_chars = []
    for chars in tang_df['end_characters'].apply(eval):  # Convert string back to list
        all_end_chars.extend(chars)
    end_char_counter = Counter(all_end_chars)
    
    viz.plot_end_character_frequency(end_char_counter, top_n=20, 
                                    save_name='tang_end_chars.png')
    
    print("\nVisualizations saved to output/")
