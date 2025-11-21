"""
Data loading utilities for Chinese poetry corpus
"""

import json
import os
import re
from pathlib import Path
import pandas as pd

class PoetryLoader:
    """Load and preprocess Chinese poetry data"""
    
    def __init__(self, data_dir='data/raw_corpus'):
        """
        Initialize loader with data directory
        
        Args:
            data_dir (str): Path to chinese-poetry repository
        """
        self.data_dir = Path(data_dir)
        self.tang_dir = self.data_dir / '全唐诗'
        self.ci_dir = self.data_dir / '宋词'
        
    def load_tang_poems(self, max_poems=None):
        """
        Load Tang Dynasty poems from JSON files
        
        Args:
            max_poems (int): Maximum number of poems to load (None = all)
            
        Returns:
            pd.DataFrame: DataFrame with columns [title, author, lines, dynasty]
        """
        poems = []
        
        # Get all Tang JSON files (poet.tang.*.json)
        tang_files = sorted(self.tang_dir.glob('poet.tang.*.json'))
        
        for file_path in tang_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for poem in data:
                    poems.append({
                        'title': poem.get('title', ''),
                        'author': poem.get('author', ''),
                        'lines': poem.get('paragraphs', []),
                        'dynasty': 'Tang'
                    })
                    
                    if max_poems and len(poems) >= max_poems:
                        break
            
            if max_poems and len(poems) >= max_poems:
                break
        
        return pd.DataFrame(poems)
    
    def load_song_ci(self, max_poems=None):
        """
        Load Song Dynasty ci 詞 poems
        
        Args:
            max_poems (int): Maximum number to load
            
        Returns:
            pd.DataFrame: DataFrame with ci poems
        """
        poems = []
        
        # Get ci JSON files
        ci_files = sorted(self.ci_dir.glob('ci.song.*.json'))
        
        for file_path in ci_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for poem in data:
                    poems.append({
                        'title': poem.get('rhythmic', ''),  # ci uses 'rhythmic' for tune name
                        'author': poem.get('author', ''),
                        'lines': poem.get('paragraphs', []),
                        'dynasty': 'Song'
                    })
                    
                    if max_poems and len(poems) >= max_poems:
                        break
            
            if max_poems and len(poems) >= max_poems:
                break
        
        return pd.DataFrame(poems)

    @staticmethod
    def clean_line(line):
        """
        Remove punctuation from a poetry line
        
        Args:
            line (str): Poetry line with punctuation
            
        Returns:
            str: Cleaned line with only Chinese characters
        """
        # Remove common Chinese punctuation
        punctuation = r'[，。！？；：、""''『』「」〈〉《》【】〔〕…—～]'
        cleaned = re.sub(punctuation, '', line)
        return cleaned.strip()
    
    @staticmethod
    def extract_chinese_chars(text):
        """
        Extract only Chinese characters from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Only Chinese characters
        """
        # Unicode range for Chinese characters
        chinese_pattern = r'[\u4e00-\u9fff]+'
        chars = re.findall(chinese_pattern, text)
        return ''.join(chars)


# Test the loader
if __name__ == '__main__':
    loader = PoetryLoader()
    
    # Load sample Tang poems
    print("Loading Tang poems...")
    tang_df = loader.load_tang_poems(max_poems=100)
    print(f"Loaded {len(tang_df)} Tang poems")
    if len(tang_df) > 0:
        print("\nSample poem:")
        print(tang_df.iloc[0])
    else:
        print("No poems loaded. Check data directory.")
    
    # Load sample Song ci
    print("\n" + "="*50)
    print("Loading Song ci...")
    ci_df = loader.load_song_ci(max_poems=50)
    print(f"Loaded {len(ci_df)} Song ci")
    if len(ci_df) > 0:
        print("\nSample ci:")
        print(ci_df.iloc[0])
    else:
        print("No ci loaded. Check data directory.")
