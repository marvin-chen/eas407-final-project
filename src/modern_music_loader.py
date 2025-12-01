"""
Loader for modern Chinese/Korean song lyrics
For extension phase
"""

import pandas as pd
import re

class ModernLyricsLoader:
    """Load and process modern song lyrics"""
    
    @staticmethod
    def clean_lyrics(text):
        """
        Clean song lyrics text
        
        Args:
            text (str): Raw lyrics
            
        Returns:
            str: Cleaned lyrics
        """
        # Remove common patterns: [Verse], [Chorus], etc.
        text = re.sub(r'\[.*?\]', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def split_into_lines(lyrics):
        """
        Split lyrics into lines
        
        Args:
            lyrics (str): Full lyrics text
            
        Returns:
            list: Lines
        """
        lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        return lines
    
    def create_manual_dataset(self):
        """
        Create a small manual dataset of modern songs for comparison
        
        Returns:
            pd.DataFrame: Modern song corpus
        """
        # Example structure - you'll manually add songs
        songs = [
            {
                'title': '青花瓷',  # Jay Chou - Blue and White Porcelain
                'artist': '周杰倫',
                'year': 2007,
                'language': 'Chinese',
                'lyrics': """天青色等煙雨 而我在等你
                            炊煙裊裊升起 隔江千萬里
                            在瓶底書漢隸仿前朝的飄逸
                            就當我為遇見你伏筆"""
            },
            # Add more songs here
        ]
        
        df = pd.DataFrame(songs)
        
        # Split lyrics into lines
        df['lines'] = df['lyrics'].apply(self.split_into_lines)
        
        return df


# Note: For modern music comparison:
# 1. Manually select 3-5 representative Chinese songs (post-2000)
# 2. Manually select 2-3 Korean songs if comparing
# 3. Focus on songs with classical imagery/language
# Recommended Chinese artists: Jay Chou (周杰倫), Faye Wong (王菲), 
#                              Li Jian (李健) - known for poetic lyrics
