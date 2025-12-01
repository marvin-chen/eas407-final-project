"""
Extract formal features from Chinese poetry
"""

import re
import numpy as np
from collections import Counter

class PoetryFeatureExtractor:
    """Extract structural features from poems"""
    
    @staticmethod
    def clean_line(line):
        """Remove punctuation, keep only Chinese characters"""
        punctuation = r'[，。！？；：、""''『』「」〈〉《》【】〔〕…—～\s]'
        cleaned = re.sub(punctuation, '', line)
        return cleaned.strip()
    
    @staticmethod
    def get_line_length(line):
        """
        Get character count of a cleaned line
        
        Args:
            line (str): Poetry line
            
        Returns:
            int: Number of Chinese characters
        """
        cleaned = PoetryFeatureExtractor.clean_line(line)
        # Count only Chinese characters
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', cleaned)
        return len(chinese_chars)
    
    @staticmethod
    def get_line_lengths(lines):
        """
        Get line lengths for all lines in poem
        
        Args:
            lines (list): List of poetry lines
            
        Returns:
            list: Line lengths
        """
        return [PoetryFeatureExtractor.get_line_length(line) for line in lines]
    
    @staticmethod
    def get_end_character(line):
        """
        Extract final character from line (rhyme position)
        
        Args:
            line (str): Poetry line
            
        Returns:
            str: Last Chinese character, or empty string if none
        """
        cleaned = PoetryFeatureExtractor.clean_line(line)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', cleaned)
        return chinese_chars[-1] if chinese_chars else ''
    
    @staticmethod
    def get_end_characters(lines):
        """
        Get end characters for all lines
        
        Args:
            lines (list): List of poetry lines
            
        Returns:
            list: End characters (empty strings filtered out)
        """
        end_chars = [PoetryFeatureExtractor.get_end_character(line) for line in lines]
        # Filter out empty strings (lines with no Chinese characters)
        return [char for char in end_chars if char]
    
    @staticmethod
    def classify_poem_form(lines):
        """
        Classify poem by structural form (shi 詩 only, not ci or qu)
        
        Note: Many classical Chinese poems format each line as two hemistichs,
        e.g., "五字，五字。" appears as 10 chars but represents a 5-char line.
        
        Args:
            lines (list): List of poetry lines
            
        Returns:
            str: Poem form classification
        """
        if not lines:
            return 'unknown'
        
        line_lengths = PoetryFeatureExtractor.get_line_lengths(lines)
        line_count = len(lines)
        
        # Check if all lines have same length
        if len(set(line_lengths)) != 1:
            return 'irregular'
        
        length = line_lengths[0]
        
        # Classify common Tang shi forms
        # Handle both single-line and double-hemistich formats
        if line_count == 4 and length == 5:
            return 'wujue'  # 五言絕句 (5-char quatrain)
        elif line_count == 4 and length == 7:
            return 'qijue'  # 七言絕句 (7-char quatrain)
        elif line_count == 4 and length == 10:  # Two 5-char hemistichs per line
            return 'wujue'  # 五言絕句
        elif line_count == 4 and length == 14:  # Two 7-char hemistichs per line
            return 'qijue'  # 七言絕句
        elif line_count == 8 and length == 5:
            return 'wulu'   # 五言律詩 (5-char regulated verse)
        elif line_count == 8 and length == 7:
            return 'qilu'   # 七言律詩 (7-char regulated verse)
        elif line_count == 8 and length == 10:  # Two 5-char hemistichs per line
            return 'wulu'   # 五言律詩
        elif line_count == 8 and length == 14:  # Two 7-char hemistichs per line
            return 'qilu'   # 七言律詩
        # Handle longer regulated verse forms
        elif line_count > 8 and length in [5, 10]:
            return 'wupai'  # 五言排律 (long regulated verse, 5-char)
        elif line_count > 8 and length in [7, 14]:
            return 'qipai'  # 七言排律 (long regulated verse, 7-char)
        else:
            return 'other'
    
    @staticmethod
    def calculate_line_length_stats(lines):
        """
        Calculate statistical measures of line length variation
        
        Args:
            lines (list): List of poetry lines
            
        Returns:
            dict: Statistics (mean, std, min, max, variance)
        """
        lengths = PoetryFeatureExtractor.get_line_lengths(lines)
        
        if not lengths:
            return {
                'mean': 0, 'std': 0, 'min': 0, 'max': 0, 
                'variance': 0, 'is_uniform': False
            }
        
        return {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'variance': np.var(lengths),
            'is_uniform': len(set(lengths)) == 1
        }
    
    @staticmethod
    def extract_all_features(poem_lines):
        """
        Extract all features for a single poem
        
        Args:
            poem_lines (list): List of poetry lines
            
        Returns:
            dict: All extracted features
        """
        line_lengths = PoetryFeatureExtractor.get_line_lengths(poem_lines)
        end_chars = PoetryFeatureExtractor.get_end_characters(poem_lines)
        stats = PoetryFeatureExtractor.calculate_line_length_stats(poem_lines)
        
        return {
            'line_count': len(poem_lines),
            'line_lengths': line_lengths,
            'end_characters': end_chars,
            'poem_form': PoetryFeatureExtractor.classify_poem_form(poem_lines),
            'avg_line_length': stats['mean'],
            'line_length_std': stats['std'],
            'line_length_variance': stats['variance'],
            'is_uniform_length': stats['is_uniform'],
            'total_characters': sum(line_lengths)
        }


# Test the extractor
if __name__ == '__main__':
    # Test with a sample Tang poem (Wang Zhihuan - 登鸛雀樓)
    sample_poem = [
        "白日依山盡，",
        "黃河入海流。",
        "欲窮千里目，",
        "更上一層樓。"
    ]
    
    extractor = PoetryFeatureExtractor()
    
    print("Sample poem:")
    for line in sample_poem:
        print(f"  {line}")
    
    print("\nExtracted features:")
    features = extractor.extract_all_features(sample_poem)
    for key, value in features.items():
        print(f"  {key}: {value}")
