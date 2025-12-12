"""
Main analysis pipeline for poetry corpus
"""

import pandas as pd
import numpy as np
from collections import Counter
from data_loader import PoetryLoader
from feature_extractor import PoetryFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    def ml_classification(self, tang_df, song_df):
        """
        Train classifier to distinguish Tang vs Song
        
        Args:
            tang_df (pd.DataFrame): Tang poems with extracted features
            song_df (pd.DataFrame): Song poems with extracted features
            
        Returns:
            tuple: (classifier, accuracy, feature_importance_df)
        """
        # Check if uniform_length column exists (boolean)
        # Convert to percentage: True -> 100, False -> 0
        tang_uniformity = tang_df.get('uniform_length', 
                                      (tang_df['line_length_std'] == 0)).astype(int) * 100
        song_uniformity = song_df.get('uniform_length',
                                      (song_df['line_length_std'] == 0)).astype(int) * 100
        
        # Create feature matrix
        tang_features = pd.DataFrame({
            'avg_lines': tang_df['line_count'],
            'avg_line_length': tang_df['avg_line_length'],
            'uniformity': tang_uniformity,
            'irregular': (tang_df['poem_form'] == 'irregular').astype(int),
            'dynasty': 0  # Tang = 0
        })
        
        song_features = pd.DataFrame({
            'avg_lines': song_df['line_count'],
            'avg_line_length': song_df['avg_line_length'],
            'uniformity': song_uniformity,
            'irregular': (song_df['poem_form'] == 'irregular').astype(int),
            'dynasty': 1  # Song = 1
        })
        
        df = pd.concat([tang_features, song_features])
        
        X = df[['avg_lines', 'avg_line_length', 'uniformity', 'irregular']]
        y = df['dynasty']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        
        print(f"\nAccuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Tang', 'Song']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Tang', 'Song'],
                    yticklabels=['Tang', 'Song'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix: Tang vs Song Classification')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved confusion matrix to confusion_matrix.png")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(8, 5))
        plt.barh(importance['feature'], importance['importance'], color='steelblue')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance for Tang vs Song Classification')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved feature importance to feature_importance.png")
        
        print("\nFeature Importance:")
        for _, row in importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return clf, accuracy, importance


# Main execution
if __name__ == '__main__':
    analyzer = PoetryAnalyzer()
    
    print("="*60)
    print("LOADING TANG POETRY CORPUS")
    print("="*60)
    
    # Load Tang poems 
    tang_raw = analyzer.loader.load_tang_poems(max_poems=None)  # Load all
    print(f"\nLoaded {len(tang_raw)} Tang poems")
    
    # Process features
    print("\nExtracting features...")
    tang_processed = analyzer.process_corpus(tang_raw)
    
    # Save processed data
    tang_processed.to_csv('data/tang_processed.csv', index=False)
    print("Saved to data/tang_processed.csv")
    
    print("\n" + "="*60)
    print("LOADING SONG POETRY CORPUS")
    print("="*60)
    
    # Load Song poems
    song_raw = analyzer.loader.load_song_ci(max_poems=None)  # Load all
    print(f"\nLoaded {len(song_raw)} Song poems")
    
    # Process features
    print("\nExtracting features...")
    song_processed = analyzer.process_corpus(song_raw)
    
    # Save processed data
    song_processed.to_csv('data/song_processed.csv', index=False)
    print("Saved to data/song_processed.csv")
    
    print("\n" + "="*60)
    print("LINE LENGTH ANALYSIS - TANG")
    print("="*60)
    
    tang_analysis = analyzer.analyze_line_length_distribution(tang_processed)
    print(f"\nForm distribution:")
    for form, count in tang_analysis['form_distribution'].items():
        print(f"  {form}: {count}")
    
    print(f"\nAverage line length: {tang_analysis['mean_length']:.2f}")
    print(f"Std deviation: {tang_analysis['std_length']:.2f}")
    
    print("\n" + "="*60)
    print("LINE LENGTH ANALYSIS - SONG")
    print("="*60)
    
    song_analysis = analyzer.analyze_line_length_distribution(song_processed)
    print(f"\nForm distribution:")
    for form, count in song_analysis['form_distribution'].items():
        print(f"  {form}: {count}")
    
    print(f"\nAverage line length: {song_analysis['mean_length']:.2f}")
    print(f"Std deviation: {song_analysis['std_length']:.2f}")
    
    print("\n" + "="*60)
    print("MACHINE LEARNING CLASSIFICATION")
    print("="*60)
    
    # Train classifier
    clf, accuracy, importance = analyzer.ml_classification(tang_processed, song_processed)
    
    print("\n" + "="*60)
    print("END CHARACTER ANALYSIS - TANG")
    print("="*60)
    
    tang_end_chars = analyzer.analyze_end_characters(tang_processed, top_n=20)
    print("\nTop 20 most common end-line characters:")
    for char, count in tang_end_chars:
        print(f"  {char}: {count}")
    
    print("\n" + "="*60)
    print("END CHARACTER ANALYSIS - SONG")
    print("="*60)
    
    song_end_chars = analyzer.analyze_end_characters(song_processed, top_n=20)
    print("\nTop 20 most common end-line characters:")
    for char, count in song_end_chars:
        print(f"  {char}: {count}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nTang poems: {len(tang_processed)}")
    print(f"Song poems: {len(song_processed)}")
    print(f"Classification accuracy: {accuracy:.1%}")
