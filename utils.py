import pandas as pd
import re
from collections import Counter

def load_dataset(filepath):
    """Load the SMS dataset and retain only necessary columns."""
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]  # Keep only label and message columns
    df.columns = ['label', 'message']
    return df


def preprocess_text(text):
    """Tokenize and clean text by removing non-alphabetic characters."""
    return re.findall(r'\b\w+\b', text.lower())

def calculate_statistics(df):
    """Compute the required statistics from the dataset."""
    # Number of SMS messages
    total_messages = len(df)
    
    # Number of spam messages
    num_spam = df[df['label'] == 'spam'].shape[0]
    
    # Tokenize messages
    df['tokens'] = df['message'].apply(preprocess_text)
    
    # Flatten the list of all words
    all_words = [word for tokens in df['tokens'] for word in tokens]
    
    # Total word count
    total_word_count = len(all_words)
    
    # Average words per message
    avg_words_per_message = total_word_count / total_messages
    
    # Frequency distribution of words
    word_counts = Counter(all_words)
    
    # 5 most frequent words
    most_frequent_words = word_counts.most_common(5)
    
    # Number of rare words (appear only once)
    num_rare_words = sum(1 for word, count in word_counts.items() if count == 1)
    
    return {
        "total_messages": total_messages,
        "num_spam": num_spam,
        "total_word_count": total_word_count,
        "avg_words_per_message": avg_words_per_message,
        "most_frequent_words": most_frequent_words,
        "num_rare_words": num_rare_words,
    }


def display_statistics(stats):
    """Display the calculated statistics."""
    print(f"Total SMS messages: {stats['total_messages']}")
    print(f"Number of spam messages: {stats['num_spam']}")
    print(f"Total word count: {stats['total_word_count']}")
    print(f"Average number of words per message: {stats['avg_words_per_message']:.2f}")
    print(f"5 most frequent words: {stats['most_frequent_words']}")
    print(f"Number of rare words: {stats['num_rare_words']}")

