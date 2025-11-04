"""
Hangman ML Hackathon - Part 1: Data Exploration & Preprocessing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import string

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the corpus
def load_corpus(file_path='corpus.txt'):
    """Load and return the corpus as a list of words"""
    with open(file_path, 'r') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return words

# Load data
corpus = load_corpus()

print("=" * 60)
print("CORPUS STATISTICS")
print("=" * 60)
print(f"Total words in corpus: {len(corpus)}")
print(f"Sample words: {corpus[:10]}")

# Word length distribution
word_lengths = [len(word) for word in corpus]
length_counter = Counter(word_lengths)

print(f"\nWord length range: {min(word_lengths)} to {max(word_lengths)}")
print(f"Average word length: {np.mean(word_lengths):.2f}")
print(f"Median word length: {np.median(word_lengths):.2f}")

# Plot word length distribution
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(word_lengths, bins=range(min(word_lengths), max(word_lengths) + 2), 
         edgecolor='black', alpha=0.7)
plt.xlabel('Word Length')
plt.ylabel('Frequency')
plt.title('Distribution of Word Lengths')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sorted_lengths = sorted(length_counter.items())
lengths, counts = zip(*sorted_lengths)
plt.bar(lengths, counts, edgecolor='black', alpha=0.7)
plt.xlabel('Word Length')
plt.ylabel('Count')
plt.title('Word Count by Length')
plt.xticks(range(min(word_lengths), max(word_lengths) + 1, 2))
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('word_length_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Letter frequency analysis
print("\n" + "=" * 60)
print("LETTER FREQUENCY ANALYSIS")
print("=" * 60)

all_letters = ''.join(corpus)
letter_counts = Counter(all_letters)

# Sort by frequency
sorted_letters = sorted(letter_counts.items(), key=lambda x: x[1], reverse=True)
letters, frequencies = zip(*sorted_letters)

print("\nTop 10 most common letters:")
for i, (letter, count) in enumerate(sorted_letters[:10], 1):
    percentage = (count / sum(letter_counts.values())) * 100
    print(f"{i}. '{letter}': {count:,} ({percentage:.2f}%)")

# Plot letter frequencies
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.bar(range(len(letters)), frequencies, edgecolor='black', alpha=0.7)
plt.xlabel('Letters')
plt.ylabel('Frequency')
plt.title('Letter Frequency Distribution (All Letters)')
plt.xticks(range(len(letters)), letters, rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
top_n = 15
plt.barh(range(top_n), [frequencies[i] for i in range(top_n)], 
         edgecolor='black', alpha=0.7)
plt.yticks(range(top_n), [letters[i] for i in range(top_n)])
plt.xlabel('Frequency')
plt.title(f'Top {top_n} Most Common Letters')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('letter_frequency.png', dpi=300, bbox_inches='tight')
plt.show()

# Positional letter frequency
print("\n" + "=" * 60)
print("POSITIONAL ANALYSIS")
print("=" * 60)

def analyze_position_frequency(words, max_length=15):
    """Analyze letter frequency at each position"""
    position_freq = {}
    
    for word in words:
        if len(word) <= max_length:
            for pos, letter in enumerate(word):
                if pos not in position_freq:
                    position_freq[pos] = Counter()
                position_freq[pos][letter] += 1
    
    return position_freq

position_freq = analyze_position_frequency(corpus)

# Starting and ending letters
starting_letters = Counter([word[0] for word in corpus])
ending_letters = Counter([word[-1] for word in corpus])

print("\nTop 10 starting letters:")
for i, (letter, count) in enumerate(starting_letters.most_common(10), 1):
    percentage = (count / len(corpus)) * 100
    print(f"{i}. '{letter}': {count:,} ({percentage:.2f}%)")

print("\nTop 10 ending letters:")
for i, (letter, count) in enumerate(ending_letters.most_common(10), 1):
    percentage = (count / len(corpus)) * 100
    print(f"{i}. '{letter}': {count:,} ({percentage:.2f}%)")

# Plot starting/ending letters
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Starting letters
start_sorted = sorted(starting_letters.items(), key=lambda x: x[1], reverse=True)[:15]
letters_start, counts_start = zip(*start_sorted)
axes[0].barh(range(len(letters_start)), counts_start, edgecolor='black', alpha=0.7)
axes[0].set_yticks(range(len(letters_start)))
axes[0].set_yticklabels(letters_start)
axes[0].set_xlabel('Frequency')
axes[0].set_title('Top 15 Starting Letters')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3)

# Ending letters
end_sorted = sorted(ending_letters.items(), key=lambda x: x[1], reverse=True)[:15]
letters_end, counts_end = zip(*end_sorted)
axes[1].barh(range(len(letters_end)), counts_end, edgecolor='black', alpha=0.7)
axes[1].set_yticks(range(len(letters_end)))
axes[1].set_yticklabels(letters_end)
axes[1].set_xlabel('Frequency')
axes[1].set_title('Top 15 Ending Letters')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('positional_frequency.png', dpi=300, bbox_inches='tight')
plt.show()

# Common bigrams and trigrams
print("\n" + "=" * 60)
print("N-GRAM ANALYSIS")
print("=" * 60)

def get_ngrams(words, n):
    """Extract n-grams from words"""
    ngrams = []
    for word in words:
        if len(word) >= n:
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i+n])
    return Counter(ngrams)

bigrams = get_ngrams(corpus, 2)
trigrams = get_ngrams(corpus, 3)

print("\nTop 15 most common bigrams:")
for i, (bigram, count) in enumerate(bigrams.most_common(15), 1):
    print(f"{i}. '{bigram}': {count:,}")

print("\nTop 15 most common trigrams:")
for i, (trigram, count) in enumerate(trigrams.most_common(15), 1):
    print(f"{i}. '{trigram}': {count:,}")

# Unique letters per word
unique_letters_count = [len(set(word)) for word in corpus]
print(f"\n" + "=" * 60)
print("UNIQUE LETTERS PER WORD")
print("=" * 60)
print(f"Average unique letters per word: {np.mean(unique_letters_count):.2f}")
print(f"Median unique letters per word: {np.median(unique_letters_count):.2f}")

# Word complexity analysis (ratio of unique letters to word length)
complexity_ratios = [len(set(word)) / len(word) for word in corpus]

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(unique_letters_count, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Unique Letters')
plt.ylabel('Frequency')
plt.title('Distribution of Unique Letters per Word')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(complexity_ratios, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Complexity Ratio (Unique Letters / Word Length)')
plt.ylabel('Frequency')
plt.title('Word Complexity Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('word_complexity.png', dpi=300, bbox_inches='tight')
plt.show()

# Save processed data for modeling
print("\n" + "=" * 60)
print("SAVING PROCESSED DATA")
print("=" * 60)

# Group words by length
words_by_length = {}
for word in corpus:
    length = len(word)
    if length not in words_by_length:
        words_by_length[length] = []
    words_by_length[length].append(word)

print(f"\nWords grouped by length:")
for length in sorted(words_by_length.keys()):
    print(f"Length {length}: {len(words_by_length[length])} words")

# Save statistics
import json

stats = {
    'total_words': len(corpus),
    'min_length': min(word_lengths),
    'max_length': max(word_lengths),
    'avg_length': float(np.mean(word_lengths)),
    'median_length': float(np.median(word_lengths)),
    'letter_frequency': dict(sorted_letters),
    'top_10_letters': [letter for letter, _ in sorted_letters[:10]],
    'words_by_length_count': {k: len(v) for k, v in words_by_length.items()}
}

with open('corpus_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\nStatistics saved to 'corpus_statistics.json'")
print("\nData exploration complete!")