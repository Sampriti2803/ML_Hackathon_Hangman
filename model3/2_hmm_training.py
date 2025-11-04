"""
Hangman ML Hackathon - Part 2: Hidden Markov Model Training
"""

import numpy as np
import pickle
from collections import defaultdict, Counter
import string
from tqdm import tqdm

class HangmanHMM:
    """
    Hidden Markov Model for Hangman
    
    Hidden States: Position in the word (0, 1, 2, ..., word_length-1)
    Emissions: Letters (a-z)
    
    We train separate HMMs for each word length to handle variable-length words.
    """
    
    def __init__(self):
        self.models_by_length = {}
        self.alphabet = list(string.ascii_lowercase)
        self.letter_to_idx = {letter: idx for idx, letter in enumerate(self.alphabet)}
        self.idx_to_letter = {idx: letter for letter, idx in self.letter_to_idx.items()}
        
    def train(self, corpus_file='corpus.txt', max_length=None):
        """
        Train HMM models for each word length
        """
        print("Loading corpus...")
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        # Group words by length
        words_by_length = defaultdict(list)
        for word in words:
            if all(c in self.alphabet for c in word):  # Only valid words
                words_by_length[len(word)].append(word)
        
        print(f"\nTraining HMMs for {len(words_by_length)} different word lengths...")
        
        for length, word_list in tqdm(sorted(words_by_length.items())):
            if max_length and length > max_length:
                continue
                
            self.models_by_length[length] = self._train_single_hmm(word_list, length)
        
        print(f"\nTraining complete! Trained models for word lengths: {sorted(self.models_by_length.keys())}")
        
    def _train_single_hmm(self, words, length):
        """
        Train a single HMM for words of a specific length
        
        Returns:
            model: dict with 'initial', 'transition', 'emission' probabilities
        """
        n_states = length  # Number of positions
        n_emissions = len(self.alphabet)  # 26 letters
        
        # Initialize counts
        initial_counts = np.zeros(n_states)
        transition_counts = np.zeros((n_states, n_states))
        emission_counts = np.zeros((n_states, n_emissions))
        
        # Count occurrences
        for word in words:
            for pos, letter in enumerate(word):
                if letter in self.letter_to_idx:
                    letter_idx = self.letter_to_idx[letter]
                    
                    # Initial state (always starts at position 0)
                    if pos == 0:
                        initial_counts[pos] += 1
                    
                    # Emission (letter at position)
                    emission_counts[pos, letter_idx] += 1
                    
                    # Transition (position to next position)
                    if pos < length - 1:
                        transition_counts[pos, pos + 1] += 1
        
        # Convert counts to probabilities with Laplace smoothing
        alpha = 0.01  # Smoothing parameter
        
        # Initial probabilities (always start at position 0)
        initial_probs = np.zeros(n_states)
        initial_probs[0] = 1.0
        
        # Transition probabilities (sequential positions)
        transition_probs = np.zeros((n_states, n_states))
        for i in range(n_states - 1):
            transition_probs[i, i + 1] = 1.0
        
        # Emission probabilities with smoothing
        emission_probs = np.zeros((n_states, n_emissions))
        for pos in range(n_states):
            total = emission_counts[pos].sum() + alpha * n_emissions
            emission_probs[pos] = (emission_counts[pos] + alpha) / total
        
        return {
            'initial': initial_probs,
            'transition': transition_probs,
            'emission': emission_probs,
            'length': length,
            'n_words': len(words)
        }
    
    def predict_letter_probabilities(self, masked_word, guessed_letters):
        """
        Given a masked word (e.g., "_a_e_") and guessed letters,
        return probability distribution over remaining letters
        
        Args:
            masked_word: str with '_' for unknown positions
            guessed_letters: set of already guessed letters
            
        Returns:
            dict: {letter: probability} for unguessed letters
        """
        length = len(masked_word)
        
        if length not in self.models_by_length:
            # Fallback to general letter frequency
            return self._fallback_probabilities(guessed_letters)
        
        model = self.models_by_length[length]
        emission_probs = model['emission']
        
        # Calculate probability for each possible letter
        letter_probs = np.zeros(len(self.alphabet))
        
        for pos, char in enumerate(masked_word):
            if char == '_':
                # Unknown position - add emission probabilities
                letter_probs += emission_probs[pos]
            else:
                # Known position - reinforce this letter
                if char in self.letter_to_idx:
                    letter_idx = self.letter_to_idx[char]
                    letter_probs[letter_idx] += emission_probs[pos, letter_idx] * 2
        
        # Normalize
        if letter_probs.sum() > 0:
            letter_probs /= letter_probs.sum()
        else:
            letter_probs = np.ones(len(self.alphabet)) / len(self.alphabet)
        
        # Filter out guessed letters
        result = {}
        for letter, prob in zip(self.alphabet, letter_probs):
            if letter not in guessed_letters:
                result[letter] = float(prob)
        
        # Renormalize
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        
        return result
    
    def _fallback_probabilities(self, guessed_letters):
        """
        Fallback to uniform distribution for unguessed letters
        """
        remaining = [l for l in self.alphabet if l not in guessed_letters]
        if not remaining:
            return {}
        
        prob = 1.0 / len(remaining)
        return {letter: prob for letter in remaining}
    
    def save(self, filepath='hmm_model.pkl'):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models_by_length': self.models_by_length,
                'alphabet': self.alphabet,
                'letter_to_idx': self.letter_to_idx,
                'idx_to_letter': self.idx_to_letter
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath='hmm_model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models_by_length = data['models_by_length']
            self.alphabet = data['alphabet']
            self.letter_to_idx = data['letter_to_idx']
            self.idx_to_letter = data['idx_to_letter']
        print(f"Model loaded from {filepath}")
        print(f"Available word lengths: {sorted(self.models_by_length.keys())}")


# Training and validation
if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING HIDDEN MARKOV MODEL")
    print("=" * 60)
    
    # Initialize and train
    hmm = HangmanHMM()
    hmm.train('corpus.txt')
    
    # Display model statistics
    print("\n" + "=" * 60)
    print("MODEL STATISTICS")
    print("=" * 60)
    
    for length in sorted(hmm.models_by_length.keys()):
        model = hmm.models_by_length[length]
        print(f"\nLength {length}:")
        print(f"  Training words: {model['n_words']}")
        print(f"  States: {model['length']}")
        print(f"  Emission shape: {model['emission'].shape}")
    
    # Test the model
    print("\n" + "=" * 60)
    print("MODEL TESTING")
    print("=" * 60)
    
    test_cases = [
        ("_____", set()),
        ("_a___", {'a'}),
        ("_a__e", {'a', 'e'}),
        ("ha__e", {'h', 'a', 'e'}),
    ]
    
    for masked_word, guessed in test_cases:
        probs = hmm.predict_letter_probabilities(masked_word, guessed)
        top_5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\nMasked word: {masked_word}")
        print(f"Guessed: {guessed if guessed else 'None'}")
        print(f"Top 5 predictions:")
        for i, (letter, prob) in enumerate(top_5, 1):
            print(f"  {i}. {letter}: {prob:.4f}")
    
    # Save the model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    hmm.save('hmm_model.pkl')
    
    # Verify save/load
    print("\nVerifying save/load...")
    hmm_test = HangmanHMM()
    hmm_test.load('hmm_model.pkl')
    
    # Test loaded model
    test_word = "_e___"
    probs = hmm_test.predict_letter_probabilities(test_word, {'e'})
    top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\nTest prediction on '{test_word}':")
    for letter, prob in top_3:
        print(f"  {letter}: {prob:.4f}")
    
    print("\nHMM training complete!")