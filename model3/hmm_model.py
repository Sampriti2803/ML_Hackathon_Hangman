"""
Hangman ML Hackathon - Part 2: Hidden Markov Model Training
(Upgraded with Forward-Backward Algorithm and N-Gram Interpolation)
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
    
    This model combines three probability sources:
    1. Positional HMM (Forward-Backward): P(letter | position and word-level context)
    2. Bigram Model: P(letter | previous_letter)
    3. Trigram Model: P(letter | two_previous_letters)
    """
    
    def __init__(self):
        self.models_by_length = {}
        self.alphabet = list(string.ascii_lowercase)
        self.letter_to_idx = {letter: idx for idx, letter in enumerate(self.alphabet)}
        self.idx_to_letter = {idx: letter for letter, idx in self.letter_to_idx.items()}
        # These will store the final probabilities, e.g., self.bigram_probs['t']['h'] = 0.15
        self.bigram_probs = defaultdict(lambda: defaultdict(float))
        self.trigram_probs = defaultdict(lambda: defaultdict(float))
    
    def train(self, corpus_file='corpus.txt', max_length=None):
        """
        Train HMM models for each word length and N-gram models
        """
        print("Loading corpus...")
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        # --- Train Positional HMMs ---
        words_by_length = defaultdict(list)
        valid_words = [] # <-- NEW: Store valid words for N-gram training
        for word in words:
            if all(c in self.alphabet for c in word):
                words_by_length[len(word)].append(word)
                valid_words.append(word) # <-- NEW
        
        print(f"\nTraining HMMs for {len(words_by_length)} different word lengths...")
        for length, word_list in tqdm(sorted(words_by_length.items())):
            if max_length and length > max_length:
                continue
            self.models_by_length[length] = self._train_single_hmm(word_list, length)
        
        print(f"\nTraining complete! Trained models for word lengths: {sorted(self.models_by_length.keys())}")

        # --- Train N-Gram Models ---
        print("\nTraining N-gram models...")
        self._train_ngrams(valid_words) # <-- NEW
        print("N-gram training complete!")
        
    
    def _train_ngrams(self, words): # <-- NEW METHOD
        """
        Train bigram and trigram probability models from the corpus
        """
        bigram_counts = defaultdict(Counter)
        trigram_counts = defaultdict(Counter)
        bigram_context_totals = Counter()
        trigram_context_totals = Counter()
        
        # Add smoothing parameter
        alpha = 1.0
        vocab_size = len(self.alphabet)

        # 1. Count all occurrences
        for word in words:
            # Bigrams
            for i in range(len(word) - 1):
                context = word[i]
                letter = word[i+1]
                bigram_counts[context][letter] += 1
                bigram_context_totals[context] += 1
            
            # Trigrams
            for i in range(len(word) - 2):
                context = word[i:i+2] # e.g., "th"
                letter = word[i+2]    # e.g., "e"
                trigram_counts[context][letter] += 1
                trigram_context_totals[context] += 1

        # 2. Normalize counts into probabilities with Laplace smoothing
        print("Normalizing bigram probabilities...")
        for context, letter_counts in tqdm(bigram_counts.items()):
            total = bigram_context_totals[context]
            for letter, count in letter_counts.items():
                self.bigram_probs[context][letter] = (count + alpha) / (total + (alpha * vocab_size))

        print("Normalizing trigram probabilities...")
        for context, letter_counts in tqdm(trigram_counts.items()):
            total = trigram_context_totals[context]
            for letter, count in letter_counts.items():
                self.trigram_probs[context][letter] = (count + alpha) / (total + (alpha * vocab_size))

    def _train_single_hmm(self, words, length):
        """
        Train a single HMM for words of a specific length
        (This function remains unchanged)
        """
        n_states = length
        n_emissions = len(self.alphabet)
        
        emission_counts = np.zeros((n_states, n_emissions))
        
        for word in words:
            for pos, letter in enumerate(word):
                if letter in self.letter_to_idx:
                    letter_idx = self.letter_to_idx[letter]
                    emission_counts[pos, letter_idx] += 1
        
        alpha = 0.01  # Smoothing parameter
        
        initial_probs = np.zeros(n_states)
        initial_probs[0] = 1.0
        
        transition_probs = np.zeros((n_states, n_states))
        for i in range(n_states - 1):
            transition_probs[i, i + 1] = 1.0
        
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

    # --- NEW: HELPER TO NORMALIZE DICTIONARIES ---
    
    def _normalize_dict(self, prob_dict, valid_actions): # <-- NEW METHOD
        """
        Normalizes a dictionary of probabilities so they sum to 1,
        considering only valid (unguessed) actions.
        """
        if not prob_dict:
            return {}
            
        total_prob_sum = 0.0
        for letter in valid_actions:
            total_prob_sum += prob_dict.get(letter, 0.0)

        if total_prob_sum == 0.0:
            return {} # Fallback, though _fallback_probabilities is safer
            
        normalized_probs = {
            letter: prob_dict.get(letter, 0.0) / total_prob_sum
            for letter in valid_actions
        }
        
        return normalized_probs

    # --- FORWARD-BACKWARD FUNCTIONS (UNCHANGED) ---
    
    def _get_observation_probs(self, model, masked_word, guessed_letters):
        """
        Get the probability of the *evidence* at each position.
        (Unchanged)
        """
        length = len(masked_word)
        emission_probs = model['emission']
        n_emissions = len(self.alphabet)
        
        B = np.ones(length)
        
        for i in range(length):
            char = masked_word[i]
            if char == '_':
                prob = 0.0
                for l_idx in range(n_emissions):
                    if self.idx_to_letter[l_idx] not in guessed_letters:
                        prob += emission_probs[i, l_idx]
                B[i] = prob
            else:
                B[i] = emission_probs[i, self.letter_to_idx[char]]
        return B

    def _calculate_forward_pass(self, model, B):
        """
        Calculates the forward probabilities alpha.
        (Unchanged)
        """
        length = model['length']
        initial = model['initial']
        transition = model['transition']
        
        alpha = np.zeros(length)
        alpha[0] = initial[0] * B[0]
        
        for t in range(1, length):
            alpha[t] = B[t] * alpha[t-1] * transition[t-1, t]
            
        return alpha

    def _calculate_backward_pass(self, model, B):
        """
        Calculates the backward probabilities beta.
        (Unchanged)
        """
        length = model['length']
        transition = model['transition']
        
        beta = np.zeros(length)
        beta[length - 1] = 1.0
        
        for t in range(length - 2, -1, -1):
            beta[t] = transition[t, t+1] * B[t+1] * beta[t+1]
            
        return beta

    # --- NEW: SEPARATE PROBABILITY CALCULATION METHODS ---

    def _get_positional_probs(self, model, masked_word, guessed_letters, blank_indices, valid_actions): # <-- NEW (Refactored from old predict function)
        """
        Uses Forward-Backward to get P(letter | position, word context)
        """
        emission_probs = model['emission']
        
        # 1. Get prob of evidence at each step (B)
        B = self._get_observation_probs(model, masked_word, guessed_letters)

        # 2. Calculate forward (alpha) and backward (beta) passes
        alpha = self._calculate_forward_pass(model, B)
        beta = self._calculate_backward_pass(model, B)
        
        prob_obs = alpha[len(masked_word) - 1]
        
        if prob_obs == 0.0:
            return {} # This pattern is impossible
            
        # 3. Calculate letter probabilities for blanks
        final_probs = {letter: 0.0 for letter in self.alphabet}
        
        for i in blank_indices:
            prob_state_i = (alpha[i] * beta[i]) / prob_obs
            
            for letter in valid_actions:
                l_idx = self.letter_to_idx[letter]
                prob_emission = emission_probs[i, l_idx]
                final_probs[letter] += prob_state_i * prob_emission

        return self._normalize_dict(final_probs, valid_actions)

    def _get_bigram_probs(self, masked_word, guessed_letters, blank_indices, valid_actions): # <-- NEW
        """
        Calculates P(letter | previous_letter) for all blanks
        """
        final_probs = {letter: 0.0 for letter in self.alphabet}
        
        for i in blank_indices:
            if i > 0 and masked_word[i-1] != '_':
                context = masked_word[i-1]
                prob_dist = self.bigram_probs.get(context)
                
                if prob_dist:
                    # Context was found, add its probabilities
                    for letter in valid_actions:
                        final_probs[letter] += prob_dist.get(letter, 0.0) # Use 0.0, smoothing was done at training
        
        return self._normalize_dict(final_probs, valid_actions)

    def _get_trigram_probs(self, masked_word, guessed_letters, blank_indices, valid_actions): # <-- NEW
        """
        Calculates P(letter | two_previous_letters) for all blanks
        """
        final_probs = {letter: 0.0 for letter in self.alphabet}
        
        for i in blank_indices:
            if i > 1 and masked_word[i-2] != '_' and masked_word[i-1] != '_':
                context = masked_word[i-2:i] # e.g., "th"
                prob_dist = self.trigram_probs.get(context)
                
                if prob_dist:
                    # Context was found, add its probabilities
                    for letter in valid_actions:
                        final_probs[letter] += prob_dist.get(letter, 0.0)
        
        return self._normalize_dict(final_probs, valid_actions)


    # --- MODIFIED PREDICTION FUNCTION (NOW A HYBRID MODEL) ---

    def predict_letter_probabilities(self, masked_word, guessed_letters): # <-- MODIFIED
        """
        Given a masked word (e.g., "_a_e_") and guessed letters,
        return a combined probability distribution over remaining letters.
        """
        length = len(masked_word)
        if length not in self.models_by_length:
            return self._fallback_probabilities(guessed_letters)
            
        model = self.models_by_length[length]
        
        valid_actions = [l for l in self.alphabet if l not in guessed_letters]
        if not valid_actions:
            return {}
            
        blank_indices = [i for i, char in enumerate(masked_word) if char == '_']
        if not blank_indices:
            return {}

        # --- Define weights for combining models ---
        # These are tunable hyperparameters.
        # W_POS: Positional HMM (Forward-Backward)
        # W_BI: Bigram model
        # W_TRI: Trigram model
        W_POS = 0.4
        W_BI  = 0.3
        W_TRI = 0.3
        
        # 1. Get Positional HMM probabilities
        P_pos = self._get_positional_probs(model, masked_word, guessed_letters, blank_indices, valid_actions)
        
        # 2. Get Bigram probabilities
        P_bi = self._get_bigram_probs(masked_word, guessed_letters, blank_indices, valid_actions)
        
        # 3. Get Trigram probabilities
        P_tri = self._get_trigram_probs(masked_word, guessed_letters, blank_indices, valid_actions)

        # 4. Combine results
        combined_probs = {letter: 0.0 for letter in self.alphabet}
        
        # If N-gram models fail (no context), give all weight to positional model
        if not P_bi and not P_tri:
            W_POS = 1.0
            
        for letter in valid_actions:
            pos_prob = P_pos.get(letter, 0.0)
            bi_prob = P_bi.get(letter, 0.0)
            tri_prob = P_tri.get(letter, 0.0)
            
            combined_probs[letter] = (W_POS * pos_prob) + (W_BI * bi_prob) + (W_TRI * tri_prob)

        # 5. Normalize and return
        final_normalized = self._normalize_dict(combined_probs, valid_actions)
        
        if not final_normalized:
            # Fallback if all models failed
            return self._fallback_probabilities(guessed_letters)
            
        return final_normalized

    def _fallback_probabilities(self, guessed_letters):
        """
        (Unchanged) Fallback to uniform distribution for unguessed letters
        """
        remaining = [l for l in self.alphabet if l not in guessed_letters]
        if not remaining:
            return {}
        prob = 1.0 / len(remaining)
        return {letter: prob for letter in remaining}
    
    def save(self, filepath='hmm_model.pkl'):
        # ... (This function remains unchanged) ...
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models_by_length': self.models_by_length,
                'alphabet': self.alphabet,
                'letter_to_idx': self.letter_to_idx,
                'idx_to_letter': self.idx_to_letter,
                # Convert defaultdicts to regular dicts for pickling
                'bigram_probs': dict(self.bigram_probs),   # <-- FIXED
                'trigram_probs': dict(self.trigram_probs) # <-- FIXED
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath='hmm_model.pkl'):
        # ... (This function remains unchanged) ...
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models_by_length = data['models_by_length']
            self.alphabet = data['alphabet']
            self.letter_to_idx = data['letter_to_idx']
            self.idx_to_letter = data['idx_to_letter']
            
            # <-- MODIFIED: Load N-gram models, wrapping in defaultdict
            if 'bigram_probs' in data:
                self.bigram_probs = defaultdict(lambda: defaultdict(float), data['bigram_probs'])
                self.trigram_probs = defaultdict(lambda: defaultdict(float), data['trigram_probs'])
                print("N-gram models loaded successfully.")
            else:
                print("Warning: N-gram models not found in .pkl file. Initializing empty models.")
            
        print(f"Model loaded from {filepath}")
        print(f"Available word lengths: {sorted(self.models_by_length.keys())}")


# --- Training and validation (Unchanged) ---
if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING HIDDEN MARKOV MODEL (with N-Grams)") # <-- MODIFIED
    print("=" * 60)
    
    hmm = HangmanHMM()
    hmm.train('corpus.txt') # <-- This now trains all models
    
    print("\n" + "=" * 60)
    print("MODEL STATISTICS")
    print("=" * 60)
    
    for length in sorted(hmm.models_by_length.keys()):
        if length > 8: continue # Just show a few
        model = hmm.models_by_length[length]
        print(f"\nLength {length}:")
        print(f"  Training words: {model['n_words']}")
        print(f"  States: {model['length']}")
        print(f"  Emission shape: {model['emission'].shape}")
    
    print("\n" + "=" * 60)
    print("MODEL TESTING (Hybrid Model)") # <-- MODIFIED
    print("=" * 60)
    
    # Test cases
    test_cases = [
        ("_____"  , set()),             # apple
        ("q____"  , {'q'}),             # Should predict 'u'
        ("qu___"  , {'q', 'u'}),        # Should predict 'i', 'a'
        ("_a__e"  , {'a', 'e'}),        # state
        ("s_a_e"  , {'s', 'a', 'e'}),   # state
    ]
    
    for masked_word, guessed in test_cases:
        probs = hmm.predict_letter_probabilities(masked_word, guessed)
        top_5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\nMasked word: {masked_word}")
        print(f"Guessed: {guessed if guessed else 'None'}")
        print(f"Top 5 predictions (hybrid):") # <-- MODIFIED
        for i, (letter, prob) in enumerate(top_5, 1):
            print(f"  {i}. {letter}: {prob:.4f}")
    
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    hmm.save('hmm_model.pkl')
    
    print("\nVerifying save/load...")
    hmm_test = HangmanHMM()
    hmm_test.load('hmm_model.pkl')
    
    print("\nHMM training complete!")