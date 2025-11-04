"""
Hangman ML Hackathon - Part 2: Hidden Markov Model Training
(Upgraded with Forward-Backward Algorithm for Inference)
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
    
    Training uses Maximum Likelihood Estimation (counting) since states are known.
    Inference (predicting) uses the Forward-Backward algorithm to
    calculate letter probabilities based on known evidence (guessed letters).
    """
    
    def __init__(self):
        self.models_by_length = {}
        self.alphabet = list(string.ascii_lowercase)
        self.letter_to_idx = {letter: idx for idx, letter in enumerate(self.alphabet)}
        self.idx_to_letter = {idx: letter for letter, idx in self.letter_to_idx.items()}
    
    def train(self, corpus_file='corpus.txt', max_length=None):
        """
        Train HMM models for each word length
        (This function remains unchanged)
        """
        print("Loading corpus...")
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        words_by_length = defaultdict(list)
        for word in words:
            if all(c in self.alphabet for c in word):
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

    # --- NEW HELPER FUNCTIONS FOR FORWARD-BACKWARD ---
    
    def _get_observation_probs(self, model, masked_word, guessed_letters):
        """
        Get the probability of the *evidence* at each position.
        """
        length = len(masked_word)
        emission_probs = model['emission']
        n_emissions = len(self.alphabet)
        
        # B[i] = P(Observation_i | State_i)
        B = np.ones(length)
        
        for i in range(length):
            char = masked_word[i]
            if char == '_':
                # If blank, the observation is "not any of the guessed letters"
                prob = 0.0
                for l_idx in range(n_emissions):
                    if self.idx_to_letter[l_idx] not in guessed_letters:
                        prob += emission_probs[i, l_idx]
                B[i] = prob
            else:
                # If a known letter, the observation is just that letter
                B[i] = emission_probs[i, self.letter_to_idx[char]]
        return B

    def _calculate_forward_pass(self, model, B):
        """
        Calculates the forward probabilities alpha.
        alpha[t] = P(O_0, O_1, ..., O_t, State_t)
        """
        length = model['length']
        initial = model['initial']
        transition = model['transition']
        
        alpha = np.zeros(length)
        
        # Initialization (t=0)
        # P(O_0, State_0) = P(State_0) * P(O_0 | State_0)
        alpha[0] = initial[0] * B[0]
        
        # Recursion (t=1 to length-1)
        for t in range(1, length):
            # Since our transitions are fixed (t-1 -> t), this is simplified
            # P(O_0..t, State_t) = P(O_t | State_t) * P(O_0..t-1, State_t-1) * P(State_t | State_t-1)
            alpha[t] = B[t] * alpha[t-1] * transition[t-1, t]
            
        return alpha

    def _calculate_backward_pass(self, model, B):
        """
        Calculates the backward probabilities beta.
        beta[t] = P(O_t+1, ..., O_T | State_t)
        """
        length = model['length']
        transition = model['transition']
        
        beta = np.zeros(length)
        
        # Initialization (t=length-1)
        beta[length - 1] = 1.0  # Prob of future given end state is 1
        
        # Recursion (t=length-2 down to 0)
        for t in range(length - 2, -1, -1):
            # P(O_t+1..T | State_t) = P(State_t+1 | State_t) * P(O_t+1 | State_t+1) * P(O_t+2..T | State_t+1)
            beta[t] = transition[t, t+1] * B[t+1] * beta[t+1]
            
        return beta

    # --- REPLACED PREDICTION FUNCTION ---

    def predict_letter_probabilities(self, masked_word, guessed_letters):
        """
        Given a masked word (e.g., "_a_e_") and guessed letters,
        return probability distribution over remaining letters using
        the Forward-Backward algorithm for contextual inference.
        """
        length = len(masked_word)
        if length not in self.models_by_length:
            return self._fallback_probabilities(guessed_letters)
            
        model = self.models_by_length[length]
        emission_probs = model['emission']
        
        valid_actions = [l for l in self.alphabet if l not in guessed_letters]
        if not valid_actions:
            return {}
            
        blank_indices = [i for i, char in enumerate(masked_word) if char == '_']
        if not blank_indices:
            return {}

        # 1. Get prob of evidence at each step (B)
        B = self._get_observation_probs(model, masked_word, guessed_letters)

        # 2. Calculate forward (alpha) and backward (beta) passes
        alpha = self._calculate_forward_pass(model, B)
        beta = self._calculate_backward_pass(model, B)
        
        # Total probability of the observation sequence
        prob_obs = alpha[length - 1]
        
        if prob_obs == 0.0:
            # This pattern is impossible given our model, fallback
            return self._fallback_probabilities(guessed_letters)
            
        # 3. Calculate letter probabilities for blanks
        final_probs = {letter: 0.0 for letter in self.alphabet}
        
        for i in blank_indices:
            # P(State_i | Observations) = (alpha[i] * beta[i]) / P(Observations)
            prob_state_i = (alpha[i] * beta[i]) / prob_obs
            
            # Sum the probability of each letter at this blank
            for letter in valid_actions:
                l_idx = self.letter_to_idx[letter]
                # P(Letter | State_i)
                prob_emission = emission_probs[i, l_idx]
                
                # P(Letter_i | Observations) = P(State_i | Obs) * P(Letter | State_i)
                final_probs[letter] += prob_state_i * prob_emission

        # 4. Normalize results
        total_prob_sum = sum(final_probs.values())
        if total_prob_sum == 0.0:
            return self._fallback_probabilities(guessed_letters)
            
        normalized_probs = {l: p / total_prob_sum for l, p in final_probs.items() if l in valid_actions}
        
        return normalized_probs

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
                'idx_to_letter': self.idx_to_letter
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
        print(f"Model loaded from {filepath}")
        print(f"Available word lengths: {sorted(self.models_by_length.keys())}")


# --- Training and validation (Unchanged) ---
if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING HIDDEN MARKOV MODEL")
    print("=" * 60)
    
    hmm = HangmanHMM()
    hmm.train('corpus.txt')
    
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
    print("MODEL TESTING (with Forward-Backward)")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        ("_____"  , set()),             # apple
        ("a____"  , {'a'}),             # apple
        ("ap___"  , {'a', 'p'}),        # apple
        ("appl_"  , {'a', 'p', 'l'}),   # apple
        ("_a__e"  , {'a', 'e'}),        # state
        ("s_a_e"  , {'s', 'a', 'e'}),   # state
    ]
    
    for masked_word, guessed in test_cases:
        probs = hmm.predict_letter_probabilities(masked_word, guessed)
        top_5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\nMasked word: {masked_word}")
        print(f"Guessed: {guessed if guessed else 'None'}")
        print(f"Top 5 predictions (context-aware):")
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