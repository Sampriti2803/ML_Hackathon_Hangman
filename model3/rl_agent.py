import numpy as np
import random
import pickle
from collections import defaultdict
import string

class HangmanEnvironment:
    """
    Hangman game environment for RL training
    """
    
    def __init__(self, word_list, max_wrong_guesses=6):
        self.word_list = word_list
        self.max_wrong_guesses = max_wrong_guesses
        self.reset()
        
    def reset(self, word=None):
        """Reset environment for a new game"""
        if word is None:
            self.target_word = random.choice(self.word_list).lower()
        else:
            self.target_word = word.lower()
            
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        self.game_over = False
        self.won = False
        
        return self.get_state()
    
    def get_masked_word(self):
        """Return current word state with underscores for unguessed letters"""
        return ''.join([c if c in self.guessed_letters else '_' 
                       for c in self.target_word])
    
    def get_state(self):
        """
        Return current game state
        
        Returns:
            dict with:
                - masked_word: current visible state
                - guessed_letters: set of guessed letters
                - wrong_guesses: number of wrong guesses
                - lives_left: remaining lives
        """
        return {
            'masked_word': self.get_masked_word(),
            'guessed_letters': self.guessed_letters.copy(),
            'wrong_guesses': self.wrong_guesses,
            'lives_left': self.max_wrong_guesses - self.wrong_guesses,
            'target_length': len(self.target_word)
        }
    
    def step(self, action):
        """
        Take an action (guess a letter)
        
        Args:
            action: letter to guess (string)
            
        Returns:
            state: new state after action
            reward: reward for this action
            done: whether game is over
            info: additional information
        """
        letter = action.lower()
        
        # Check for repeated guess
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            reward = -10  # Heavy penalty for repeated guess
            return self.get_state(), reward, self.game_over, {'repeated': True}
        
        # Add to guessed letters
        self.guessed_letters.add(letter)
        
        # Check if letter is in word
        if letter in self.target_word:
            # Correct guess
            count = self.target_word.count(letter)
            reward = count * 5  # Reward proportional to number of occurrences
            
            # Check if word is complete
            if all(c in self.guessed_letters for c in self.target_word):
                self.game_over = True
                self.won = True
                reward += 50  # Bonus for winning
        else:
            # Wrong guess
            self.wrong_guesses += 1
            reward = -5  # Penalty for wrong guess
            
            # Check if game over
            if self.wrong_guesses >= self.max_wrong_guesses:
                self.game_over = True
                self.won = False
                reward -= 20  # Additional penalty for losing
        
        return self.get_state(), reward, self.game_over, {'repeated': False}
    
    def get_valid_actions(self):
        """Return list of valid actions (unguessed letters)"""
        return [l for l in string.ascii_lowercase if l not in self.guessed_letters]
    
class QLearningAgent:
    """
    Q-Learning agent that learns a policy over *strategies*, not individual letters.
    """
    
    def __init__(self, hmm_model, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.hmm = hmm_model
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Our actions are now strategies
        self.n_actions = 4  # (HMM_1, HMM_2, Vowel, Consonant)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        
        # From EDA (1_data_exploration.ipynb)
        self.common_vowels = ['e', 'a', 'i', 'o', 'u']
        self.common_consonants = ['r', 'n', 't', 's', 'l', 'c', 'd', 'p', 'm', 'h']
        
    def get_state_key(self, state):
        """
        Convert state dict to a hashable key for Q-table
        """
        masked_word = state['masked_word']
        guessed = ''.join(sorted(state['guessed_letters']))
        lives = state['lives_left']
        return f"{masked_word}:{guessed}:{lives}"

    def _get_letter_for_action(self, action_index, state, valid_actions):
        """Helper to map a strategy (action_index) to a specific letter."""
        
        # Get HMM probabilities
        hmm_probs = self.hmm.predict_letter_probabilities(
            state['masked_word'], 
            state['guessed_letters']
        )
        
        # Sort HMM guesses (only valid ones)
        hmm_guesses = sorted(
            [letter for letter in hmm_probs if letter in valid_actions],
            key=lambda l: hmm_probs[l], 
            reverse=True
        )
        
        if action_index == 0: # HMM Top 1
            if hmm_guesses:
                return hmm_guesses[0]
                
        elif action_index == 1: # HMM Top 2
            if len(hmm_guesses) > 1:
                return hmm_guesses[1]
            elif hmm_guesses:
                return hmm_guesses[0] # Fallback to Top 1
                
        elif action_index == 2: # Common Vowel
            for v in self.common_vowels:
                if v in valid_actions:
                    return v
                    
        elif action_index == 3: # Common Consonant
            for c in self.common_consonants:
                if c in valid_actions:
                    return c
        
        # Fallback: If strategy fails (e.g., no vowels left), just pick best HMM guess
        if hmm_guesses:
            return hmm_guesses[0]
        
        # Final fallback: random valid action
        if valid_actions:
            return random.choice(valid_actions)
            
        return None # No actions left
        
    def choose_action(self, state, env, training=True):
        """
        Choose a *strategy* using epsilon-greedy, then map to a *letter*.
        Returns: (action_index, letter_to_guess)
        """
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None, None
            
        state_key = self.get_state_key(state)
        
        # Exploration
        if training and random.random() < self.epsilon:
            action_index = random.randint(0, self.n_actions - 1)
        # Exploitation
        else:
            q_values = self.q_table[state_key]
            action_index = np.argmax(q_values)
            
        # Map strategy (action_index) to a specific letter
        letter_to_guess = self._get_letter_for_action(action_index, state, valid_actions)
        
        # Ensure we always return a valid action
        if letter_to_guess is None or letter_to_guess not in valid_actions:
             # This is a safety net
            if valid_actions:
                letter_to_guess = random.choice(valid_actions)
            else:
                return None, None # Game is stuck
        
        return action_index, letter_to_guess
    
    def update_q_value(self, state, action_index, reward, next_state, done):
        """
        Update Q-value using Q-learning formula for the *strategy*
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        current_q = self.q_table[state_key][action_index]
        
        if done:
            target_q = reward  # Terminal state
        else:
            # Get max Q-value for next state
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[state_key][action_index] = current_q + self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath='q_agent.pkl'):
        # ... (save function remains the same) ...
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'learning_rate': self.lr,
                'discount_factor': self.gamma
            }, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath='q_agent.pkl'):
        # ... (load function remains the same) ...
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data['q_table'])
            self.epsilon = data['epsilon']
            self.lr = data['learning_rate']
            self.gamma = data['discount_factor']
        print(f"Agent loaded from {filepath}")

def train_episode(env, agent, training=True):
    """
    Run one episode of training (modified for new action type)
    """
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Choose action
        action_index, letter_to_guess = agent.choose_action(state, env, training=training)
        
        if letter_to_guess is None:
            break
        
        # Take action
        next_state, reward, done, info = env.step(letter_to_guess)
        total_reward += reward
        
        # Update Q-values (only during training)
        if training:
            agent.update_q_value(state, action_index, reward, next_state, done)
        
        state = next_state
    
    if training:
        agent.decay_epsilon()
    
    return {
        'total_reward': total_reward,
        'won': env.won,
        'wrong_guesses': env.wrong_guesses,
        'repeated_guesses': env.repeated_guesses,
        'target_word': env.target_word
    }

def evaluate_agent(env, agent, n_episodes=100):
    """
    Evaluate agent performance (uses the new train_episode)
    """
    # ... (this function remains the same) ...
    results = []
    
    for _ in range(n_episodes):
        result = train_episode(env, agent, training=False)
        results.append(result)
    
    wins = sum(1 for r in results if r['won'])
    total_wrong = sum(r['wrong_guesses'] for r in results)
    total_repeated = sum(r['repeated_guesses'] for r in results)
    avg_reward = np.mean([r['total_reward'] for r in results])
    
    return {
        'success_rate': wins / n_episodes,
        'avg_wrong_guesses': total_wrong / n_episodes,
        'avg_repeated_guesses': total_repeated / n_episodes,
        'avg_reward': avg_reward,
        'total_wrong': total_wrong,
        'total_repeated': total_repeated
    }