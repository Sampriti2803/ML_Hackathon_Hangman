"""
Hangman ML Hackathon - Part 3: Reinforcement Learning Environment & Agent
"""

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
    Q-Learning agent for Hangman with HMM integration
    """
    
    def __init__(self, hmm_model, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.hmm = hmm_model
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alphabet = list(string.ascii_lowercase)
        
    def get_state_key(self, state):
        """
        Convert state dict to a hashable key for Q-table
        """
        masked_word = state['masked_word']
        guessed = ''.join(sorted(state['guessed_letters']))
        lives = state['lives_left']
        return f"{masked_word}:{guessed}:{lives}"
    
    def choose_action(self, state, env, training=True):
        """
        Choose action using epsilon-greedy policy with HMM guidance
        """
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return None
        
        # Exploration
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: combine Q-values with HMM probabilities
        state_key = self.get_state_key(state)
        hmm_probs = self.hmm.predict_letter_probabilities(
            state['masked_word'], 
            state['guessed_letters']
        )
        
        # Calculate combined score for each action
        action_scores = {}
        for action in valid_actions:
            q_value = self.q_table[state_key][action]
            hmm_prob = hmm_probs.get(action, 0.01)
            
            # Weighted combination: Q-value and HMM probability
            # Adjust weights based on how much experience we have
            if abs(q_value) < 0.1:  # Little experience, trust HMM more
                combined_score = 0.2 * q_value + 0.8 * hmm_prob
            else:  # More experience, trust Q-values more
                combined_score = 0.7 * q_value + 0.3 * hmm_prob
            
            action_scores[action] = combined_score
        
        # Choose action with highest combined score
        return max(action_scores.items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning formula
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            # Terminal state
            target_q = reward
        else:
            # Get max Q-value for next state
            next_q_values = self.q_table[next_state_key]
            max_next_q = max(next_q_values.values()) if next_q_values else 0
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath='q_agent.pkl'):
        """Save Q-table and parameters"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor
            }, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath='q_agent.pkl'):
        """Load Q-table and parameters"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
            self.epsilon = data['epsilon']
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']
        print(f"Agent loaded from {filepath}")


class DQNAgent:
    """
    Deep Q-Network agent for more complex state representations
    (Optional advanced implementation)
    """
    
    def __init__(self, hmm_model, state_size, action_size):
        # This would use a neural network for Q-value approximation
        # Implementation left for advanced users
        pass


# Training utilities
def train_episode(env, agent, training=True):
    """
    Run one episode of training
    
    Returns:
        total_reward: cumulative reward for episode
        won: whether the game was won
        wrong_guesses: number of wrong guesses
        repeated_guesses: number of repeated guesses
    """
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Choose action
        action = agent.choose_action(state, env, training=training)
        
        if action is None:
            break
        
        # Take action
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Update Q-values (only during training)
        if training:
            agent.update_q_value(state, action, reward, next_state, done)
        
        state = next_state
    
    # Decay epsilon after episode
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
    Evaluate agent performance
    """
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