"""
Hangman ML Hackathon - Part 4: Complete Training Loop
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import json

# Import our custom modules
from hmm_model import HangmanHMM
from rl_agent import HangmanEnvironment, QLearningAgent, train_episode, evaluate_agent


def plot_training_progress(training_history, window_size=100):
    # ... (this function remains the same) ...
    episodes = list(range(len(training_history['rewards'])))
    
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards
    ax = axes[0, 0]
    rewards = training_history['rewards']
    if len(rewards) > window_size:
        smoothed = moving_average(rewards, window_size)
        ax.plot(episodes[window_size-1:], smoothed, label='Smoothed', linewidth=2) # Corrected x-axis
    ax.plot(episodes, rewards, alpha=0.3, label='Raw')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Win rate
    ax = axes[0, 1]
    win_rate = training_history['win_rate']
    if len(win_rate) > window_size:
        smoothed = moving_average(win_rate, window_size)
        ax.plot(episodes[window_size-1:], smoothed, label='Smoothed', linewidth=2) # Corrected x-axis
    ax.plot(episodes, win_rate, alpha=0.3, label='Raw')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Wrong guesses
    ax = axes[1, 0]
    wrong_guesses = training_history['wrong_guesses']
    if len(wrong_guesses) > window_size:
        smoothed = moving_average(wrong_guesses, window_size)
        ax.plot(episodes[window_size-1:], smoothed, label='Smoothed', linewidth=2) # Corrected x-axis
    ax.plot(episodes, wrong_guesses, alpha=0.3, label='Raw')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Wrong Guesses')
    ax.set_title('Wrong Guesses per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Epsilon decay
    ax = axes[1, 1]
    epsilon = training_history['epsilon']
    ax.plot(episodes, epsilon, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (Epsilon) Decay')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_evaluation_progress(eval_results):
    # ... (this function remains the same) ...
    if not eval_results:
        return
    
    episodes = [r['episode'] for r in eval_results]
    success_rates = [r['success_rate'] for r in eval_results]
    avg_wrong = [r['avg_wrong_guesses'] for r in eval_results]
    avg_repeated = [r['avg_repeated_guesses'] for r in eval_results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(episodes, success_rates, marker='o', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Validation Success Rate')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.1])
    
    axes[1].plot(episodes, avg_wrong, marker='o', linewidth=2, color='red')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Avg Wrong Guesses')
    axes[1].set_title('Average Wrong Guesses')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(episodes, avg_repeated, marker='o', linewidth=2, color='orange')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Avg Repeated Guesses')
    axes[2].set_title('Average Repeated Guesses')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_progress.png', dpi=300, bbox_inches='tight')
    plt.show()


def train_agent(corpus_file='corpus.txt', n_episodes=10000, eval_interval=500):
    """
    Complete training pipeline
    """
    print("=" * 70)
    print("HANGMAN RL TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load corpus
    print("\n[1/5] Loading corpus...")
    with open(corpus_file, 'r') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    print(f"Loaded {len(words)} words")
    
    np.random.shuffle(words)
    split_idx = int(0.9 * len(words))
    train_words = words[:split_idx]
    val_words = words[split_idx:]
    print(f"Training set: {len(train_words)} words")
    print(f"Validation set: {len(val_words)} words")
    
    # Step 2: Load or train HMM
    print("\n[2/5] Loading HMM model...")
    try:
        hmm = HangmanHMM()
        hmm.load('hmm_model.pkl')
    except:
        print("HMM model not found. Please run HMM training first.")
        return
    
    # Step 3: Initialize environment and agent
    print("\n[3/5] Initializing environment and agent...")
    env = HangmanEnvironment(train_words, max_wrong_guesses=6)
    agent = QLearningAgent(
        hmm_model=hmm,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.9995, # Kept your decay rate
        epsilon_min=0.05
    )
    
    # Step 4: Train agent
    print(f"\n[4/5] Training agent for {n_episodes} episodes...")
    print("This may take a while...\n")
    
    training_history = {
        'rewards': [],
        'win_rate': [],
        'wrong_guesses': [],
        'repeated_guesses': [],
        'epsilon': []
    }
    
    eval_results = []
    
    for episode in tqdm(range(n_episodes)):
        result = train_episode(env, agent, training=True)
        
        training_history['rewards'].append(result['total_reward'])
        training_history['win_rate'].append(1.0 if result['won'] else 0.0)
        training_history['wrong_guesses'].append(result['wrong_guesses'])
        training_history['repeated_guesses'].append(result['repeated_guesses'])
        training_history['epsilon'].append(agent.epsilon)
        
        if (episode + 1) % eval_interval == 0:
            print(f"\n--- Evaluation at episode {episode + 1} ---")
            val_env = HangmanEnvironment(val_words, max_wrong_guesses=6)
            eval_result = evaluate_agent(val_env, agent, n_episodes=100)
            
            print(f"Success Rate: {eval_result['success_rate']:.2%}")
            print(f"Avg Wrong Guesses: {eval_result['avg_wrong_guesses']:.2f}")
            print(f"Avg Repeated Guesses: {eval_result['avg_repeated_guesses']:.2f}")
            print(f"Avg Reward: {eval_result['avg_reward']:.2f}")
            print(f"Current Epsilon: {agent.epsilon:.4f}\n")
            
            eval_results.append({
                'episode': episode + 1,
                **eval_result
            })
    
    # Step 5: Final evaluation
    print("\n[5/5] Final evaluation on validation set...")
    val_env = HangmanEnvironment(val_words, max_wrong_guesses=6)
    final_eval = evaluate_agent(val_env, agent, n_episodes=len(val_words))
    
    print("\n" + "=" * 70)
    print("FINAL TRAINING RESULTS")
    print("=" * 70)
    print(f"Success Rate: {final_eval['success_rate']:.2%}")
    print(f"Average Wrong Guesses: {final_eval['avg_wrong_guesses']:.2f}")
    print(f"Average Repeated Guesses: {final_eval['avg_repeated_guesses']:.2f}")
    print(f"Average Reward: {final_eval['avg_reward']:.2f}")
    
    # Calculate score
    n_games = len(val_words)
    score = (final_eval['success_rate'] * 2000) - \
            (final_eval['total_wrong'] * 5) - \
            (final_eval['total_repeated'] * 2)
    print(f"\nEstimated Final Score (on 2000 test words): {score:.2f}")
    print("=" * 70)
    
    print("\nSaving trained agent...")
    agent.save('q_agent.pkl')
    
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)
    
    with open('eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print("\nGenerating training plots...")
    plot_training_progress(training_history)
    plot_evaluation_progress(eval_results)
    
    print("\nTraining complete!")
    return agent, training_history, eval_results

# Run training
if __name__ == "__main__":
    agent, history, eval_results = train_agent(
        corpus_file='corpus.txt',
        n_episodes=10000,
        eval_interval=500
    )
    
    print("\n✓ Training pipeline complete!")
    # ... (rest of the print statements) ...
    print("Generated files:")
    print("  - q_agent.pkl (trained Q-learning agent)")
    print("  - training_history.pkl (training metrics)")
    print("  - eval_results.json (evaluation results)")
    print("  - training_progress.png (training plots)")
    print("  - evaluation_progress.png (evaluation plots)")