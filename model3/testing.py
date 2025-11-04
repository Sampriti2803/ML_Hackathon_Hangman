"""
Hangman ML Hackathon - Part 5: Testing and Final Evaluation
"""

import numpy as np
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import required classes (ensure they're available)
from hmm_model import HangmanHMM
from rl_agent import HangmanEnvironment, QLearningAgent


class HangmanTester:
    """
    Complete testing and evaluation framework
    """
    
    def __init__(self, hmm_path='hmm_model.pkl', agent_path='q_agent.pkl'):
        """Load trained models"""
        print("Loading trained models...")
        
        # Load HMM
        self.hmm = HangmanHMM()
        self.hmm.load(hmm_path)
        
        # Load RL agent
        self.agent = QLearningAgent(self.hmm)
        self.agent.load(agent_path)
        # Set epsilon to minimum for testing (no exploration)
        self.agent.epsilon = 0.0
        
        print("Models loaded successfully!")
    
    def test_on_dataset(self, test_file='test_words.txt', max_wrong=6):
        """
        Run complete test on test dataset
        
        Args:
            test_file: path to file containing test words
            max_wrong: maximum wrong guesses allowed (default: 6)
            
        Returns:
            dict with detailed results
        """
        print(f"\n{'='*70}")
        print("TESTING ON DATASET")
        print(f"{'='*70}")
        
        # Load test words
        with open(test_file, 'r') as f:
            test_words = [line.strip().lower() for line in f if line.strip()]
        
        print(f"Test set size: {len(test_words)} words")
        print(f"Max wrong guesses: {max_wrong}\n")
        
        # Initialize results storage
        results = {
            'total_games': len(test_words),
            'wins': 0,
            'losses': 0,
            'total_wrong_guesses': 0,
            'total_repeated_guesses': 0,
            'game_details': []
        }
        
        # Test each word
        print("Running tests...")
        for word in tqdm(test_words):
            env = HangmanEnvironment([word], max_wrong_guesses=max_wrong)
            game_result = self._play_single_game(env, word)
            
            # Update results
            if game_result['won']:
                results['wins'] += 1
            else:
                results['losses'] += 1
            
            results['total_wrong_guesses'] += game_result['wrong_guesses']
            results['total_repeated_guesses'] += game_result['repeated_guesses']
            results['game_details'].append(game_result)
        
        # Calculate metrics
        results['success_rate'] = results['wins'] / results['total_games']
        results['avg_wrong_guesses'] = results['total_wrong_guesses'] / results['total_games']
        results['avg_repeated_guesses'] = results['total_repeated_guesses'] / results['total_games']
        
        # Calculate final score using competition formula
        results['final_score'] = (
            (results['success_rate'] * results['total_games']) -
            (results['total_wrong_guesses'] * 5) -
            (results['total_repeated_guesses'] * 2)
        )
        
        return results
    
    def _play_single_game(self, env, word):
        """Play a single game and record details"""
        state = env.reset(word)
        guesses_made = []
        
        while not env.game_over:
            # choose_action now returns (strategy_index, letter_guess)
            # We only need the letter_guess for testing.
            strategy_index, letter_guess = self.agent.choose_action(state, env, training=False)
            
            if letter_guess is None:
                break
            
            guesses_made.append(letter_guess)
            next_state, reward, done, info = env.step(letter_guess) # Pass the letter, not the tuple
            state = next_state
        
        return {
            'word': word,
            'won': env.won,
            'wrong_guesses': env.wrong_guesses,
            'repeated_guesses': env.repeated_guesses,
            'guesses_made': guesses_made,
            'total_guesses': len(guesses_made)
        }
    
    def print_results(self, results):
        """Print formatted results"""
        print(f"\n{'='*70}")
        print("FINAL TEST RESULTS")
        print(f"{'='*70}")
        print(f"Total Games: {results['total_games']}")
        print(f"Wins: {results['wins']}")
        print(f"Losses: {results['losses']}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"\nAverage Wrong Guesses: {results['avg_wrong_guesses']:.3f}")
        print(f"Average Repeated Guesses: {results['avg_repeated_guesses']:.3f}")
        print(f"Total Wrong Guesses: {results['total_wrong_guesses']}")
        print(f"Total Repeated Guesses: {results['total_repeated_guesses']}")
        print(f"\n{'='*70}")
        print(f"FINAL SCORE: {results['final_score']:.2f}")
        print(f"{'='*70}")
    
    def save_results(self, results, output_file='test_results.json'):
        """Save results to JSON file"""
        # Convert results to JSON-serializable format
        json_results = {
            'total_games': results['total_games'],
            'wins': results['wins'],
            'losses': results['losses'],
            'success_rate': results['success_rate'],
            'avg_wrong_guesses': results['avg_wrong_guesses'],
            'avg_repeated_guesses': results['avg_repeated_guesses'],
            'total_wrong_guesses': results['total_wrong_guesses'],
            'total_repeated_guesses': results['total_repeated_guesses'],
            'final_score': results['final_score'],
            'game_details': results['game_details']
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    def analyze_results(self, results):
        """Generate detailed analysis and visualizations"""
        print(f"\n{'='*70}")
        print("DETAILED ANALYSIS")
        print(f"{'='*70}")
        
        game_details = results['game_details']
        
        # Word length analysis
        print("\n1. Performance by Word Length:")
        length_stats = {}
        for game in game_details:
            length = len(game['word'])
            if length not in length_stats:
                length_stats[length] = {'wins': 0, 'total': 0, 'wrong': 0}
            
            length_stats[length]['total'] += 1
            if game['won']:
                length_stats[length]['wins'] += 1
            length_stats[length]['wrong'] += game['wrong_guesses']
        
        for length in sorted(length_stats.keys()):
            stats = length_stats[length]
            success_rate = stats['wins'] / stats['total']
            avg_wrong = stats['wrong'] / stats['total']
            print(f"   Length {length}: {success_rate:.2%} success, "
                  f"{avg_wrong:.2f} avg wrong guesses ({stats['total']} words)")
        
        # Worst performing words
        print("\n2. Most Difficult Words (Lost Games):")
        lost_games = [g for g in game_details if not g['won']]
        lost_games.sort(key=lambda x: x['wrong_guesses'], reverse=True)
        
        for i, game in enumerate(lost_games[:10], 1):
            print(f"   {i}. '{game['word']}' - {game['wrong_guesses']} wrong guesses")
        
        # Best performing words
        print("\n3. Easiest Words (Won with Fewest Guesses):")
        won_games = [g for g in game_details if g['won']]
        won_games.sort(key=lambda x: x['total_guesses'])
        
        for i, game in enumerate(won_games[:10], 1):
            print(f"   {i}. '{game['word']}' - {game['total_guesses']} total guesses, "
                  f"{game['wrong_guesses']} wrong")
        
        # Generate visualizations
        self._plot_analysis(results, length_stats)
    
    def _plot_analysis(self, results, length_stats):
        """Generate analysis plots"""
        game_details = results['game_details']
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Success rate by word length
        ax1 = plt.subplot(2, 3, 1)
        lengths = sorted(length_stats.keys())
        success_rates = [length_stats[l]['wins'] / length_stats[l]['total'] for l in lengths]
        ax1.bar(lengths, success_rates, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Word Length')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate by Word Length')
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Average wrong guesses by word length
        ax2 = plt.subplot(2, 3, 2)
        avg_wrong = [length_stats[l]['wrong'] / length_stats[l]['total'] for l in lengths]
        ax2.bar(lengths, avg_wrong, color='red', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Word Length')
        ax2.set_ylabel('Avg Wrong Guesses')
        ax2.set_title('Wrong Guesses by Word Length')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Distribution of wrong guesses
        ax3 = plt.subplot(2, 3, 3)
        wrong_guesses = [g['wrong_guesses'] for g in game_details]
        ax3.hist(wrong_guesses, bins=range(0, max(wrong_guesses) + 2), 
                edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Wrong Guesses')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Wrong Guesses')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Distribution of total guesses
        ax4 = plt.subplot(2, 3, 4)
        total_guesses = [g['total_guesses'] for g in game_details]
        ax4.hist(total_guesses, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax4.set_xlabel('Total Guesses')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Total Guesses per Game')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Win/Loss pie chart
        ax5 = plt.subplot(2, 3, 5)
        sizes = [results['wins'], results['losses']]
        labels = [f"Wins\n({results['wins']})", f"Losses\n({results['losses']})"]
        colors = ['#90EE90', '#FFB6C6']
        ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
               startangle=90, textprops={'fontsize': 11})
        ax5.set_title('Win/Loss Distribution')
        
        # 6. Word count by length
        ax6 = plt.subplot(2, 3, 6)
        word_counts = [length_stats[l]['total'] for l in lengths]
        ax6.bar(lengths, word_counts, edgecolor='black', alpha=0.7, color='purple')
        ax6.set_xlabel('Word Length')
        ax6.set_ylabel('Number of Words')
        ax6.set_title('Test Set Distribution by Length')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('test_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nAnalysis plots saved to 'test_analysis.png'")


def run_complete_testing(test_file='test_words.txt'):
    """
    Run complete testing pipeline
    """
    # Initialize tester
    tester = HangmanTester(
        hmm_path='hmm_model.pkl',
        agent_path='q_agent.pkl'
    )
    
    # Run tests
    results = tester.test_on_dataset(test_file, max_wrong=6)
    
    # Print results
    tester.print_results(results)
    
    # Save results
    tester.save_results(results)
    
    # Analyze results
    tester.analyze_results(results)
    
    return results


# Interactive demo
def interactive_demo():
    """
    Interactive demo where user can test specific words
    """
    print("\n" + "="*70)
    print("INTERACTIVE HANGMAN DEMO")
    print("="*70)
    
    # Load models
    tester = HangmanTester()
    
    while True:
        word = input("\nEnter a word to test (or 'quit' to exit): ").strip().lower()
        
        if word == 'quit':
            break
        
        if not word or not all(c.isalpha() for c in word):
            print("Please enter a valid word with only letters.")
            continue
        
        # Play game
        env = HangmanEnvironment([word], max_wrong_guesses=6)
        state = env.reset(word)
        
        print(f"\nTarget word: {'_' * len(word)}")
        print(f"Lives: {env.max_wrong_guesses}")
        print("-" * 50)
        
        guess_num = 1
        while not env.game_over:
            action = tester.agent.choose_action(state, env, training=False)
            
            if action is None:
                break
            
            next_state, reward, done, info = env.step(action)
            
            print(f"\nGuess {guess_num}: '{action}'")
            print(f"Current: {state['masked_word']} → {next_state['masked_word']}")
            print(f"Lives left: {next_state['lives_left']}")
            
            if info['repeated']:
                print("⚠ Repeated guess!")
            elif action in word:
                print("✓ Correct!")
            else:
                print("✗ Wrong!")
            
            state = next_state
            guess_num += 1
        
        print("\n" + "-" * 50)
        if env.won:
            print(f"✓ WON! Word: {word}")
        else:
            print(f"✗ LOST! Word: {word}")
        print(f"Wrong guesses: {env.wrong_guesses}")
        print(f"Repeated guesses: {env.repeated_guesses}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # Run interactive demo
        interactive_demo()
    else:
        # Run complete testing
        results = run_complete_testing('test_words.txt')
        
        print("\n✓ Testing complete!")
        print("Generated files:")
        print("  - test_results.json (detailed results)")
        print("  - test_analysis.png (analysis plots)")