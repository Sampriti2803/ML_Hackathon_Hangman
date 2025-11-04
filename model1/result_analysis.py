"""
Hangman ML Hackathon - Part 6: Advanced Result Analysis & Confusion Matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter, defaultdict
import string

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class AdvancedResultAnalyzer:
    """
    Advanced analysis tools for Hangman results
    """
    
    def __init__(self, results_file='test_results.json'):
        """Load test results"""
        print("Loading test results...")
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        print(f"Loaded results for {self.results['total_games']} games")
        self.game_details = self.results['game_details']
    
    def create_letter_confusion_matrix(self):
        """
        Create confusion matrix for letter predictions
        Shows which letters were predicted when target letter was needed
        """
        print("\nGenerating Letter Confusion Matrix...")
        
        alphabet = list(string.ascii_lowercase)
        confusion = np.zeros((26, 26))
        
        # Analyze each game
        for game in self.game_details:
            word = game['word']
            guesses = game['guesses_made']
            
            # Track what was guessed vs what was needed
            target_letters = set(word)
            
            for guess in guesses:
                guess_idx = ord(guess) - ord('a')
                
                # If guess was wrong, mark confusion with all target letters
                if guess not in word:
                    for target in target_letters:
                        target_idx = ord(target) - ord('a')
                        confusion[target_idx][guess_idx] += 1
        
        # Create DataFrame for better visualization
        confusion_df = pd.DataFrame(
            confusion,
            index=alphabet,
            columns=alphabet
        )
        
        # Plot heatmap
        plt.figure(figsize=(16, 14))
        sns.heatmap(confusion_df, annot=False, cmap='YlOrRd', 
                   cbar_kws={'label': 'Frequency'})
        plt.title('Letter Confusion Matrix\n(Row: Target Letter Needed, Column: Letter Guessed)', 
                 fontsize=14, pad=20)
        plt.xlabel('Guessed Letter', fontsize=12)
        plt.ylabel('Target Letter in Word', fontsize=12)
        plt.tight_layout()
        plt.savefig('letter_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Letter confusion matrix saved!")
        
        # Print most confused pairs
        print("\nTop 10 Most Confused Letter Pairs (Target → Guess):")
        confusion_pairs = []
        for i, target in enumerate(alphabet):
            for j, guess in enumerate(alphabet):
                if i != j and confusion[i][j] > 0:
                    confusion_pairs.append((target, guess, confusion[i][j]))
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        for i, (target, guess, count) in enumerate(confusion_pairs[:10], 1):
            print(f"   {i}. '{target}' → '{guess}': {int(count)} times")
        
        return confusion_df
    
    def analyze_guess_patterns(self):
        """
        Analyze patterns in guessing behavior
        """
        print("\n" + "="*70)
        print("GUESS PATTERN ANALYSIS")
        print("="*70)
        
        # First guess distribution
        first_guesses = Counter([g['guesses_made'][0] for g in self.game_details 
                                if g['guesses_made']])
        
        print("\nTop 10 Most Common First Guesses:")
        for i, (letter, count) in enumerate(first_guesses.most_common(10), 1):
            pct = (count / len(self.game_details)) * 100
            print(f"   {i}. '{letter}': {count} times ({pct:.1f}%)")
        
        # Guess position analysis
        position_analysis = defaultdict(lambda: {'correct': 0, 'wrong': 0})
        
        for game in self.game_details:
            word = game['word']
            for i, guess in enumerate(game['guesses_made'], 1):
                if guess in word:
                    position_analysis[i]['correct'] += 1
                else:
                    position_analysis[i]['wrong'] += 1
        
        # Plot guess accuracy by position
        max_position = max(position_analysis.keys())
        positions = list(range(1, max_position + 1))
        correct = [position_analysis[i]['correct'] for i in positions]
        wrong = [position_analysis[i]['wrong'] for i in positions]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Stacked bar chart
        ax = axes[0]
        ax.bar(positions, correct, label='Correct', alpha=0.7, color='green')
        ax.bar(positions, wrong, bottom=correct, label='Wrong', alpha=0.7, color='red')
        ax.set_xlabel('Guess Position')
        ax.set_ylabel('Frequency')
        ax.set_title('Guess Accuracy by Position')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Accuracy rate
        ax = axes[1]
        total = [c + w for c, w in zip(correct, wrong)]
        accuracy = [c / t if t > 0 else 0 for c, t in zip(correct, total)]
        ax.plot(positions, accuracy, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Guess Position')
        ax.set_ylabel('Accuracy Rate')
        ax.set_title('Guess Accuracy Rate by Position')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('guess_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nGuess pattern analysis saved!")
    
    def analyze_failure_modes(self):
        """
        Deep dive into why the agent fails
        """
        print("\n" + "="*70)
        print("FAILURE MODE ANALYSIS")
        print("="*70)
        
        lost_games = [g for g in self.game_details if not g['won']]
        print(f"\nAnalyzing {len(lost_games)} lost games...")
        
        # Common characteristics of lost games
        lost_lengths = [len(g['word']) for g in lost_games]
        lost_unique = [len(set(g['word'])) for g in lost_games]
        
        # Letter frequency in lost words
        lost_letters = Counter(''.join([g['word'] for g in lost_games]))
        
        print("\n1. Word Length Distribution in Lost Games:")
        length_dist = Counter(lost_lengths)
        for length in sorted(length_dist.keys()):
            count = length_dist[length]
            pct = (count / len(lost_games)) * 100
            print(f"   Length {length}: {count} words ({pct:.1f}%)")
        
        print("\n2. Unique Letters in Lost Words:")
        print(f"   Average: {np.mean(lost_unique):.2f}")
        print(f"   Median: {np.median(lost_unique):.1f}")
        
        print("\n3. Most Common Letters in Lost Words:")
        for i, (letter, count) in enumerate(lost_letters.most_common(10), 1):
            print(f"   {i}. '{letter}': {count} occurrences")
        
        # Uncommon letter analysis
        print("\n4. Difficult Letter Combinations:")
        lost_bigrams = Counter()
        lost_trigrams = Counter()
        
        for game in lost_games:
            word = game['word']
            # Bigrams
            for i in range(len(word) - 1):
                lost_bigrams[word[i:i+2]] += 1
            # Trigrams
            for i in range(len(word) - 2):
                lost_trigrams[word[i:i+3]] += 1
        
        print("   Top challenging bigrams:")
        for i, (bigram, count) in enumerate(lost_bigrams.most_common(5), 1):
            print(f"      {i}. '{bigram}': {count} times")
        
        print("   Top challenging trigrams:")
        for i, (trigram, count) in enumerate(lost_trigrams.most_common(5), 1):
            print(f"      {i}. '{trigram}': {count} times")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Word length distribution
        ax = axes[0, 0]
        ax.hist(lost_lengths, bins=range(min(lost_lengths), max(lost_lengths) + 2),
               edgecolor='black', alpha=0.7, color='red')
        ax.set_xlabel('Word Length')
        ax.set_ylabel('Frequency')
        ax.set_title('Word Length Distribution in Lost Games')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Unique letters distribution
        ax = axes[0, 1]
        ax.hist(lost_unique, bins=range(min(lost_unique), max(lost_unique) + 2),
               edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('Unique Letters')
        ax.set_ylabel('Frequency')
        ax.set_title('Unique Letter Count in Lost Words')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Letter frequency
        ax = axes[1, 0]
        top_letters = lost_letters.most_common(15)
        letters, counts = zip(*top_letters)
        ax.barh(range(len(letters)), counts, edgecolor='black', alpha=0.7)
        ax.set_yticks(range(len(letters)))
        ax.set_yticklabels(letters)
        ax.set_xlabel('Frequency')
        ax.set_title('Most Common Letters in Lost Words')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Wrong guess progression
        ax = axes[1, 1]
        wrong_progression = [g['wrong_guesses'] for g in lost_games]
        ax.hist(wrong_progression, bins=range(0, 8), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Wrong Guesses')
        ax.set_ylabel('Frequency')
        ax.set_title('Wrong Guess Count in Lost Games')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('failure_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nFailure analysis saved!")
    
    def generate_performance_report(self):
        """
        Generate comprehensive performance report
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("="*70)
        
        # Overall statistics
        print("\n📊 OVERALL STATISTICS")
        print("-" * 70)
        print(f"Total Games: {self.results['total_games']}")
        print(f"Wins: {self.results['wins']} ({self.results['success_rate']:.2%})")
        print(f"Losses: {self.results['losses']}")
        print(f"Average Wrong Guesses: {self.results['avg_wrong_guesses']:.3f}")
        print(f"Average Repeated Guesses: {self.results['avg_repeated_guesses']:.3f}")
        print(f"\n🏆 FINAL SCORE: {self.results['final_score']:.2f}")
        
        # Performance breakdown
        won_games = [g for g in self.game_details if g['won']]
        lost_games = [g for g in self.game_details if not g['won']]
        
        print("\n✅ WON GAMES ANALYSIS")
        print("-" * 70)
        if won_games:
            won_wrong = [g['wrong_guesses'] for g in won_games]
            won_total = [g['total_guesses'] for g in won_games]
            print(f"Average wrong guesses: {np.mean(won_wrong):.3f}")
            print(f"Average total guesses: {np.mean(won_total):.3f}")
            print(f"Efficiency rate: {(1 - np.mean(won_wrong)/np.mean(won_total)):.2%}")
        
        print("\n❌ LOST GAMES ANALYSIS")
        print("-" * 70)
        if lost_games:
            lost_wrong = [g['wrong_guesses'] for g in lost_games]
            lost_total = [g['total_guesses'] for g in lost_games]
            print(f"Average wrong guesses: {np.mean(lost_wrong):.3f}")
            print(f"Average total guesses: {np.mean(lost_total):.3f}")
        
        # Comparative metrics
        print("\n📈 COMPARATIVE METRICS")
        print("-" * 70)
        
        # Calculate percentiles
        all_wrong = [g['wrong_guesses'] for g in self.game_details]
        print(f"Wrong Guesses Distribution:")
        print(f"  Min: {min(all_wrong)}")
        print(f"  25th percentile: {np.percentile(all_wrong, 25):.1f}")
        print(f"  Median: {np.median(all_wrong):.1f}")
        print(f"  75th percentile: {np.percentile(all_wrong, 75):.1f}")
        print(f"  Max: {max(all_wrong)}")
        
        # Score breakdown
        print(f"\n💰 SCORE BREAKDOWN")
        print("-" * 70)
        win_contribution = self.results['success_rate'] * self.results['total_games'] * 2
        wrong_penalty = self.results['total_wrong_guesses'] * 5
        repeated_penalty = self.results['total_repeated_guesses'] * 2
        
        print(f"Win contribution: +{win_contribution:.2f}")
        print(f"Wrong guess penalty: -{wrong_penalty:.2f}")
        print(f"Repeated guess penalty: -{repeated_penalty:.2f}")
        print(f"{'='*70}")
        print(f"Final Score: {self.results['final_score']:.2f}")
        
        return {
            'overall': self.results,
            'won_analysis': {
                'avg_wrong': np.mean(won_wrong) if won_games else 0,
                'avg_total': np.mean(won_total) if won_games else 0
            },
            'lost_analysis': {
                'avg_wrong': np.mean(lost_wrong) if lost_games else 0,
                'avg_total': np.mean(lost_total) if lost_games else 0
            }
        }
    
    def create_summary_dashboard(self):
        """
        Create a comprehensive summary dashboard
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Win/Loss Pie Chart
        ax1 = fig.add_subplot(gs[0, 0])
        sizes = [self.results['wins'], self.results['losses']]
        labels = ['Wins', 'Losses']
        colors = ['#90EE90', '#FFB6C6']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Win/Loss Ratio')
        
        # 2. Wrong Guesses Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        wrong_guesses = [g['wrong_guesses'] for g in self.game_details]
        ax2.hist(wrong_guesses, bins=range(0, max(wrong_guesses) + 2),
                edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Wrong Guesses')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Wrong Guesses Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Score Components
        ax3 = fig.add_subplot(gs[0, 2])
        win_contrib = self.results['success_rate'] * self.results['total_games'] * 2
        wrong_penalty = self.results['total_wrong_guesses'] * 5
        repeated_penalty = self.results['total_repeated_guesses'] * 2
        
        components = ['Wins', 'Wrong\nGuesses', 'Repeated\nGuesses', 'Final']
        values = [win_contrib, -wrong_penalty, -repeated_penalty, self.results['final_score']]
        colors_bar = ['green', 'red', 'orange', 'blue']
        ax3.bar(components, values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Score')
        ax3.set_title('Score Components')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Performance by Word Length
        ax4 = fig.add_subplot(gs[1, :])
        length_stats = {}
        for game in self.game_details:
            length = len(game['word'])
            if length not in length_stats:
                length_stats[length] = {'wins': 0, 'total': 0}
            length_stats[length]['total'] += 1
            if game['won']:
                length_stats[length]['wins'] += 1
        
        lengths = sorted(length_stats.keys())
        success_rates = [length_stats[l]['wins'] / length_stats[l]['total'] for l in lengths]
        word_counts = [length_stats[l]['total'] for l in lengths]
        
        ax4_twin = ax4.twinx()
        line = ax4.plot(lengths, success_rates, marker='o', color='blue',
                       linewidth=2, markersize=8, label='Success Rate')
        bars = ax4_twin.bar(lengths, word_counts, alpha=0.3, color='gray', label='Word Count')
        
        ax4.set_xlabel('Word Length')
        ax4.set_ylabel('Success Rate', color='blue')
        ax4_twin.set_ylabel('Number of Words', color='gray')
        ax4.set_title('Performance by Word Length')
        ax4.set_ylim([0, 1.1])
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4_twin.tick_params(axis='y', labelcolor='gray')
        
        lines_1, labels_1 = ax4.get_legend_handles_labels()
        lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left')
        
        # 5. Efficiency Metrics
        ax5 = fig.add_subplot(gs[2, :])
        metrics = ['Success\nRate', 'Avg Wrong\nGuesses', 'Avg Repeated\nGuesses']
        values_metrics = [
            self.results['success_rate'] * 100,
            self.results['avg_wrong_guesses'],
            self.results['avg_repeated_guesses']
        ]
        colors_metrics = ['green', 'red', 'orange']
        
        bars = ax5.bar(metrics, values_metrics, color=colors_metrics, alpha=0.7, edgecolor='black')
        ax5.set_ylabel('Value')
        ax5.set_title('Key Performance Metrics')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nPerformance dashboard saved!")


def run_complete_analysis(results_file='test_results.json'):
    """
    Run complete analysis pipeline
    """
    print("="*70)
    print("ADVANCED RESULT ANALYSIS")
    print("="*70)
    
    analyzer = AdvancedResultAnalyzer(results_file)
    
    # 1. Generate performance report
    report = analyzer.generate_performance_report()
    
    # 2. Create confusion matrix
    confusion_df = analyzer.create_letter_confusion_matrix()
    
    # 3. Analyze guess patterns
    analyzer.analyze_guess_patterns()
    
    # 4. Analyze failure modes
    analyzer.analyze_failure_modes()
    
    # 5. Create summary dashboard
    analyzer.create_summary_dashboard()
    
    print("\n✓ Complete analysis finished!")
    print("\nGenerated files:")
    print("  - letter_confusion_matrix.png")
    print("  - guess_patterns.png")
    print("  - failure_analysis.png")
    print("  - performance_dashboard.png")
    
    return analyzer, report


if __name__ == "__main__":
    analyzer, report = run_complete_analysis('test_results.json')