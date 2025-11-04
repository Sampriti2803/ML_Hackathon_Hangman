"""
Hangman ML Hackathon - Master Pipeline
Run this script to execute the entire pipeline from start to finish
"""

import os
import sys
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def check_file_exists(filepath, file_description):
    """Check if required file exists"""
    if not os.path.exists(filepath):
        print(f"❌ Error: {file_description} not found at '{filepath}'")
        print(f"   Please ensure the file exists before running this pipeline.")
        return False
    print(f"✓ Found {file_description}: {filepath}")
    return True

def run_pipeline(corpus_file='corpus.txt', test_file='test_words.txt', 
                n_training_episodes=10000):
    """
    Run complete Hangman ML pipeline
    
    Args:
        corpus_file: path to training corpus
        test_file: path to test words
        n_training_episodes: number of RL training episodes
    """
    
    start_time = datetime.now()
    
    print_header("HANGMAN ML HACKATHON - COMPLETE PIPELINE")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Corpus file: {corpus_file}")
    print(f"  - Test file: {test_file}")
    print(f"  - Training episodes: {n_training_episodes:,}")
    print()
    
    # Check prerequisites
    print_header("STEP 0: Checking Prerequisites")
    
    if not check_file_exists(corpus_file, "Training corpus"):
        return False
    
    if not check_file_exists(test_file, "Test words"):
        return False
    
    print("\n✓ All prerequisites satisfied!")
    
    # Step 1: Data Exploration
    print_header("STEP 1: Data Exploration & Preprocessing")
    print("Running exploratory data analysis...")
    
    try:
        # Import or execute data exploration code
        exec(open('1_data_exploration.py').read())
        print("\n✓ Data exploration complete!")
        print("   Generated: word_length_distribution.png, letter_frequency.png, etc.")
    except Exception as e:
        print(f"❌ Error in data exploration: {e}")
        print("   Continuing with next step...")
    
    # Step 2: HMM Training
    print_header("STEP 2: Training Hidden Markov Model")
    print("Training HMM on corpus...")
    
    try:
        from hmm_model import HangmanHMM
        
        hmm = HangmanHMM()
        hmm.train(corpus_file)
        hmm.save('hmm_model.pkl')
        
        print("\n✓ HMM training complete!")
        print("   Saved: hmm_model.pkl")
    except Exception as e:
        print(f"❌ Error in HMM training: {e}")
        return False
    
    # Step 3: RL Agent Training
    print_header("STEP 3: Training Reinforcement Learning Agent")
    print(f"Training Q-Learning agent for {n_training_episodes:,} episodes...")
    print("This will take some time...")
    
    try:
        from rl_training import train_agent
        
        agent, history, eval_results = train_agent(
            corpus_file=corpus_file,
            n_episodes=n_training_episodes,
            eval_interval=500
        )
        
        print("\n✓ RL agent training complete!")
        print("   Saved: q_agent.pkl, training_history.pkl")
        print("   Generated: training_progress.png, evaluation_progress.png")
    except Exception as e:
        print(f"❌ Error in RL training: {e}")
        return False
    
    # Step 4: Testing
    print_header("STEP 4: Testing on Test Dataset")
    print(f"Running tests on {test_file}...")
    
    try:
        from testing import run_complete_testing
        
        results = run_complete_testing(test_file)
        
        print("\n✓ Testing complete!")
        print("   Saved: test_results.json")
        print("   Generated: test_analysis.png")
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        return False
    
    # Step 5: Advanced Analysis
    print_header("STEP 5: Advanced Result Analysis")
    print("Generating comprehensive analysis and visualizations...")
    
    try:
        from result_analysis import run_complete_analysis
        
        analyzer, report = run_complete_analysis('test_results.json')
        
        print("\n✓ Analysis complete!")
        print("   Generated: confusion_matrix.png, performance_dashboard.png, etc.")
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        print("   Basic results are still available.")
    
    # Pipeline Complete
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("PIPELINE COMPLETE!")
    
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    
    print("\n📁 Generated Files:")
    print("   Models:")
    print("     - hmm_model.pkl (Hidden Markov Model)")
    print("     - q_agent.pkl (Trained Q-Learning Agent)")
    print("\n   Results:")
    print("     - test_results.json (Detailed test results)")
    print("     - training_history.pkl (Training metrics)")
    print("     - eval_results.json (Evaluation checkpoints)")
    print("     - corpus_statistics.json (Data statistics)")
    print("\n   Visualizations:")
    print("     - word_length_distribution.png")
    print("     - letter_frequency.png")
    print("     - training_progress.png")
    print("     - evaluation_progress.png")
    print("     - test_analysis.png")
    print("     - letter_confusion_matrix.png")
    print("     - guess_patterns.png")
    print("     - failure_analysis.png")
    print("     - performance_dashboard.png")
    
    print("\n🏆 FINAL RESULTS:")
    try:
        import json
        with open('test_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"   Total Games: {results['total_games']}")
        print(f"   Success Rate: {results['success_rate']:.2%}")
        print(f"   Avg Wrong Guesses: {results['avg_wrong_guesses']:.3f}")
        print(f"   Avg Repeated Guesses: {results['avg_repeated_guesses']:.3f}")
        print(f"   FINAL SCORE: {results['final_score']:.2f}")
    except:
        print("   (Results file not found)")
    
    print("\n✨ Thank you for using the Hangman ML Pipeline!")
    print("   For questions or issues, please refer to the documentation.")
    
    return True


def quick_test(word='apple'):
    """
    Quick test of trained model on a single word
    """
    print_header(f"QUICK TEST: Testing on word '{word}'")
    
    try:
        from hmm_model import HangmanHMM
        from rl_training import HangmanEnvironment, QLearningAgent
        
        # Load models
        print("Loading models...")
        hmm = HangmanHMM()
        hmm.load('hmm_model.pkl')
        
        agent = QLearningAgent(hmm)
        agent.load('q_agent.pkl')
        agent.epsilon = 0.0  # No exploration
        
        # Play game
        env = HangmanEnvironment([word], max_wrong_guesses=6)
        state = env.reset(word)
        
        print(f"\nTarget word: {'_' * len(word)}")
        print(f"Lives: {env.max_wrong_guesses}")
        print("-" * 50)
        
        guess_num = 1
        while not env.game_over:
            action = agent.choose_action(state, env, training=False)
            
            if action is None:
                break
            
            next_state, reward, done, info = env.step(action)
            
            print(f"Guess {guess_num}: '{action}' → {next_state['masked_word']} " +
                  f"(Lives: {next_state['lives_left']})")
            
            state = next_state
            guess_num += 1
        
        print("-" * 50)
        if env.won:
            print(f"✓ WON! Word: {word}")
        else:
            print(f"✗ LOST! Word: {word}")
        print(f"Wrong guesses: {env.wrong_guesses}")
        print(f"Repeated guesses: {env.repeated_guesses}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure models are trained first (run with '--full' flag)")


def show_help():
    """Show usage instructions"""
    print("""
Hangman ML Hackathon - Master Pipeline

USAGE:
    python master_pipeline.py [OPTIONS]

OPTIONS:
    --full              Run complete pipeline (data exploration → training → testing)
    --test [word]       Quick test on a single word (requires trained models)
    --episodes N        Set number of training episodes (default: 10000)
    --corpus FILE       Specify corpus file (default: corpus.txt)
    --testset FILE      Specify test file (default: test_words.txt)
    --help             Show this help message

EXAMPLES:
    # Run complete pipeline with default settings
    python master_pipeline.py --full

    # Run with custom number of training episodes
    python master_pipeline.py --full --episodes 20000

    # Quick test on a specific word
    python master_pipeline.py --test hangman

    # Use custom corpus file
    python master_pipeline.py --full --corpus my_corpus.txt

PIPELINE STEPS:
    1. Data Exploration & Preprocessing
    2. Hidden Markov Model Training
    3. Reinforcement Learning Agent Training
    4. Testing on Test Dataset
    5. Advanced Result Analysis

OUTPUT:
    The pipeline generates:
    - Trained models (.pkl files)
    - Result files (.json files)
    - Visualization plots (.png files)
    
For more information, refer to the documentation.
""")


if __name__ == "__main__":
    if len(sys.argv) == 1 or '--help' in sys.argv:
        show_help()
    
    elif '--test' in sys.argv:
        # Quick test mode
        idx = sys.argv.index('--test')
        word = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'apple'
        quick_test(word)
    
    elif '--full' in sys.argv:
        # Full pipeline mode
        corpus_file = 'corpus.txt'
        test_file = 'test_words.txt'
        n_episodes = 10000
        
        # Parse custom arguments
        if '--corpus' in sys.argv:
            idx = sys.argv.index('--corpus')
            corpus_file = sys.argv[idx + 1]
        
        if '--testset' in sys.argv:
            idx = sys.argv.index('--testset')
            test_file = sys.argv[idx + 1]
        
        if '--episodes' in sys.argv:
            idx = sys.argv.index('--episodes')
            n_episodes = int(sys.argv[idx + 1])
        
        # Run pipeline
        success = run_pipeline(corpus_file, test_file, n_episodes)
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Pipeline completed with errors.")
            sys.exit(1)
    
    else:
        print("❌ Invalid arguments. Use --help for usage instructions.")
        sys.exit(1)