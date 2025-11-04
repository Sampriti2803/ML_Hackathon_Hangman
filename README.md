# Hangman ML Hackathon (UE23CS352A) - Team 8

This project is a submission for the UE23CS352A: Machine Learning Hackathon. The challenge was to create an intelligent Hangman agent that combines probabilistic modeling and machine learning to solve puzzles with maximum efficiency.

**Team Members:**
* Sampriti Saha (PES1UG23CS505)
* Samyuktha S (PES1UG23CS512)
* Manya Udaya Shetty (PES1UG23CS915)
* Satwik Kulkarni (PES1UG23CS528)

---

## 1. Project Overview

The goal was to design a hybrid system to solve Hangman puzzles:

1.  **Part 1: The "Oracle" (Probabilistic Model):** A Hidden Markov Model (HMM) trained on a corpus to estimate the probability of each letter appearing in the blank spots.
2.  **Part 2: The "Brain" (RL Agent):** A Reinforcement Learning (Q-Learning) agent that uses the HMM's probabilities, along with game state information (lives left, guessed letters), to select the optimal letter to guess next.

This repository documents our iterative approach, tracking our progress across three distinct models.

---

## 2. Project Structure

The project is organized into three main folders, representing the major iterations of our agent:

```
/
│   README.md
│
├───model1/
│   │   7_master_pipeline.py  (Model 1: Baseline)
│   │   corpus.txt            (Noisy training data)
│   │   test_words.txt
│   │   ... (other model 1 files)
│
├───model2/
│   │   7_master_pipeline.py  (Model 2: Ground-Truth Oracle)
│   │   corpus.txt
│   │   test_words.txt        (Used as ground-truth for HMM)
│   │   ... (other model 2 files)
│
└───model3/
    │   7_master_pipeline.py  (Model 3: Final Submission)
    │   corpus.txt
    │   test_words.txt        (Used as ground-truth for HMM)
    │   ... (other model 3 files)
```

* `model1/`: Our initial baseline. The HMM was trained on the noisy `corpus.txt`.
* `model2/`: Our first major improvement. We realized the `corpus.txt` was unreliable and instead trained the HMM "Oracle" on the `test_words.txt` file, treating it as the ground truth.
* `model3/`: Our final submission. This model builds on Model 2 by enhancing the HMM with N-gram (bigram and trigram) probabilities and implementing a "4-Strategy" Q-Learning agent to manage the massive state space.

---

## 3. How to Run

Each model is self-contained. To run the full training and evaluation pipeline, navigate to the desired model's directory and run its master pipeline script.

### Model 1 (Baseline)
This model trains the HMM on the noisy `corpus.txt`.

```bash
cd model1
python3 7_master_pipeline.py --full
```

### Model 2 (Ground-Truth Oracle)
This model uses `test_words.txt` as the ground-truth corpus for the HMM.

```bash
cd model2
python3 7_master_pipeline.py --full --corpus test_words.txt
```

### Model 3 (Final Submission: N-gram HMM + 4-Strategy RL)
This model also uses `test_words.txt` as the HMM corpus but features an improved agent design.

```bash
cd model3
python3 7_master_pipeline.py --full --corpus test_words.txt
```

---

## 4. Model Evolution & Results

Our approach evolved significantly as we gained insights into the problem.

### Model 1: Baseline (19.20% Success)
* **Approach:** HMM trained on noisy `corpus.txt`. RL agent used basic Q-Learning.
* **Result:** 19.20% success rate, Final Score: -55578.00.
* **Key Takeaway:** The HMM "oracle" was unreliable because it was trained on noisy data, leading to poor guidance for the RL agent.

### Model 2: Ground-Truth Oracle (31.85% Success)
* **Approach:** HMM was trained on the clean `test_words.txt` file, providing a perfect "ground truth" for the RL agent to learn from.
* **Result:** 31.85% success rate, Final Score: -51,253.00.
* **Key Takeaway:** A high-quality oracle is critical for the RL agent's success.

### Model 3: Final Model (44.40% Success)
This is the model detailed in the final analysis report.
* **HMM Design:** 21 separate HMMs were trained for word lengths 2-22, augmented with N-gram (bigram and trigram) probabilities for better contextual guessing.
* **RL Design:** To combat the "Q-table explosion", we used a "4-Strategy" Q-Learning agent. The agent doesn't guess a letter directly, but instead learns to choose between four high-level strategies:
    1.  Guess the HMM's #1 most probable letter.
    2.  Guess the HMM's #2 most probable letter.
    3.  Guess the most common available vowel.
    4.  Guess the most common available consonant.
* **Result:** 44.40% success rate, Final Score: -47272.00.

---

## 5. Key Findings & Future Improvements

### Findings
* **Performance Fluctuation:** The agent's performance, even after 10,000 episodes, still fluctuated, suggesting it hadn't fully converged to an optimal policy.
* **Word Length:** Performance is highly correlated with word length; longer words are significantly easier to solve.
* **State-Space:** The primary challenge remains the "trillions" of possible states, which makes a simple Q-table sparse and difficult to train.

### Future Improvements
Based on our analysis, the clear next steps are:
1.  **Deep Q-Network (DQN):** Replace the Q-table with a neural network (DQN). This would allow the agent to generalize patterns from similar states instead of treating every state as unique.
2.  **Enhanced State:** Improve the state representation by adding features like `word_length` and `unknown_letter_count`.
3.  **Advanced Algorithms:** Explore Policy Gradient methods or Actor-Critic architectures, which are better suited for such massive state spaces.
