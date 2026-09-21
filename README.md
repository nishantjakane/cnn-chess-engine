# CNN Chess Engine

An experimental **CNN-based chess engine** built to explore whether neural networks could learn chess move selection and position evaluation.

> ⚠️ **Experimental / Unsuccessful Project** — The approaches explored here did not produce a strong or practical chess engine.

## What I Tried

### 1. GM Move Prediction

I trained a CNN on chess games played by Grandmasters with the goal of predicting the **next move played by a GM** from a given board position.

The idea was to see whether a neural network could learn patterns in strong human play directly from game data.

### 2. Stockfish Evaluation Prediction

I also experimented with training a CNN to predict **Stockfish's position evaluations** from board positions.

The goal was to approximate the evaluation function of a traditional chess engine using a neural network.

Neither approach produced results that were good enough to build a competitive engine.

## What I Learned

The project was mainly about understanding the challenges involved in applying deep learning to chess, including:

* Representing chess positions for neural networks
* Dataset generation and preprocessing
* CNN architecture design
* Training on large chess datasets
* Move prediction vs. position evaluation
* The difficulty of learning strong chess purely from supervised data

It also made it clear why modern chess engines typically require much more than simply predicting moves or evaluations from a static board position.

## Status

🛑 **Archived / Experimental**

The project is no longer actively developed, but I'm keeping it as a record of an experiment that didn't work out as expected.

---

**Built by [Nishant Jakane](https://github.com/nishantjakane)**

*An experiment in applying deep learning to chess.*
