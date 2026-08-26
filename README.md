# REASONX

<p align="justify">
REASONX provides declarative, interactive explanations for decision trees, which can be the ML models under analysis, or global/local surrogate models of any black-box model (i.e., model-agnostic). Additionally, it can incorporate background or common sense knowledge in the form of linear constraints. Explanations are provided as factual and contrastive rules, and in the form of closest contrastive examples via MILP optimization.
</p>

Here, we provide both the Prolog program, as well as a Python layer to access REASONX.

More information can be found in our papers:

1) "Declarative Reasoning on Explanations Using Constraint Logic Programming" by Laura State, Salvatore Ruggieri and Franco Turini

    Accepted at [JELIA 2023](https://jelia2023.inf.tu-dresden.de/)

    A [paper](http://export.arxiv.org/abs/2309.00422) on the theoretical background of REASONX (Constraint Logic Programming, Prolog, Meta-Interpreter).

2) "Reason to explain: Interactive contrastive explanations (REASONX)" by Laura State, Salvatore Ruggieri and Franco Turini

    Accepted at [xAI 2023](https://xaiworldconference.com/)

    An interdisciplinary [paper](https://arxiv.org/abs/2305.18143), demonstrating main capabilites of REASONX via a synthetic example, and based on the Adult Income Dataset.

## Extension of the code base (notebooks/extension)

- Reasoning over time

- Reasoning over models
  
- Diversity optimization

- Detecting biases

- Evaluation: folder containing a couple of files, updated on 2024-10-01

- Runtimes: folder containing a couple of files, updated on 2024-10-01

This extension includes updates of the basic files of REASONX (post.pl/reasonx.py).
