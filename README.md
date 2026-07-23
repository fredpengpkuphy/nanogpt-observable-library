# nanoGPT Observable Explorer

nanoGPT Observable Explorer is a public, interactive view of what happens inside
a GPT-style language model during training. It connects training curves to the
model architecture, making it easier to inspect how individual layers, modules,
weights, gradients, activations, and attention patterns evolve over time.

## What You Can Explore

- Compare training and validation loss across recorded runs.
- Navigate the model from embeddings through transformer blocks to the output head.
- Inspect observables for individual modules and training steps.
- Compare the same observable across layers or experimental setups.
- View residuals against a baseline run.
- Zoom into curves, select step ranges, and open charts in fullscreen mode.
- Read the mathematical definition and plain-language description of each observable.

## Observable Families

The explorer includes measurements derived from:

- Model weights
- Gradients
- Parameter updates
- Layer activations and pre-activations
- Attention entropy and attention-sink behavior
- GELU activation patterns and activation outliers
- Output logits
- Training and validation loss

Available measurements depend on the data recorded for each training run.

## Using the Explorer

1. Select **Start Exploration** on the home page.
2. Choose a recorded training run.
3. Select a transformer layer and module.
4. Choose an observable to display its curve.
5. Enable layer or setup comparison when compatible data is available.
6. Open a chart in fullscreen mode to inspect individual steps and public notes.

The **Formulas** page provides a searchable catalog of observable definitions.
The **Announcements** and **Suggestions** pages provide project updates and a
public channel for feedback.

## Public Notes and Suggestions

Visitors can post notes on charts and submit suggestions without displaying a
public username. Notes and replies are visible to everyone, so please do not
include private, sensitive, or identifying information.

## About the Data

The site displays previously recorded training runs; it does not run or modify a
model in the browser. Curves may differ between runs because of changes in
training configuration, initialization, optimization, or instrumentation.

This project is intended as a research and educational tool for studying neural
network training dynamics.
