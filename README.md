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
- Keep the same observable selected while moving between transformer blocks.
- View residuals against a baseline run.
- Zoom into curves, select step ranges, and open charts in fullscreen mode.
- Read the mathematical definition and plain-language description of each observable.
- Publish detailed discoveries with zoomed curve evidence and discussion.
- Read an introduction to the library, with a three-regime Block 1 attention-entropy finding as a worked example.
- Inspect each run's model, optimizer, learning-rate, and batching configuration.

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
   Use **Inspect training config** on a run card to review its setup first.
3. Select a transformer layer and module.
4. Choose an observable to display its curve.
5. Enable layer or setup comparison when compatible data is available.
6. Open a chart in fullscreen mode to inspect individual steps and public notes.

The **Formulas** page provides a searchable catalog of observable definitions.
The **Announcements** and **Suggestions** pages provide project updates and a
public channel for feedback.

## Public Notes, Discoveries, and Suggestions

Visitors can post notes on charts and add replies with an optional public display
name. Leaving the name blank keeps the post anonymous. Notes, names, and replies
are visible to everyone, so please do not include private or sensitive information.
The public **Navigate** control on a note or reply focuses the chart on its
attached step or step range.

The separate **Discoveries** page is intended for longer research findings. A
discovery can cite up to six regions from any recorded setup, layer or block,
observable, and step range. All cited regions are overlaid in one zoomed chart,
with a labeled key and a direct Explorer link for each curve. Discovery charts
support curve visibility, linear/logarithmic scales, step/τ axes, zoom, reset,
and an expanded view. Authors can hold nested public discussions with optional
names.

Discovery records and replies use the same Firebase Authentication and Firestore
backend as Notes. Deploy `firestore.rules` (or use the included `firebase.json`)
before enabling Discoveries in production. Curators can delete discoveries and
any reply.

## About the Data

The site displays previously recorded training runs; it does not run or modify a
model in the browser. Curves may differ between runs because of changes in
training configuration, initialization, optimization, or instrumentation.

This project is intended as a research and educational tool for studying neural
network training dynamics.
