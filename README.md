# FlyWire LM

Byte-level language model whose recurrent state is processed by the frozen
FlyWire v783 connectome: 139,255 neurons and 2,700,513 directed connections.

The experiment asks a simple question: can a real biological connectome act as
the recurrent core of a language model without changing its internal
connectivity? The FlyWire graph is completely frozen. Only the interfaces that
encode bytes into afferent activity and decode efferent activity are trained.

After training on a small prefix of WikiText-2 Raw, the model already learns
byte-level English structure and produces primitive text-like output.

```text
UTF-8 byte
  -> embedding and input projection
  -> 19,265 afferent neurons
  -> frozen FlyWire connectome
  -> 1,489 efferent neurons
  -> linear decoder
  -> next-byte logits
```

Only the byte embedding, afferent input projection and efferent decoder are
trained. Connectome topology, recurrent weights and dynamics stay fixed.

The fixed recurrent dynamics are:

```text
h_next = (1 - alpha) h + alpha tanh(g A h + input)
```

Here `A` is the frozen FlyWire adjacency matrix and `h` contains the state of
all 139,255 neurons. Language input enters only through afferent neurons, and
the decoder reads only efferent neurons.

The connectome files come from
[FlyWire FAFB v783](https://zenodo.org/records/10676866), and flow annotations
come from the
[FlyWire annotations repository](https://github.com/flyconnectome/flywire_annotations).
WikiText-2 Raw is obtained from
[`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext).

## Current results

After sequential training on the first 4 MiB of WikiText-2 Raw with `K=2`:

```text
Validation loss:    2.2641
Byte perplexity:    9.62
Next-byte accuracy: 35.45%
```

Example autoregressive generation:

```text
= Valkyria Chronicles III =

Nover Cend ond of the marectien on thad the le foment an .
The seromes re four the the cheber and te the the al bastor...
```

This is not fluent language yet. It is evidence that next-byte structure can
be learned while the full internal connectome remains unchanged.

## Project layout

```text
train.py                 training command
flylm/config.py          training parameters
flylm/model.py           frozen-connectome model
flylm/data.py            byte streams and tiny corpus
flylm/generation.py      autoregressive generation
flylm/training.py        trainer and evaluation loop
datasets/prepare_graph.py
datasets/analyze_reachability.py
datasets/download_wikitext2.py
benchmark/operations.py  sparse implementations
benchmark/runner.py      measurement logic
benchmark/benchmark_graph.py
benchmark/probe_training.py
tests/
```

## Setup

Run commands from the project root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m datasets.prepare_graph
.\.venv\Scripts\python.exe -m datasets.analyze_reachability
.\.venv\Scripts\python.exe -m datasets.download_wikitext2
.\.venv\Scripts\python.exe -m unittest -v
```

`datasets.download_wikitext2` is the reproducible Internet download for the
exact `Salesforce/wikitext` configuration `wikitext-2-raw-v1`. It exports the
train, validation and test splits as UTF-8 text and records source URLs, sizes
and SHA-256 hashes. The downloaded corpus is intentionally excluded from Git,
so a public checkout needs this script unless the user supplies the files
manually.

## Commands

```powershell
.\.venv\Scripts\python.exe -m benchmark.benchmark_graph
.\.venv\Scripts\python.exe -m benchmark.probe_training --recurrent-steps 2 --batch-size 16 --bptt 32
.\.venv\Scripts\python.exe train.py --recurrent-steps 2 --batch-size 16 --bptt 32 --output-dir artifacts/tiny
```

The benchmark results measured on an RTX 5070 12 GB are in
[benchmark/BENCHMARK.md](benchmark/BENCHMARK.md). Raw datasets, prepared graph
files, benchmark JSON, checkpoints and training logs are excluded from Git.
