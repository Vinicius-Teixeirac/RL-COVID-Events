# RL-COVID-Events  

This repository provides the necessary code and resources for conducting experiments that evaluate **neighborhood topological preservation** in widely used **dimensionality reduction techniques** applied to **event-related COVID-19 data**. The main goal is determine whether or not such techniques are useful for event analysis' representation learning.

## Overview  

This project processes a dataset of **COVID-19-related news articles**, where each article describes an event. Every event in the dataset includes the following key components:  

- **Text Component:** The news article's **headline**, which serves as a textual summary of the event.  
- **Geographical Component:** The **latitude and longitude** of the location where the event occurred or was reported.  
- **Temporal Component:** The **date** associated with the reported event.  

To analyze these data, we make samples of events based on predefined **labels**. For each sample, which will became itself an usage dataset, we extract and transform its attributes into numerical *event features* for further processing:  

- **Semantic Features:** The news headlines are converted into **768-dimensional embeddings** using a **pre-trained BERT model**. These embeddings capture the semantic meaning of the text.  
- **Geospatial Features:** The event's **latitude and longitude** are used as raw geographical coordinates to represent spatial positioning.  
- **Temporal Features:** The **event timestamps** are extracted and converted into a numerical format that preserves temporal relationships.  

From each of these three kinds of features individually, we generate **nearest-neighbor graphs**, emphasizing a different type of relationship:  

- **Semantic Similarity Graph:** Connects events with most similar textual meanings based on their BERT embeddings.  
- **Geospatial Proximity Graph:** Connects events that occurred nearest to each other geographically.  
- **Temporal Closeness Graph:** Connects events that happened within the closest time frame.  

<div style="text-align: center; margin-bottom: 20px;">
    <img src="Images/semantic_graph_example.png" alt="Semantic Graph" style="width: 100%; max-width: 500px; height: auto;">
    <img src="Images/geospatial_graph_example.png" alt="Geospatial Graph" style="width: 100%; max-width: 500px; height: auto;">
</div>

<div style="text-align: center; margin-bottom: 20px;">
    <img src="Images/temporal_graph_example.png" alt="Temporal Graph" style="width: 100%; max-width: 1000px; height: auto;">
</div>

<p style="text-align: left; margin-bottom: 20px;">
    We then construct a <strong>reference consistency graph</strong> by summing the adjacency matrices of the three graphs and applying a threshold to the resulting values. An edge is established between two nodes if their summed value meets or exceeds this threshold, meaning that such edge exists at least in <em>threshold</em> of the three graphs. This final structure serves as a foundation for evaluating how effectively various dimensionality reduction techniques preserve local neighborhood structures in high-dimensional event-related data.
</p>

We also concatenate these three kinds of features, constructing an **initial raw feature vector** for each event. This  **raw feature vector** is what will be passed for the dimensionality reduction tecniques, which will create embeddings/representations in low dimension for the events.

<div style="text-align: center; margin-bottom: 20px;">
    <img src="Images/event_analysis.png" alt="Event Analysis" style="width: 100%; max-width: 1000px; height: auto;">
</div>

## Methodology

Dimensionality reduction techniques are applied to the multidimensional raw feature vector. The repository explores the following techniques:

- **ICA**
- **Isomap**
- **Locally Linear Embeddings (LLE)**
- **PCA**
- **Gaussian Random Projection**
- **Laplacian Eigenmaps**
- **t-SNE**
- **UMAP**

These methods were selected because they are well-established in the literature and offer computational efficiency. Other techniques were tested but ultimately excluded for various reasons:

* Usage Restrictions: For example, NMF decomposition requires strictly positive data, which cannot be assumed in this context.

* Instability: Implementations of diffusion maps did not converge reliably in experiments.

* High Computational Cost: Methods such as Kernel PCA and MDS proved too computationally intensive.

Future work will focus on exploring more advanced techniques, including autoencoders, variational autoencoders, parametric UMAP, supervised UMAP, Dense UMAP, parametric t-SNE, and PacMap. Feedback on additional promising methods is welcome.

The methodology involves:
1. **Reduction:** Converting the high-dimensional raw features into a lower-dimensional space using the aforementioned techniques.
2. **Graph Generation:** Constructing a new neighborhood graph based on the reduced features.
3. **Comparison:** Evaluating the quality of the neighborhood preservation by comparing the generated graph with reference consistency graphs using the following metrics:
   - **Precision:** Amount of generated edges that are correct (according to the reference graph), over all generated edges.
   - **Recall:** Amount of reference edges that are correctly generated, over the amount edges in reference graph.
   - **F1-score:** The harmonic mean of precision and recall.

## Hyperparameters

The experiments vary several hyperparameters across both the reference graphs and the dimensionality reduction techniques. Key hyperparameters include:

### Reference Graphs
- **Number of neighbors:**
  - Semantic (`k_s`)
  - Geospatial (`k_g`)
  - Temporal (`k_t`)

### Dimensionality-Reduced Graphs
- **Number of neighbors:** Specific parameters (e.g., `k_ica`, `k_isomap`, etc.) for each technique.

### Technique-Specific Hyperparameters
- **ICA:**
  - Whitening strategy (`whiten`)
  - Functional form (`fun`) for approximating neg-entropy.
- **Isomap:**
  - Number of neighbors (`n_neighbors`) to consider for each point, when attempting to learn the manifold structure.
- **Locally Linear Embeddings (LLE):**
  - Number of neighbors (`n_neighbors`)
  - Algorithm/method (`method`)
- **PCA:**
  - Whitening option (`whiten`), which ensures outputs with unit component-wise variances.
- **Laplacian Eigenmaps:**
  - Number of neighbors (`n_neighbors`)
- **t-SNE and UMAP:**
  - Initialization (`PCA` or `Spectral`)
  - Metric (`euclidean` or `cosine`)
- **t-SNE:**
  - Perplexity (`perplexity`)
- **UMAP:**
  - Number of neighbors (`n_neighbors`)
  - Minimum distance (`min_dist`) between points in the lower-dimensional representation.

## Evaluation

Each combination of hyperparameters might produce a different graph. Therefore, each technique and consistency setting generate multiple graphs that need to be compared. Every graph produced by a technique is compared to each consistency graph. To evaluate a technique's performance for a given consistency graph, we take the mean of the top 20% of the technique's generated graphs for this consistency graph. Critical Difference Diagrams are then used to analyze performance across the consistency graphs, helping to identify which dimensionality reduction method produces the most accurate feature vectors for each usage dataset.

# Technical Details

## Event Analysis

Event Analysis is a research area focused on understanding events and their interconnections. An event is defined as something that happens—with a description (what), a cause (why), at a location (where), at a specific time (when), by an agent (who), and in a particular manner (how). In short, an event is an action or series of actions, or a change that occurs due to specific reasons, involving entities such as objects, humans, and locations.

An event example can be as follows: _Researchers at the National Institute for Space Research discovered that fires in the Amazon Rainforest increased by 30% in 2019 using environmental data analysis_. We shall break this event in the previous components: 
- **Who:** Researchers at the National Institute for Space Research.
- **What:** Discovered that fires increased by 30%.
- **When:** In 2019.
- **Where:** In the Amazon Rainforest.
- **How:** Using environmental data analysis.

Many applications are data-driven, and machine learning offers powerful tools to analyze events. However, because events are multi-dimensional, encompassing semantic, spatial, and temporal components, the challenge becomes: **How can events be modeled for machine learning while preserving these components?**

## Representation Learning
Representation Learning is a foundational area in modern machine learning. The seminal paper, ["Representation Learning: A Review and New Perspectives" by Bengio, Courville, and Vincent](https://arxiv.org/pdf/1206.5538), outlines key attributes of effective representations, such as smoothness, multiple explanatory factors, hierarchical organization, semi-supervised learning, shared factors across tasks, manifolds, natural clustering, temporal and spatial coherence, and simplicity in dependencies.

Building on these concepts, event analysis requires representations that incorporate additional constraints:
- The new feature space must be low-dimensional to facilitate human analysis, explanation, and interpretation of downstream machine learning decisions.
- The new space should preserve, as much as possible, the original semantic, geospatial, and temporal proximities. Events with close description, geographical coordinates and dates must remain close in the new space.

These constraints arise because, unlike in other machine learning applications, understanding decisions and the initial relationships between events is paramount to event analysis.

# Purpose

This repository aims to evaluate whether tecniques like PCA, t-SNE, and UMAP can generate representations that faithfully preserve the original multi-dimensional proximities of event data. By leveraging consistency graphs —constructed from independently generated nearest-neighbor graphs for semantic, geospatial, and temporal features— we can assess how well these dimensionality reduction methods maintain the inherent structure of the data.

If they perform well in that task, the new feature space is low-dimensional and maintain the original proximities, besides the fact they may be the main simple options for dimensionality reduction. If not, there's a path explore modern representation learning techniques.

# Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 – 3.12 |
| Package management | [uv](https://docs.astral.sh/uv/) |
| Pipeline orchestration | [Snakemake](https://snakemake.readthedocs.io/) |
| DR — linear | [scikit-learn](https://scikit-learn.org/) — PCA, ICA (FastICA), Gaussian Random Projection |
| DR — manifold | [scikit-learn](https://scikit-learn.org/) — Isomap, LLE, Spectral Embedding |
| DR — t-SNE | [openTSNE](https://opentsne.readthedocs.io/) |
| DR — UMAP | [umap-learn](https://umap-learn.readthedocs.io/) |
| Graphs | [NetworkX](https://networkx.org/) |
| Numerical | [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) · [pandas](https://pandas.pydata.org/) |
| Parallelism | [joblib](https://joblib.readthedocs.io/) |
| Notebook execution | [papermill](https://papermill.readthedocs.io/) |
| Statistical diagrams | [critdd](https://github.com/mirkobunse/critdd) |

# How to run

## Environment setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync                   # install runtime dependencies (Python 3.11–3.12)
uv sync --group workflow  # also install Snakemake
```

## Running the full pipeline

```bash
uv run snakemake --cores all
```

This executes the complete workflow: preprocessing → feature extraction → graph generation (all 8 DR methods + consistency graphs) → evaluation → critical difference diagrams.

To preview what would run without executing:

```bash
uv run snakemake --cores all -n
```

## Tuning for your hardware

Edit `workflow/config.yaml` before running:

| Key | Default | Effect |
|-----|---------|--------|
| `cores` | `8` | Set to your CPU core count |
| `eval_n_jobs` | `100` | Parallel workers inside each evaluation step |
| `n_components` | `2` | Output dimensionality for all DR methods |
| `random_state` | `42` | Seed for all stochastic methods |

## Execution workflow

The pipeline stages, in order:

1. **Preprocessing** — runs `DataPreparation/preprocessing_datasets.ipynb` via papermill; produces sampled datasets in `DataPreparation/UsageDatasets/`
2. **Feature extraction** — extracts semantic (768-d BERT), geospatial (lat/lng), and temporal features per dataset; saves to `DatasetEventFeatures/`
3. **Graph generation** — builds k-NN graphs for all hyperparameter combinations; the k-value sweep is derived automatically from `sqrt(n_rows)` per dataset
4. **Evaluation** — compares each DR graph against every consistency graph; results saved as Parquet files in `EvaluationResults/`
5. **Critical difference diagrams** — ranks methods via `critdd`; output notebooks in `CritddResults/`

Snakemake tracks all file dependencies and skips steps whose outputs already exist, so partial runs can be resumed safely.

> **Note:** A full run across 6 datasets is compute-intensive and was originally run on a 128-core machine. On a typical workstation, expect several hours per dataset. The original shell scripts are archived in `Scripts/legacy/` for reference.

# How to contribute
