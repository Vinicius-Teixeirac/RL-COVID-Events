# RL-COVID-Events 

This repository contains the code necessary to conduct experiments evaluating neighborhood topological preservation in well-established dimensionality reduction methods applied to event-related COVID data.

## Overview

The project processes a dataset of COVID-related news articles. Each instance includes:
- **Text Content:** The full news article.
- **Geographical Coordinates:** Latitude and longitude of reported events.
- **Timestamps:** The corresponding dates of the events.

We get samples from the dataset. From each sample, we extract:
- **Semantic Features:** 768-dimensional text embeddings generated using a BERT model.
- **Geospatial Features:** Latitude-longitude coordinates.
- **Temporal Features:** Timestamps.

These features are used to construct an initial raw feature vector and to generate three independent nearest-neighborhood graphs representing:
- Semantic similarity.
- Geospatial proximity.
- Temporal closeness.

A reference consistency graph is then constructed by merging these three graphs.

(IMAGE HERE ILLUSTRATING THE CORE IDEA)

## Methodology

Dimensionality reduction techniques—**PCA**, **t-SNE**, and **UMAP**—are applied to the multidimensional raw feature vector (concatenating the embeddings, coordinates, and timestamps). The resulting lower-dimensional feature vectors are used to generate a new neighborhood graph, which is compared to the reference graph to evaluate preservation quality.

The comparison considers the edges of each graph by computing:
- **Precision:** Proportion of predicted edges that are correct.
- **Recall:** Proportion of reference edges that were correctly predicted.

## Hyperparameters

The evaluation varies several hyperparameters, including:
- Number of neighbors in the semantic, geospatial, and temporal neighbors graphs.
- Number of neighbors in the PCA-, t-SNE-, and UMAP-reduced neighbors graphs.
- t-SNE and UMAP initialization (PCA or Spectral).
- t-SNE perplexity.
- UMAP parameters: `n_neighbors` and `min_dist`.

## Evaluation

After the evaluation step, Critical Difference Diagrams are generated to determine which dimensionality reduction method produces the most accurate feature vector compared to the reference consistency graph.

# Technical Details

## Event Analysis
Event Analysis is a research area focused on understanding events and their interconnections. An event is defined as something that happens—with a description (what), a cause (why), at a location (where), at a specific time (when), by an agent (who), and in a particular manner (how). In short, an event is an action or series of actions, or a change that occurs due to specific reasons, involving entities such as objects, humans, and locations.

For example, researchers at the National Institute for Space Research discovered that fires in the Amazon Rainforest increased by 30% in 2019 using environmental data analysis techniques. This example illustrates how different dimensions of an event (who, what, when, where, how) are essential for a complete understanding:
- **Who:** Researchers at the National Institute for Space Research.
- **What:** Discovered that fires increased by 30%.
- **When:** In 2019.
- **Where:** In the Amazon Rainforest.
- **How:** Using environmental data analysis techniques.

Many applications are data-driven, and machine learning offers powerful tools to analyze events. However, because events are multi-dimensional—encompassing semantic, spatial, and temporal components—the challenge becomes: **How can events be modeled for machine learning while respecting these constraints?**

## Representation Learning
Representation Learning is a foundational area in modern machine learning. The seminal paper, ["Representation Learning: A Review and New Perspectives" by Bengio, Courville, and Vincent](https://arxiv.org/pdf/1206.5538), outlines key attributes of effective representations, such as smoothness, multiple explanatory factors, hierarchical organization, semi-supervised learning, shared factors across tasks, manifolds, natural clustering, temporal and spatial coherence, and simplicity in dependencies.

Building on these concepts, event analysis requires representations that incorporate additional constraints:
- The new feature space must be low-dimensional to facilitate human analysis, explanation, and interpretation of downstream machine learning decisions.
- The new space should preserve, as much as possible, the original semantic, geospatial, and temporal proximities. Events with close description, geographical coordinates and dates must remain close in the new space.

# Purpose
This repository aims to evaluate whether methods like PCA, t-SNE, and UMAP can generate representations that faithfully preserve the original multi-dimensional proximities of event data. By leveraging consistency graphs —constructed from independently generated nearest-neighbor graphs for semantic, geospatial, and temporal features— we can assess how well these dimensionality reduction methods maintain the inherent structure of the data.

If they perform well in that task, the new feature space is low-dimensional and maintain the original proximities, besides the fact they may be the main simple options for dimensionality reduction. If not, there's a path explore modern representation learning techniques.

# How to run
Every step present in the experiments is inside the run_experiments.sh, so a command like ./run_experiments in bash must be enough to generate the results. 

(Side note: in future versions, an main.py will substitute run_experiments to achive a cross-plataform compatibility)

The experiments start with DataPreparation folder. The pre_processing_dataset.ipynb has the original dataset and its preprocessing steps to generate the sampled datasets to be used in our experiments. Then, for each dataset, event features are extracted by event_features.py

After the feature extraction, in GraphEvaluation folder is possible to generate the reference neighbors graphs (by consistency_main.py) and in GraphGeneration folder the dimensionality reduction applied to data graphs (pca_main.py, tsne_main.py & umap_main.py)

Once the graph generation process is finished, back in GraphEvaluation, there are the evaluation_main.py to examine the 

# How to contribute
