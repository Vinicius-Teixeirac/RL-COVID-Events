import os
import json
import math
from pathlib import Path

# Make internal packages importable without modifying PYTHONPATH externally
os.environ.setdefault("PYTHONPATH", os.getcwd())

configfile: "workflow/config.yaml"

# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

DATASETS, = glob_wildcards(config["dataset_dir"] + "/{dataset}.pkl")
METHODS = [
    "ICA", "Isomap", "LLE", "PCA", "RandomProjection",
    "Spectral", "TSNE", "TSNE+PCA", "UMAP",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_params(dataset):
    """Return the k-value dict produced by the generate_params checkpoint."""
    p = checkpoints.generate_params.get(dataset=dataset).output[0]
    return json.load(open(p))


def method_graph_dir(method):
    base = config["graph_dir"]
    mapping = {"TSNE+PCA": "TSNE_PCA"}
    return f"{base}/{mapping.get(method, method)}"


# ---------------------------------------------------------------------------
# Top-level target
# ---------------------------------------------------------------------------

rule all:
    input:
        expand(
            config["critdd_dir"] + "/critdd_{dataset}.ipynb",
            dataset=DATASETS,
        )


# ---------------------------------------------------------------------------
# Preprocessing  (run once; output dir is treated as sentinel)
# ---------------------------------------------------------------------------

rule preprocess:
    output:
        touch("logs/preprocessing.done"),
    log:
        "logs/preprocessing.log",
    shell:
        "papermill DataPreparation/pre_processing_datasets.ipynb /dev/null"
        " -p output_dir {config[dataset_dir]}"
        " > {log} 2>&1"


# ---------------------------------------------------------------------------
# Per-dataset parameter generation (checkpoint — drives all downstream rules)
# ---------------------------------------------------------------------------

checkpoint generate_params:
    input:
        dataset=config["dataset_dir"] + "/{dataset}.pkl",
        prereq="logs/preprocessing.done",
    output:
        "workflow/params/{dataset}.json",
    run:
        import pandas as pd
        n = len(pd.read_pickle(input.dataset))
        max_k = max(1, int(math.isqrt(n)))
        step = max((max_k - 1) // 9, 1)
        k_vals = sorted(set([1 + i * step for i in range(9)] + [max_k]))
        n_nbrs = [k + 5 for k in k_vals]
        with open(output[0], "w") as f:
            json.dump(
                {
                    "k_values": k_vals,
                    "n_neighbors": n_nbrs,
                    "max_neighbors": max(n_nbrs),
                },
                f,
            )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

rule event_features:
    input:
        dataset=config["dataset_dir"] + "/{dataset}.pkl",
        params="workflow/params/{dataset}.json",
    output:
        expand(
            config["features_dir"] + "/{{dataset}}/{feat}.pkl",
            feat=["semantic", "geospatial", "temporal", "final"],
        ),
    log:
        "logs/{dataset}/event_features.log",
    shell:
        "python DataPreparation/event_features.py"
        " --dataset_path {input.dataset}"
        " --output_dir {config[features_dir]}"
        " > {log} 2>&1"


# ---------------------------------------------------------------------------
# k-NN precomputation (one file per metric, used by UMAP)
# ---------------------------------------------------------------------------

rule precompute_knn:
    input:
        final=config["features_dir"] + "/{dataset}/final.pkl",
        params="workflow/params/{dataset}.json",
    output:
        config["precomputed_knn_dir"] + "/{dataset}_{metric}_knn.pkl",
    log:
        "logs/{dataset}/knn_{metric}.log",
    params:
        random_state=config["random_state"],
    run:
        p = load_params(wildcards.dataset)
        shell(
            "python Utils/NeighborhoodUtils/compute_knn.py"
            f" --dataset_name {wildcards.dataset}"
            f" --max_neighbors {p['max_neighbors']}"
            f" --metric {wildcards.metric}"
            f" --random_state {params.random_state}"
            f" --output_path {output[0]}"
            f" > {log[0]} 2>&1"
        )


# ---------------------------------------------------------------------------
# Consistency (reference) graphs
# ---------------------------------------------------------------------------

rule consistency_graph:
    input:
        config["features_dir"] + "/{dataset}/semantic.pkl",
        config["features_dir"] + "/{dataset}/geospatial.pkl",
        config["features_dir"] + "/{dataset}/temporal.pkl",
    output:
        config["graph_dir"] + "/Consistency/{dataset}/consistency_edges_{ks}_{kg}_{kt}.pkl",
    log:
        "logs/{dataset}/Consistency/consistency_{ks}_{kg}_{kt}.log",
    shell:
        "python GraphEvaluation/consistency_main.py"
        " --dataset_name {wildcards.dataset}"
        " --k_s {wildcards.ks} --k_g {wildcards.kg} --k_t {wildcards.kt}"
        " --output_dir {config[graph_dir]}/Consistency"
        " > {log} 2>&1"


def all_consistency_outputs(wildcards):
    p = load_params(wildcards.dataset)
    k = p["k_values"]
    return expand(
        config["graph_dir"] + "/Consistency/{dataset}/consistency_edges_{ks}_{kg}_{kt}.pkl",
        dataset=wildcards.dataset, ks=k, kg=k, kt=k,
    )


rule all_consistency:
    input: all_consistency_outputs
    output: touch("logs/{dataset}/consistency.done")


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

rule pca_graph:
    input:
        config["features_dir"] + "/{dataset}/final.pkl",
    output:
        config["graph_dir"] + "/PCA/{dataset}/pca_edges_{k}_{whiten}.pkl",
    log:
        "logs/{dataset}/PCA/pca_{k}_{whiten}.log",
    params:
        n_components=config["n_components"],
        random_state=config["random_state"],
    shell:
        "python GraphGeneration/PCA/pca_main.py"
        " --dataset_name {wildcards.dataset}"
        " --k_pca {wildcards.k} --whiten {wildcards.whiten}"
        " --n_components {params.n_components}"
        " --random_state {params.random_state}"
        " --output_dir {config[graph_dir]}/PCA"
        " > {log} 2>&1"


def all_pca_outputs(wildcards):
    p = load_params(wildcards.dataset)
    return expand(
        config["graph_dir"] + "/PCA/{dataset}/pca_edges_{k}_{w}.pkl",
        dataset=wildcards.dataset, k=p["k_values"], w=config["pca_whitens"],
    )


rule all_pca:
    input: all_pca_outputs
    output: touch("logs/{dataset}/PCA.done")


# ---------------------------------------------------------------------------
# ICA
# ---------------------------------------------------------------------------

rule ica_graph:
    input:
        config["features_dir"] + "/{dataset}/final.pkl",
    output:
        config["graph_dir"] + "/ICA/{dataset}/ica_edges_{k}_{whiten}_{fun}.pkl",
    log:
        "logs/{dataset}/ICA/ica_{k}_{whiten}_{fun}.log",
    params:
        n_components=config["n_components"],
        random_state=config["random_state"],
    shell:
        "python GraphGeneration/ICA/ica_main.py"
        " --dataset_name {wildcards.dataset}"
        " --k_ica {wildcards.k} --whiten {wildcards.whiten} --fun {wildcards.fun}"
        " --n_components {params.n_components}"
        " --random_state {params.random_state}"
        " --output_dir {config[graph_dir]}/ICA"
        " > {log} 2>&1"


def all_ica_outputs(wildcards):
    p = load_params(wildcards.dataset)
    return expand(
        config["graph_dir"] + "/ICA/{dataset}/ica_edges_{k}_{w}_{f}.pkl",
        dataset=wildcards.dataset,
        k=p["k_values"],
        w=config["ica_whitens"],
        f=config["ica_funs"],
    )


rule all_ica:
    input: all_ica_outputs
    output: touch("logs/{dataset}/ICA.done")


# ---------------------------------------------------------------------------
# Isomap
# ---------------------------------------------------------------------------

rule isomap_graph:
    input:
        config["features_dir"] + "/{dataset}/final.pkl",
    output:
        config["graph_dir"] + "/Isomap/{dataset}/isomap_edges_{k}_{n_neighbors}.pkl",
    log:
        "logs/{dataset}/Isomap/isomap_{k}_{n_neighbors}.log",
    params:
        n_components=config["n_components"],
    shell:
        "python GraphGeneration/Isomap/isomap_main.py"
        " --dataset_name {wildcards.dataset}"
        " --k_isomap {wildcards.k} --n_neighbors {wildcards.n_neighbors}"
        " --n_components {params.n_components}"
        " --output_dir {config[graph_dir]}/Isomap"
        " > {log} 2>&1"


def all_isomap_outputs(wildcards):
    p = load_params(wildcards.dataset)
    pairs = list(zip(p["k_values"], p["n_neighbors"]))
    return [
        f"{config['graph_dir']}/Isomap/{wildcards.dataset}/isomap_edges_{k}_{nn}.pkl"
        for k, nn in pairs
    ]


rule all_isomap:
    input: all_isomap_outputs
    output: touch("logs/{dataset}/Isomap.done")


# ---------------------------------------------------------------------------
# LLE
# ---------------------------------------------------------------------------

rule lle_graph:
    input:
        config["features_dir"] + "/{dataset}/final.pkl",
    output:
        config["graph_dir"] + "/LLE/{dataset}/lle_edges_{k}_{n_neighbors}_{method}.pkl",
    log:
        "logs/{dataset}/LLE/lle_{k}_{n_neighbors}_{method}.log",
    params:
        n_components=config["n_components"],
        random_state=config["random_state"],
    shell:
        "python GraphGeneration/LLE/lle_main.py"
        " --dataset_name {wildcards.dataset}"
        " --k_lle {wildcards.k} --n_neighbors {wildcards.n_neighbors}"
        " --method {wildcards.method}"
        " --n_components {params.n_components}"
        " --random_state {params.random_state}"
        " --output_dir {config[graph_dir]}/LLE"
        " > {log} 2>&1"


def all_lle_outputs(wildcards):
    p = load_params(wildcards.dataset)
    pairs = list(zip(p["k_values"], p["n_neighbors"]))
    return [
        f"{config['graph_dir']}/LLE/{wildcards.dataset}/lle_edges_{k}_{nn}_{m}.pkl"
        for k, nn in pairs
        for m in config["lle_methods"]
    ]


rule all_lle:
    input: all_lle_outputs
    output: touch("logs/{dataset}/LLE.done")


# ---------------------------------------------------------------------------
# Random Projection
# ---------------------------------------------------------------------------

rule rp_graph:
    input:
        config["features_dir"] + "/{dataset}/final.pkl",
    output:
        config["graph_dir"] + "/RandomProjection/{dataset}/rp_edges_{k}.pkl",
    log:
        "logs/{dataset}/RandomProjection/rp_{k}.log",
    params:
        n_components=config["n_components"],
        random_state=config["random_state"],
    shell:
        "python GraphGeneration/RandomProjection/random_main.py"
        " --dataset_name {wildcards.dataset}"
        " --k_rp {wildcards.k}"
        " --n_components {params.n_components}"
        " --random_state {params.random_state}"
        " --output_dir {config[graph_dir]}/RandomProjection"
        " > {log} 2>&1"


def all_rp_outputs(wildcards):
    p = load_params(wildcards.dataset)
    return expand(
        config["graph_dir"] + "/RandomProjection/{dataset}/rp_edges_{k}.pkl",
        dataset=wildcards.dataset, k=p["k_values"],
    )


rule all_rp:
    input: all_rp_outputs
    output: touch("logs/{dataset}/RandomProjection.done")


# ---------------------------------------------------------------------------
# Spectral Embedding
# ---------------------------------------------------------------------------

rule spectral_graph:
    input:
        config["features_dir"] + "/{dataset}/final.pkl",
    output:
        config["graph_dir"] + "/Spectral/{dataset}/spectral_edges_{k}_{n_neighbors}.pkl",
    log:
        "logs/{dataset}/Spectral/spectral_{k}_{n_neighbors}.log",
    params:
        n_components=config["n_components"],
        random_state=config["random_state"],
    shell:
        "python GraphGeneration/SpectralEmbedding/spectral_main.py"
        " --dataset_name {wildcards.dataset}"
        " --k_spectral {wildcards.k} --n_neighbors {wildcards.n_neighbors}"
        " --n_components {params.n_components}"
        " --random_state {params.random_state}"
        " --output_dir {config[graph_dir]}/Spectral"
        " > {log} 2>&1"


def all_spectral_outputs(wildcards):
    p = load_params(wildcards.dataset)
    pairs = list(zip(p["k_values"], p["n_neighbors"]))
    return [
        f"{config['graph_dir']}/Spectral/{wildcards.dataset}/spectral_edges_{k}_{nn}.pkl"
        for k, nn in pairs
    ]


rule all_spectral:
    input: all_spectral_outputs
    output: touch("logs/{dataset}/Spectral.done")


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

rule tsne_graph:
    input:
        config["features_dir"] + "/{dataset}/final.pkl",
    output:
        config["graph_dir"] + "/TSNE/{dataset}/tsne_edges_{k}_{perplexity}_{init}_{metric}_{apply_pca}.pkl",
    log:
        "logs/{dataset}/TSNE/tsne_{k}_{perplexity}_{init}_{metric}_{apply_pca}.log",
    params:
        n_components=config["n_components"],
        random_state=config["random_state"],
    shell:
        "python GraphGeneration/TSNE/tsne_main.py"
        " --dataset_name {wildcards.dataset}"
        " --k_tsne {wildcards.k}"
        " --perplexity {wildcards.perplexity}"
        " --initialization {wildcards.init}"
        " --metric {wildcards.metric}"
        " --apply_pca {wildcards.apply_pca}"
        " --n_components {params.n_components}"
        " --random_state {params.random_state}"
        " --output_dir {config[graph_dir]}/TSNE"
        " > {log} 2>&1"


def all_tsne_outputs(wildcards):
    p = load_params(wildcards.dataset)
    return expand(
        config["graph_dir"] + "/TSNE/{dataset}/tsne_edges_{k}_{perp}_{init}_{metric}_{apca}.pkl",
        dataset=wildcards.dataset,
        k=p["k_values"],
        perp=config["tsne_perplexities"],
        init=config["tsne_inits"],
        metric=config["metrics"],
        apca=config["tsne_apply_pca"],
    )


rule all_tsne:
    input: all_tsne_outputs
    output: touch("logs/{dataset}/TSNE.done")


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

rule umap_graph:
    input:
        features=config["features_dir"] + "/{dataset}/final.pkl",
        knn=config["precomputed_knn_dir"] + "/{dataset}_{metric}_knn.pkl",
    output:
        config["graph_dir"] + "/UMAP/{dataset}/umap_edges_{k}_{n_neighbors}_{min_dist}_{init}_{metric}.pkl",
    log:
        "logs/{dataset}/UMAP/umap_{k}_{n_neighbors}_{min_dist}_{init}_{metric}.log",
    params:
        n_components=config["n_components"],
        random_state=config["random_state"],
    shell:
        "python GraphGeneration/UMAP/umap_main.py"
        " --dataset_name {wildcards.dataset}"
        " --k_umap {wildcards.k}"
        " --n_neighbors {wildcards.n_neighbors}"
        " --min_dist {wildcards.min_dist}"
        " --initialization {wildcards.init}"
        " --metric {wildcards.metric}"
        " --n_components {params.n_components}"
        " --random_state {params.random_state}"
        " --precomputed_knn_path {input.knn}"
        " --output_dir {config[graph_dir]}/UMAP"
        " > {log} 2>&1"


def all_umap_outputs(wildcards):
    p = load_params(wildcards.dataset)
    pairs = list(zip(p["k_values"], p["n_neighbors"]))
    return [
        f"{config['graph_dir']}/UMAP/{wildcards.dataset}"
        f"/umap_edges_{k}_{nn}_{md}_{init}_{metric}.pkl"
        for k, nn in pairs
        for md in config["umap_min_dists"]
        for init in config["umap_inits"]
        for metric in config["metrics"]
    ]


rule all_umap:
    input: all_umap_outputs
    output: touch("logs/{dataset}/UMAP.done")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_METHOD_TO_DONE_RULE = {
    "ICA": "ICA",
    "Isomap": "Isomap",
    "LLE": "LLE",
    "PCA": "PCA",
    "RandomProjection": "RandomProjection",
    "Spectral": "Spectral",
    "TSNE": "TSNE",
    "TSNE+PCA": "TSNE",
    "UMAP": "UMAP",
}


rule evaluate:
    input:
        consistency="logs/{dataset}/consistency.done",
        method_graphs=lambda w: f"logs/{w.dataset}/{_METHOD_TO_DONE_RULE[w.method]}.done",
    output:
        config["eval_dir"] + "/{dataset}/{method}_results.csv",
    log:
        "logs/{dataset}/Evaluation/eval_{method}.log",
    params:
        n_jobs=config["eval_n_jobs"],
        comp_folder=lambda w: method_graph_dir(w.method),
    shell:
        "python GraphEvaluation/evaluation_main.py"
        " --dataset_name {wildcards.dataset}"
        " --reference_folder {config[graph_dir]}/Consistency"
        " --comparison_folder {params.comp_folder}"
        " --method {wildcards.method}"
        " --output_dir {config[eval_dir]}"
        " --n_jobs {params.n_jobs}"
        " > {log} 2>&1"


# ---------------------------------------------------------------------------
# Critical difference diagrams
# ---------------------------------------------------------------------------

rule critdd:
    input:
        expand(
            config["eval_dir"] + "/{{dataset}}/{method}_results.csv",
            method=METHODS,
        ),
    output:
        config["critdd_dir"] + "/critdd_{dataset}.ipynb",
    log:
        "logs/{dataset}/critdd.log",
    shell:
        "papermill GraphEvaluation/critdd_template.ipynb"
        " {output}"
        " -p dataset {wildcards.dataset}"
        " > {log} 2>&1"
