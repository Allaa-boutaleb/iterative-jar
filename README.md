# 🫙 Exploring Multi-Table Retrieval Through Iterative Search

This repository contains the code for **Greedy Join-Aware Multi-Table Retrieval**, as proposed in the paper:

> **Exploring Multi-Table Retrieval Through Iterative Search**  
> *(Anonymous Author(s))*

Our work introduces a fast, **iterative search** approach to multi-table retrieval, providing a scalable alternative to the computationally intensive **Mixed-Integer Programming (MIP)** formulation of the original **Join-Aware Retrieval (JAR)** method.

---

## 📄 Original JAR Citation (MIP-Based Baseline)

This project is a **fork** of the repository for the **Join-Aware Retrieval (JAR)** method.  
If you use the `ilp.py` script or foundational components from the original work, please cite:

```bibtex
@article{chen2024table,
  title={Is Table Retrieval a Solved Problem? Join-Aware Multi-Table Retrieval},
  author={Chen, Peter Baile and Zhang, Yi and Roth, Dan},
  journal={arXiv preprint arXiv:2404.09889},
  year={2024}
}
```

---

## 🧩 Multi-Table Retrieval Components

This repository includes implementations of both the **MIP-based optimization** from the original JAR approach and the **Greedy Iterative Search** algorithm proposed in our work.

### Algorithms
* **Greedy Join-Aware Retrieval:** Iterative heuristic-based table selection (`greedy.py`)
* **Join-Aware Retrieval (JAR):** Mixed-Integer Programming (MIP) optimizer for provably optimal retrieval (`ilp.py`)

### Shared Modules
* Query decomposition — `decomp.py`  
* Table-table compatibility scoring — `compatibility.py`  
* Dense retrieval baselines — `contriever.py`, `tapas.py`  
* Evaluation metrics and utilities — `metrics.py`, `utils.py`

---

## 🚀 Usage

### 1. Setup

Make sure to download Spider and BIRD datasets here: [Google drive folder](https://drive.google.com/drive/folders/1PtLan7Guu98J42lqCxhZZc-EyZpvwWVk?usp=sharing).

To execute the MIP program (`ilp.py`), we recommend installing the [Gurobi solver](https://docs.python-mip.com/en/latest/install.html#gurobi-installation-and-configuration-optional) for significantly faster optimization.

Pre-computed **table compatibility scores** and **query decomposition data** for *Spider* and *Bird* are typically stored under `data/`.  
Database dumps (`dev_database`) and fine-tuned `TAPAS-large` checkpoints are available from the original authors’ public drive.

---

### 2. Prerequisite Scores

Before running the main retrieval algorithms (`ilp.py` or `greedy.py`), you must pre-compute the relevance scores which serve as inputs for re-ranking.

1. Run `contriever.py` or `tapas.py` to compute **coarse-grained relevance scores**.  
2. Run `contriever.py` again (with fine-tuned parameters) to obtain **fine-grained relevance scores**.

These scores form the **Top-K initial candidate set** (e.g., Top-20 tables) used for both methods.

---

### 3. Running the Algorithms

Parallelized versions of the scripts use partition arguments specified by `--num_partitions` and `-p`.

#### A. Greedy Join-Aware Retrieval (Iterative Search)

Run the heuristic, iterative method via:

```bash
# Example: Spider dataset with K=5
python greedy.py --dataset spider --K 5 --topk 20 \
  --lambda_cov 2.0 --lambda_join 1.0 --lambda_coarse 3.0
```

#### Arguments

| Argument | Description | Default (Used in Paper) |
|-----------|--------------|--------------------------|
| `--dataset` | Benchmark dataset (`bird`, `spider`, etc.) | N/A |
| `--K` | Number of tables selected in the final set | 2, 5, or 10 |
| `--topk` | Size of the initial candidate pool (Top-K from Contriever) | 20 |
| `--lambda_cov` | Weight for Marginal Coverage Gain \( G_{cov} \) | 2.0 |
| `--lambda_join` | Weight for Marginal Join Gain \( G_{join} \) | 1.0 |
| `--lambda_coarse` | Weight for Coarse Relevance Gain \( G_{coarse} \) | 3.0 |

---

#### B. Join-Aware Retrieval (MIP Optimization)

Run the original one-shot MIP optimization approach:

```bash
# Example for parallel execution
python ilp.py -p 0 & python ilp.py -p 1 & ...
```

---

## 📫 Contact

For questions or feedback, please reach out via the repository’s issue tracker or contact the author(s) directly.

---
