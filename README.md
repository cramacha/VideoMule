# VideoMule

## VideoMule Consensus Simulation
This repository contains a Python script that simulates the core consensus learning mechanism of VideoMule, a framework designed for multi-label classification of multimodal data (specifically, video and text in the original context).
The simulation focuses on the graph-based label propagation step, which combines the initial "beliefs" from various learning models (supervised and unsupervised) into a final, robust multi-label prediction.
## How VideoMule Works (Simulated Steps)
The simulation runs through four main stages:
1. Multimodal Feature Generation
 * Goal: Create synthetic feature data for two modalities: Video (X_{\text{video}}) and Text (X_{\text{text}}), along with a binary multi-label ground truth (Y_{\text{true}}).
 * Implementation: The generate_simulated_data function uses numpy to generate random matrices to stand in for complex, real-world features.
2. MultiLabel Tree Simulation (Initial Beliefs)
 * Goal: Simulate the output of individual learning models (\lambda) trained on different modalities, which form the initial knowledge graph nodes.
 * Models:
   * Supervised: DecisionTreeClassifier on X_{\text{video}} and X_{\text{text}}.
   * Unsupervised: KMeans clustering on X_{\text{video}} and X_{\text{text}}.
 * Output: The soft predictions (probabilities or cluster membership) from all models are concatenated to form the High-Dimensional Belief Graph (Y_{\text{initial\_G}}).
3. VideoMule Consensus Algorithm (Label Propagation)
 * Goal: Iteratively refine the label probabilities by propagating knowledge across the belief graph.
 * Mechanism: Implemented in the VideoMuleConsensus class, which uses the Label Propagation algorithm:
   * Affinity Matrix (M): Computed based on the Cosine Similarity of the high-dimensional belief vectors (Y_{\text{initial\_G}}).
   * Normalized Graph (P): The similarity matrix is normalized.
   * Iteration: The final probability matrix Q is solved iteratively.
4. Evaluation and Final Labels
 * Goal: Convert the consensus probability matrix (Q) into binary multi-labels (S_{\text{labels}}) and evaluate performance.
 * Metrics: The results are evaluated using common multi-label metrics from sklearn.metrics, including Subset Accuracy, Micro-averaged Precision, and Micro-averaged Recall.
## Prerequisites
The script requires the following Python libraries:
 * numpy
 * scikit-learn (sklearn)
 * scipy
You can install them using pip:
pip install numpy scikit-learn scipy

## Getting Started
 * Run the script from your terminal:
python VideoMule.py

## Modifying Parameters
You can easily adjust the simulation by changing parameters within the run_videomule_simulation function:
| Parameter | Location | Default Value | Description |
|---|---|---|---|
| n_samples | run_videomule_simulation | 100 | Number of data samples simulated. |
| n_labels | run_videomule_simulation | 5 | Number of possible multi-labels/classes. |
| n_clusters | run_videomule_simulation | 4 | Number of clusters for the K-Means models. |
| alpha | VideoMuleConsensus.__init__ | 0.9 | The propagation coefficient in the consensus step. |

## Key Classes and Functions
| Component | Type | Description |
|---|---|---|
| VideoMuleConsensus | Class | Implements the graph-based consensus learning using the Label Propagation technique. |
| generate_simulated_data | Function | Creates random feature and multi-label matrices for the simulation. |
| multilabel_tree_simulate | Function | Trains base models (Decision Tree, K-Means) and extracts initial soft predictions (beliefs). |
| run_videomule_simulation | Function | Orchestrates the entire process from data generation to consensus and evaluation. |

## Paper

```
@inproceedings{ramachandran2009videomule,
  title={Videomule: a consensus learning approach to multi-label classification from noisy user-generated videos},
  author={Ramachandran, Chandrasekar and Malik, Rahul and Jin, Xin and Gao, Jing and Nahrstedt, Klara and Han, Jiawei},
  booktitle={Proceedings of the 17th ACM international conference on Multimedia},
  pages={721--724},
  year={2009}
}
```
