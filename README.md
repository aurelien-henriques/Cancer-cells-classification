# Cancer-cells-classification

Single-cell RNA-seq classification
A RAMP data-challenge on the prediction of cellular types based on genes expression level

Nicolas Jouvin (MIA Paris-Saclay, Associate Professor @ Univ. Paris-Saclay), François Caud (DATAIA, Univ. Paris-Saclay)
This data-challenge was created for the data-camp course of the Master 2 Data-Science of Université Évry (Paris-Saclay)

Biologically, it is known that, while cells carry (almost) the same genomic information, they tend to express only a fraction of their genes leading to specialization into specific types with different biological functions. Thus, cell-types study and classification is of primary interest for many biological and medical applications. In the past decade, measuring genes expression level at the scale of a unique cell has become possible with the rise of high-throughput technologies named single-cell RNA-seq (scRNA-seq).

The goal of this data challenge is the supervised classification of cell-types thanks to the scMARK benchmark dataset from Mendonca et. al. The authors compiled 100, 000 cells expression from 10 different studies to serve as a comparison for different machine learning approaches, in an analogy with the MNIST benchmark dataset for computer vision.

This data-challenge uses a small extraction with only 4 cell-types (the labels to predict) from scMARK:

1. Cancer_cells
2. NK_cells
3. T_cells_CD4+
4. T_cells_CD8+

The public dataset contains 1500 points splitted in 1000 training points and 500 test points. It will serve as local benchmark for developing submissions. On the server side, submission will use the whole 1500 public points as the training set, and another private and unavailable test dataset, containing 1500 supplementary test points, will be used for the ranking of participants. The labels' distribution in the public (resp. private) training and testing datasets are the same. Deep learning not allowed. 

To answer this problematic I tried different types of machine learning algorithms (Gradient boost, Random Forest, logistic regression etc) and the best performances I got was 88% after hyperparameters optimization with a model composed of 3 steps. First, variable selection with LASSO to reduce the high dimensionnality of data (14000 variables). Second, a stacking model between random forest and MLP, to get the best of each one and finally a MLP to learn the answers proposed by the stacking. This model is pretty simple yet very effective for this task, you can find it in the "classifier.py" file. 
