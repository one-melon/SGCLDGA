# SGCLDGA
Predicting drug-gene interactions is crucial for various aspects of drug discovery. However, traditional biological
experiments are not only time-consuming and costly, but also inefficient and susceptible to external environment. To
address this issue, we propose a novel computational model, called SGCLDGA, which is based on graph neural network
and self-supervised contrastive learning, and can predict unknown drug-gene associations. The main idea of SGCLDGA
is to use GCN to extract vector representations of drugs and genes from the original drug-gene bipartite graph, and
then use SVD to enhance the graph and generate multiple views. Then, SGCLDGA performs contrastive learning across
different views, and optimizes the vector representations by using contrastive loss function, which enables them to better
distinguish positive and negative samples. Finally, SGCLDGA uses inner product to calculate the association scores
between drugs and genes. Our experiments on DGIdb4.0 dataset show that SGCLDGA outperforms five state-of-the-art
methods. In addition, we also conduct ablation studies and case studies, which verify the important roles of contrastive
learning and SVD, and the potential of SGCLDGA in discovering new drug-gene associations.

# Requirements
- torch 1.8.1
- python 3.7.16
- numpy 1.21.6
- pandas 1.3.5
- scikit-learn 1.0.2
- scipy 1.7.3

# Run the demo
```
python main.py
```
