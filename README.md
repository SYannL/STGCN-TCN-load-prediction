# STGCN-TCN
A project implementing Load prediction based on TCN-STGCN
# Spatio-Temporal Graph Convolutional Networks (STGCN) Project

This project performs analysis and forecasting on spatio-temporal data by implementing Spatio-Temporal Graph Convolutional Networks (STGCN). The project includes implementations of Temporal Convolutional Networks (TCN) and Spatio-Temporal Graph Convolutional Networks (STGCN), along with scripts for data preprocessing and model training.

## Project Structure

- `stgcn_tcn.py`: Contains the implementation of Spatio-Temporal Graph Convolutional Networks (STGCN), along with modifications for 2D convolution in the original TCN.
- `get_Adjacency_X.py`: Contains utility functions for data loading, preprocessing, and preparation of the graph adjacency matrix.
- `run.py`: Main execution script, including the code for model training, validation, and testing.

## Environment Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

You can install the dependencies with the following command:
```bash
pip install -r requirements.txt


## Feature Extraction in Temporal and Spatial Domains

The process of feature extraction in this project is carried out through spatio-temporal convolutional blocks, which consist of three main components: a Temporal Convolution Block, a Spatial Convolution Block, and another Temporal Convolution Block, in that sequence. These blocks are designed to capture the inherent spatial and temporal dependencies in the data.

### Temporal Convolution Block:

The Temporal Convolution Blocks are implemented using Temporal Convolution Networks (TCN). TCN employs dilated convolutions to capture long-range dependencies in the time series data. The dilation factor increases with the depth of the network, which exponentially expands the receptive field without increasing the number of parameters or computation. This design enables the model to capture temporal patterns over different time scales.

### Spatial Convolution Block:

The Spatial Convolution is carried out using a Graph Convolutional Network (GCN) based on a first-order Chebyshev approximation. In the context of an electrical grid, the abstract distance between different nodes is determined by the similarity of their load sequences. This spatial convolution process captures the spatial dependencies between different nodes in the grid, identifying patterns and relationships in the spatial domain.

In this project, the similarity relationships in the spatial domain are considered using Spearman Coefficient and Mutual Information to construct a meaningful graph structure upon which the GCN operates.

### Flow within Spatio-Temporal Convolution Blocks:

The flow of operations within a spatio-temporal convolution block is as follows:
1. **Temporal Convolution Block**: The first temporal convolution block processes the input data, focusing on extracting temporal features and dependencies.
2. **Spatial Convolution Block**: The spatial convolution block then processes the output of the temporal block, focusing on capturing spatial relationships and dependencies.
3. **Temporal Convolution Block**: Finally, another temporal convolution block processes the output of the spatial block to refine and capture any additional temporal dependencies before passing the information to the subsequent layers or blocks in the network.

The combination of temporal and spatial convolutions within these blocks allows the model to effectively capture and represent the complex spatio-temporal relationships inherent in the data.

