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
