# fcma_exp

This repository contains the experiments for the paper describing
[FCMA](https://github.com/asi-uniovi/fcma).

## Installation

1. Clone the repository:

```bash
git clone https://github.com/asi-uniovi/fcma_exp.git
```
2. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```


2. Install the requirements:

```bash
pip install -r requirements.txt
```

## Usage

There are two set of experiments:
1. **Comparison with other algorithms**: This set of experiments compares FCMA with other
   algorithms. The results are saved in the file `data_comparison.csv`.
2. **Scalability analysis**: This set of experiments analyzes the scalability of FCMA. The
   results are saved in the file `data_scalability.csv`.

To run the experiments, execute the following commands:

```bash
python run_comparison.py
python run_scalability.py
```

The results will be saved in the aforementioned CSV files.

To generate the figures and tables, run the notebook
[analysis_comparison.ipynb](analysis_comparison.ipynb) for the comparison experiments and
[analysis_scalability.ipynb](analysis_scalability.ipynb) for the scalability experiments.

There's another file, `example.py`, that solves the problem presented as an example in the
paper. You can run it with:

```bash
python example.py
```
