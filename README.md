# Classification with Strategically Withheld Data
This repo reproduces the paper [Classification with Strategically Withheld Data](https://arxiv.org/abs/2012.10203). The paper appeared in [AAAI 2021](https://aaai.org/Conferences/AAAI-21/) and [IML 2020](https://gradanovic.github.io/incentives_in_ML_icml2020_ws/). This repo requires Python 3.7.

### Downloading the dataset

3 of the 4 datasets are directly retrieved from the internet. One of the dataset needs to be downloaded from the [UCI ML Repo](https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip).

### Running the experiment
To run the experiment
```
python credit.py <dataset> <fraction> <balanced-flag>
```
where `<dataset>` can be "australia", "germany", "poland" or "taiwan", specifying the dataset to run the experiment on; `<fraction>` can be any value between 0 and 1 (in our work, we do 0.0, 0.1, 0.2, 0.3, 0.4, 0.5), specifying the fraction of data missing for non-strategic reasons; `<balanced-flag>` can be "bal" or blank, specifying whether or not to balance the dataset before experiement.

To reproduce the paper, run with each of the 4x6x2=48 specifications.

### Plotting the results
To generate all tables and figures in tex and png:
```
python parse.py
```
