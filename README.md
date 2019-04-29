# cis700project
---
### Background
Text classification is a difficult but highly real-world relevant task. Here, we explored various approaches to improve the classification of Wikipedia articles based on the first paragraph of the corresponding article. We obtained our data from the DBPedia Abstract Corpus (specifically the long abstract dataset). This dataset encapsulates approximately ~1.3 million Wikipedia articles. For each of these articles it includes a long abstract (the first paragraph of the article) as well as multiple levels of categorization of that article with varying degrees of specificity. For example, an article about the city Vienna in Austria could have a top-level label of geography, a next level label of cities, a next level label of Austria, etc.

Because of the complexity of each abstract, building a model to classify articles into categories is a non-trivial task. However, since in this dataset we have multiple labels for each article, we have additional information available that we can use to improve our article classifications. Specifically, we did the following. We first identified two levels of labels, i.e. fine and coarse labels, and discarded all other labels. In other words, each abstract in our dataset had a fine and a coarse label. We had a total of 180 coarse classes and 370 fine classes (see Exploratory Data Analysis). As a baseline, we trained logistic regression, LSTM, and Self-attention (Transformer encoder) models (see below for actual implementation) to classify the articles into fine and coarse labels (so 6 baselines in total). We then attempted to improve performance by the following approaches:
Method 1: Starting with a network trained to classify the articles into fine categories, we generated a new network able to classify the articles into coarser categories by simply retraining the top layer.
Method 2:  To improve performance on fine label classification, we first pre-trained the network on the coarse categories before continuing to train on fine categories.

---

### Code organization

This repository hosts a python project containing the infrastructural code for
loading datasets and tokenizing texts that are shared by multiple exploration
notebooks.

In order to obtain the data and install the shared code, follow the instructions
in the next section.

Next, you should be able to load each notebook into a jupyter session, and step
through the cells.

The organization for the ipynb is as follows:

`LogReg.ipynb`: Logistic Regression Baseline

`LSTM.ipynb`: LSTM Baseline

`Transformer.ipynb`: Self-attention baseline and Experiment 2 (Bootstrapping)

`Transformer_transferlearning.ipynb`:  Experiment 1 (Transfer Learning)

In addition to the datasets (which can be obtained below), LSTM.ipynb and Transformer_transferlearning.ipynb require the files catstats.csv and supercatstats.csv to be saved to your google drive. 
Some other relevant files:

`cis700/dataset.py`: code for loading data as a pytorch dataset

`cis700/prepare.py`: code for pre-processing DBPedia data into a format convenient for us

`cis700/tokenizer.py`: code for initializing BERT tokenizer (which we used only for tokenizing)

`cis700/utils.py`: various utility code to help training

`cis700/vocab/bert-base-uncased-vocab.txt`: BERT vocab file for initializing the tokenizer

---

### To obtain a copy of the data

1. Download the zip file here: https://drive.google.com/file/d/14eBl391iTjj-X6OLiQNyBUovIFcZ-Xs_/view?usp=sharing
2. Extract content to a directory
3. clone this repo somewhere
4. `cd cis700project`
5. `python3 -m venv ./venv`
6. `source ./venv/bin/activate` this creates an isolated python environment, so anything we install here won't interfere with your system packages
7. `pip3 install --editable .`
8. `prepare-data -d dir/that/contains/data/files`
9. Verify that there is now a `joinedlonabstract_en.nt` in the data directory

Step 5 only needs to be performed once, but step 6 needs to run everytime you start a new shell.

---

### Using BERT tokenizer

```
from cis700 import tokenizer

tok = tokenizer.build_tokenizer()
tokens = tok.tokenize('this is some arbitrary text data.')
# this is what we would pass to the network
ids = tok.convert_tokens_to_ids(tokens)
converted_back = tok.convert_ids_to_tokens(ids)
```

Alternatively, checkout the `main` function for `cis700/tokenizer.py`, and run
`tokenizer-demo` (after running `pip3 install --editable .`) to see it in action.
