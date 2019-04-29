# cis700project

---

### Code organization

This repository hosts a python project containing the infrastructural code for
loading datasets and tokenizing texts that are shared by multiple exploration
notebooks.

In order to obtain the data and install the shared code, follow the instructions
in the next section.

Next, you should be able to load each notebook into a jupyter session, and step
through the cells.

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
