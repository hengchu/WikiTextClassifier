# cis700project

=======

### To obtain a copy of the data

1. Download the zip file here: https://drive.google.com/file/d/14eBl391iTjj-X6OLiQNyBUovIFcZ-Xs_/view?usp=sharing
2. Extract content to a directory
3. clone this repo somewhere
4. `cd cis700project`
5. `source ~/venv/bin/activate` This creates an isolated python environment, so anything we install here won't interfere with your system packages
6. `pip3 install --editable .`
7. `prepare-data -d dir/that/contains/data/files`
8. Verify that there is now a `joinedlonabstract_en.nt` in the data directory
