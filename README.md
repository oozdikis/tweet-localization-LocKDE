

#### Introduction
This code is used to predict geographical locations for tweets using their text. The method determines different settings of kernel functions for every term in tweets based on the location
indicativeness of these terms. It also identifies bigrams with significant spatial patterns in training set and extends the feature space using these bigrams. The code is an implementation of LocKDE-SCoP published in: 

*O. Ozdikis, H. Ramampiaro and K. Nørvåg, (2019), Locality-Adapted Kernel Densities of Term Co-occurrences for Location Prediction of Tweets. In Information Processing & Management (IPM). (Accepted for publication)*



#### Dependencies
The code was tested with Python 2.7 and libraries below: 
* geopy (v1.11.0)
* numpy (v1.10.2)
* scipy (v0.19.1)
* shapely (v1.5.13)


#### Input files
The paths to input files should be defined in `data.__init__.py`.

* Grid definition: The grid that divides the region of interest into smaller grid cells should be defined in a file at `data.grid_file`. Each line in this file is expected to represent a grid cell with an id and coordinates of its South-West and North-East corners. **Example:** For a 2x2 grid spanning the region bounded by (10.0, 10.0) in South-West and (20.0, 20.0) in North-East, the content of `grid_file` should contain the following four lines:  

```
0	10.0	10.0	15.0	15.0
1	15.0	10.0	20.0	15.0
2	10.0	15.0	15.0	20.0
3	15.0	15.0	20.0	20.0
```

<a href='images/grid.png'>see visual</a>

* Training data: Training data should be provided in a file located at `data.training_file`. Each line in this file is expected to represent a tweet text and its location. First column represents the grid cell corresponding to the latitude (second column) and longitude (third column) of the tweet location. The last column contains the tokens in tweet text separated by space. **Example:**  

```
1	16.0	11.0	lorem ipsum dolor sit amet
2	12.0	17.0	consectetur adipiscing elit
0	12.0	12.0	sed do eiusmod tempor incididunt ut labore et dolore magna aliqua
```


* Test data: It has the same structure as the training file. The lines in this file are used to test the classifier that is trained using the training file.   


#### Running the program

* Step 1: Set paths of input files in `data.__init__.py`
* Step 2: Run `cooc.main_cooc` in Python.
It finds the bigrams in training data with attraction and repulsion patterns, and writes them to `data.kscore_analysis_file`.

* Step 3: Run `prediction.main_prediction` in Python.
It predicts locations for tweet texts in `test_file`, and prints the median of error distances between the estimated coordinates and the expected coordinates according to the ground truths in test file. 



