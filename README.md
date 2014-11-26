# Crammer and Singer's Multi-Class Support Vector Machines

This is a new implementation of Crammer and Singer's Multi-Class Support Vector Machines. This algorithm solves multi-class problem as a single optimization problem instead of decomposing the multi-class problem into a series of binary problems like common algorithms do. This document explains the use of LibVM.

## Table of Contents

* [Installation and Data Format](#installation-and-data-format)
* ["mcsvm-offline" Usage](#mcsvm-offline-usage)
* ["mcsvm-cv" Usage](#mcsvm-cv-usage)
* [Tips on Practical Use](#tips-on-practical-use)
* [Examples](#examples)
* [Bibliography](#bibliography)
* [Additional Information](#additional-information)
* [Acknowledgments](#acknowledgments)

## Installation and Data Format[↩](#table-of-contents)

On Unix systems, type `make` to build the `mcsvm-offline` program. Run it without arguments to show the usage of it.

The format of training and testing data file is:
```
<label> <index1>:<value1> <index2>:<value2> ...
...
...
...
```

Each line contains an instance and is ended by a `'\n'` character (Unix line ending). For classification, `<label>` is an integer indicating the class label (multi-class is supported). For regression, `<label>` is the target value which can be any real number. The pair `<index>:<value>` gives a feature (attribute) value: `<index>` is an integer starting from 1 and `<value>` is the value of the attribute, which could be an integer number or real number. Indices must be in **ASCENDING** order. Labels in the testing file are only used to calculate accuracies and errors. If they are unknown, just fill the first column with any numbers.

A sample classification data set included in this package is `iris_scale` for training and `iris_scale_t` for testing.

Type `mcsvm-offline iris_scale iris_scale_t`, and the program will read the training data and testing data and then output the result into `iris_scale_t_output` file by default. The model file `iris_scale_model` will not be saved by default, however, adding `-s model_file_name` to `[option]` will save the model to `model_file_name`. The output file contains the predicted labels and the lower and upper bounds of probabilities for each predicted label.

## "mcsvm-offline" Usage[↩](#table-of-contents)
```
Usage: mcsvm-offline [options] train_file test_file [output_file]
options:
  -t redopt_type : set type of reduced optimization (default 0)
    0 -- exact (EXACT)
    1 -- approximate (APPROX)
    2 -- binary (BINARY)
  -k kernel_type : set type of kernel function (default 2)
    0 -- linear: u'*v
    1 -- polynomial: (gamma*u'*v + coef0)^degree
    2 -- radial basis function: exp(-gamma*|u-v|^2)
    3 -- sigmoid: tanh(gamma*u'*v + coef0)
    4 -- precomputed kernel (kernel values in training_set_file)
  -d degree : set degree in kernel function (default 1)
  -g gamma : set gamma in kernel function (default 1.0/num_features)
  -r coef0 : set coef0 in kernel function (default 0)
  -s model_file_name : save model
  -l model_file_name : load model
  -b beta : set margin (default 1e-4)
  -w delta : set approximation tolerance for approximate method (default 1e-4)
  -m cachesize : set cache memory size in MB (default 100)
  -e epsilon : set tolerance of termination criterion (default 1e-3)
  -z epsilon0 : set initialize margin (default 1-1e-6)
  -q : quiet mode (no outputs)
```
`train_file` is the data you want to train with.  
`test_file` is the data you want to predict.  
`mcsvm-offline` will produce outputs in the `output_file` by default.

## "mcsvm-cv" Usage[↩](#table-of-contents)
```
Usage: mcsvm-cv [options] data_file [output_file]
options:
  -t redopt_type : set type of reduced optimization (default 0)
    0 -- exact (EXACT)
    1 -- approximate (APPROX)
    2 -- binary (BINARY)
  -k kernel_type : set type of kernel function (default 2)
    0 -- linear: u'*v
    1 -- polynomial: (gamma*u'*v + coef0)^degree
    2 -- radial basis function: exp(-gamma*|u-v|^2)
    3 -- sigmoid: tanh(gamma*u'*v + coef0)
    4 -- precomputed kernel (kernel values in training_set_file)
  -d degree : set degree in kernel function (default 1)
  -g gamma : set gamma in kernel function (default 1.0/num_features)
  -r coef0 : set coef0 in kernel function (default 0)
  -v num_folds : set number of folders in cross validation (default 5)
  -b beta : set margin (default 1e-4)
  -w delta : set approximation tolerance for approximate method (default 1e-4)
  -m cachesize : set cache memory size in MB (default 100)
  -e epsilon : set tolerance of termination criterion (default 1e-3)
  -z epsilon0 : set initialize margin (default 1-1e-6)
  -q : turn off quiet mode (no outputs)
```
`data_file` is the data you want to do cross validation on.  
`mcsvm-cv` will produce outputs in the `output_file` by default.

## Tips on Practical Use[↩](#table-of-contents)
* Scale your data. For example, scale each attribute to [0,1] or [-1,+1].
* Try different kernels. Some data sets will not achieve good results on some kernels.
* Change parameters for better results.

## Examples[↩](#table-of-contents)
```
> mcsvm-offline train_file test_file output_file
```

Train a Crammer and Singer's multi-class classifier with default settings on `train_file`. Then conduct this classifier to `test_file` and output the results to `output_file`.

```
> mcsvm-offline -t 1 -s model_file train_file test_file
```

Train a Crammer and Singer's multi-class classifier with approximate reduced optimization on `train_file`. Then conduct this classifier to `test_file` and output the results to the default output file, also the model will be saved to file `model_file`.

```
> mcsvm-cv -v 10 data_file
```

Do a 10-folds cross validation of Crammer and Singer's multi-class classifiers on `data_file`. And output the results to the default output file.

## Bibliography[↩](#table-of-contents)

[1] Koby Crammer and Yoram Singer, 
    "On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines", 
    *Journal of Machine Learning Research*, 2001.

[2] Koby Crammer and Yoram Singer, 
    "Ultraconservative Online Algorithms for Multiclass Problems", 
    *Journal of Machine Learning Research*, 2003.

[3] Koby Crammer and Yoram Singer, 
    "On the Learnability and Design of Output Codes for Multiclass Problems", 
    *Machine Learning* 47, 2002. 

## Additional Information[↩](#table-of-contents)
For any questions and comments, please email [c.zhou@cs.rhul.ac.uk](mailto:c.zhou@cs.rhul.ac.uk)

## Acknowledgments[↩](#table-of-contents)
Special thanks to Koby Crammer [koby@ee.technion.ac.il](mailto:koby@ee.technion.ac.il), the author of the original implementation "MCSVM_1.0: C Code for Multiclass SVM", [http://webee.technion.ac.il/people/koby/](http://webee.technion.ac.il/people/koby/).