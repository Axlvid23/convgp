# convgp:  Stanford and Geology Datasets
A forked version of the convgp github page to use additional datasets.  The first dataset is the Stanford NLP dataset that has some success with ConvNets and the other is a research dataset that uses hyperspectral data to approximate mineralogy.  At this time there is a compatibility issue with Tensorflow 1.0, which is required to run both convgp and GPflow with inter-domain support.

## Paper
The accompanying paper can be found on [arXiv](https://arxiv.org/abs/1709.01894).

## Setup and Directions
### GPflow with inter-domain support
In order to run the examples here, you need to have a branch of GPflow installed that supports inter-domain inducing
variables. The branch located [here](https://github.com/markvdw/GPflow-inter-domain) and can be installed following the
usual GPflow instructions. You can either use your favourite virtualenv to install the branch in, or switch back and
forth between the main and customised versions of GPflow by running their setup script. 
### Tensorflow 1.0
In order to run convgp with GPflow (inter-domain support), you will need to use tensorflow 1.0.

### Datasets: 

### My contribution:  CASERM and Stanford NLP Dataset

These datasets should be set up to run and begin to iterate using the Convolutional Gaussian Process SVM classification algorithm provided by the forked code from markvdw.

## Caserm:  Research data for geology.  Currently, this dataset will run and obtain a classification accuracy near 60% before crashing while trying to record.

## Stanford NLP:  

Code to perform NLP preprocessing was obtained from: [Convnets for sentence classification](https://github.com/davidsbatista/ConvNets-for-sentence-classification/blob/master/Stanford-Sentiment-Treebank-%20movie%20reviews%20with%20fine-grained%20labels.ipynb)

For the original convgp datasets, you will also need to setup the datasets. In `/datasets`, run:
 ```
 python process_cifar10.py
 python process_rectangles.py
 ```

### Running tests
Tests to check the correct functioning of most of the code is included. To run the tests use one of the two following:
```
python -m unittest
nosetests testing --nologcapture --with-coverage --cover-package=convgp --cover-html
```
The test coverage includes everything except the class `MultiOutputInducingSVGP`.

## Experiments
### Speed considerations
Running on the GPU is possible, and often significantly faster when using `float32` data. This reduced precision is fine
when using stochastic optimisation, but often problematic when using (variants of) BFGS. We recommend `float32` to be
used only with stochastic optimisation, and `float64` otherwise. This has to be adjusted manually in the `gpflowrc`file.

### Running experiments
## Axlvid Experiments:  You can run the following experiments I have added until they generate errors.
- `stanford.py`:  Stanford NLP Dataset (rbf, conv, weighted conv kernels)
- `caserm.py`: Geology Dataset (rbf, conv, weighted conv kernels)`


We have the following experiments:
 - `rectangles.py`: Rectangles dataset (rbf, conv, and weighted conv kernels).
 - `mnist01.py`: Zeros vs ones MNIST (rbf, conv, and weighted conv kernels).
 - `mnist.py`: Full multiclass MNIST (rbf, conv, and weighted conv kernels).
 - `sumkern_mnist.py`: Full multiclass MNIST (rbf + conv / wconv, rbf + poly + conv / wconv).
 
Many experiments have several command line options that can be used to modify a run. All have `--name`, which determines
the name of the file in `./results/` that stores the optimisation trace. Experiments are resumed if a file of the
correct name exists. Other options change the learning rate or minibatch size. See below for example experiments.

Optimisation traces can be displayed using `display.py`. The results files are passed as a positional argument, e.g.:
```
python display.py ./results/fullmnist*
```
#### Stanford (The hyperparameters may require adjustment to achieve maximum accuracy)
```
python stanford.py -k rbf -M 100
python stanford.py -k conv -M 50
python stanford.py -k wconv -M 50
```
#### Caserm (The hyperparameters may require adjustment to achieve maximum accuracy)
```
python caserm.py -k rbf -M 100
python caserm.py -k conv -M 50
python caserm.py -k wconv -M 50
```
#### Rectangles
```
python rectangles.py -k conv -M 16 --minibatch-size 100 -l 0.01 -n rectangles-paper  # Paper
python rectangles.py -k fullgp-rbf --optimiser l-bfgs-b -M 0  # Optimal RBF
python rectangles.py -k wconv -M 16 --minibatch-size 100 -l 0.01  # Full solution
python rectangles.py -k conv -M 35 --minibatch-size 100 -l 0.01  # Full support on the GP
python rectangles.py -k wconv -M 35 --minibatch-size 100 -l 0.01  # Idem
python rectangles.py -k wconv -M 200 --minibatch-size 100 -l 0.01 --dataset rectangles-image --Zinit patches
```
The results for the rectangles-image dataset aren't super impressive. Need to play around with the learning rate, and
perhaps other kernels (possibly additive).

#### Mnist 0 vs 1
```
python mnist01.py -k rbf -M 100
python mnist01.py -k conv -M 50
python mnist01.py -k wconv -M 50
```

#### Mnist
```
python mnist.py -k rbf -M 750 --learning-rate-block-iters=60000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200
python mnist.py -k conv -M 750 --learning-rate-block-iters=30000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200
python mnist.py -k wconv -M 750 --learning-rate-block-iters=30000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200
python sumkern_mnist.py -k1 wconv -k2 rbf -M 750 --vardist full --learning-rate-block-iters=20000 --learning-rate "0.001 * 10**-(i // b / 3) --minibatch-size 200
```
The learning rate decay of the sum kernel experiment is set too aggressively for convergence of the variational
objective function. However, this rate was chosen as it repeatably converges to the (near-optimal) performance reported
in the paper with 24 hours of time on a GTX1080. We also ran the experiment for several times longer, which showed
little improvement in performance and no signs of over-fitting.

#### CIFAR-10
```
python cifar.py -k wconv -M 1000 --minibatch-size 50
python cifar.py -k multi -M 1000 --minibatch-size 30
python cifar.py -k addwconv -M 1000 --minibatch-size 30
```


## Reproducing the plots from the paper
For reference, you can download the pickled optimisation histories for the results in the paper for
[full MNIST (223 MB)](http://mlg.eng.cam.ac.uk/mvdwilk/convgp/convgp-fullmnist-results.tar.gz) and for
[CIFAR-10 (408 MB)](http://mlg.eng.cam.ac.uk/mvdwilk/convgp/convgp-cifar10-results.tar.gz). SHA checksums can be found
in`./results/`.

After running the above experiments, you can run `python paper-plots.py` to recreate the figures from the paper.

## Notes on the code
While the repositories for `gpflow-inter-domain` and `convgp` are separate, they rely on some modifications in each
other. The most non-elegant adaptation to GPflow is to allow variables internal to the TensorFlow optimiser to be
restored through opt_tools. The whole set up is a bit less than ideal, it would probably be better to use the internal
TensorFlow loading and storing mechanisms, but this would require larger edits to GPflow.
