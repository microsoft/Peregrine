# AutoToken

Right-sizing resource allocations for big-data queries, particularly
in serverless environments, is critical for improving infrastructure
operational efficiency, capacity availability, query performance predictability, 
and for reducing unnecessary wait times. For more details check the paper:

[_AutoToken: Predicting Peak Parallelism for Big Data Analytics at Microsoft_](http://www.vldb.org/pvldb/vol13/p3326-sen.pdf)<br>
Rathijit Sen, Alekh Jindal, Hiren Patel, Shi Qiao. VLDB 2020.

**AutoToken** is a simple and effective predictor for estimating the peak 
resource usage of recurring analytical jobs. It uses multiple
signatures to identify recurring job templates and learns simple,
per-signature models with the goal of reducing over-allocations for
future instances of those jobs. AutoToken is computationally light,
for both training and scoring, is easily deployable at scale, and is
integrated with the Peregrine workload optimization infrastructure.

[![AutoToken Video](https://img.youtube.com/vi/H61rl_kMHWI/hq1.jpg)](https://youtu.be/H61rl_kMHWI)
<br>

**Dataset Simulator.** This directory includes a dataset simulator to synthesize datasets of arbitrary size for AutoToken. It contains the following sub-directories.

Folder|Contents|
------|--------|
[datagen](datagen/README.md)| This contains scripts for analyzing distributions of feature values for an input dataset and generating synthetic datasets of a desired size with similar distributions.|
[distributions](distributions/README.md)| This contains distributions from which training and testing datasets can be generated for the AutoToken models.|
