# AutoToken

Right-sizing resource allocations for big-data queries, particularly
in serverless environments, is critical for improving infrastructure
operational efficiency, capacity availability, query performance predictability, 
and for reducing unnecessary wait times. For more details check the paper:

_AutoToken: Predicting Peak Parallelism for Big Data Analytics at Microsoft, VLDB 2020_<br>
_Rathijit Sen (Microsoft), Alekh Jindal (Microsoft), Hiren Patel (Microsoft), Shi Qiao (Microsoft)_

**AutoToken** is a simple and effective predictor for estimating the peak 
resource usage of recurring analytical jobs. It uses multiple
signatures to identify recurring job templates and learns simple,
per-signature models with the goal of reducing over-allocations for
future instances of those jobs. AutoToken is computationally light,
for both training and scoring, is easily deployable at scale, and is
integrated with the Peregrine workload optimization infrastructure.

This directory includes the following sub-directories.

Folder|Contents|
------|--------|
[datagen](datagen/README.md)| This contains scripts for analyzing distributions of feature values for an input dataset and generating synthetic datasets of a desired size with similar distributions.|
[distributions](distributions/README.md)| This contains distributions from which training and testing datasets can be generated for the AutoToken models.|




