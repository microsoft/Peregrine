Distributions:
--------

There are five sub-directories in this directory, Trace1 to Trace5. Each of those directories contains two sub-directories: training_distributions and testing_distributions, which contain statistical properties of a subset of recurring job pipelines in the _Cosmos big data processing infrastructure at Microsoft_.
Together, there are distributions for **9,561** recurring job pipelines (5,000 for training and 4,561 for testing).

Each of the distributions directories contains two csv files: distributions.csv and header.csv. The header.csv file contains the names of 15 attributes for each job that we simulate. The format of the distributions.csv files is as follows.

Field Number| Field Description|
------------|------------------|
1| primary signature number|
2| list of mean values for each of the attributes|
3| list of standard deviation values for each of the attributes|
4| covariance matrix for the attributes|
5| list of dependent attributes|
6| list of attribute numbers with integer types|

Each row in the distributions.csv file correspond to properties of a group of jobs having the same primary signature number in the input dataset.

The ../datagen directory contains scripts to generate simulated traces of arbitrary size from the above distributions. It also contains a script that shows how the distributions were extracted from the original traces.

The following steps describe how to generate the datasets.\
        cd ../datagen\
        ./datagen.sh <trace> <mode> <size>\
    for example, ./datagen.sh Trace3 training 1000. Refer to the README in the ../datagen directory for more information.\
    Run the datagen script for each of the traces Trace1, ..., Trace5 and for both training and testing. This will generate the training.csv and testing.csv files in the corresponding Trace sub-directories in this (distributions) directory.


### Terminology:
* A _job_ a dataflow script consisting of DQL-like statements that get compile into a single DAG of operators with one or more inputs and producing one or more inputs and one or more outputs.
* Clusters are logically partitioned into _Virtual Clusters_ (VC). A job will start to run on the VC it is submitted to when the requested number of tokens is available on that VC. 
* A _hashtag_ or a _signature_ is a hash of the operators in the logical query plan of the job, 
while excluding the parameters and inputs.


### Schema:
Each line in the training and testing csv files contains information about a single job. The schema is described as follows.

Field Number|Field Name| Field Type| Field Description|
------------|----------|-----------|------------------|
1| HT1 | Integer| primary signature number|
2| VC | Integer| virtual cluster number|
3| HT2 | Integer| secondary signature component_1 number|
4| HT3 | Integer| secondary signature component_2 number|
5| EstCardinality | Integer| estimated cardinality of job output|
6| InputCardinality | Integer| aggregate data size (sum) of the leaf inputs|
7| InputChildrenCardinality | Integer| estimated aggregate (sum) of input cardinalities of children operators of the job root|
8| AvgRowLength | Double| average length (bytes) of input rows|
9| EstCost | Double| estimated aggregate cost (sum) at the job root|
10| EstExclusiveCost | Double| estimated cost of the root operator|
11| VertexCount | Integer| number of vertices in the job graph|
12| RequestedTokens | Integer| number of guaranteed tokens requested (default allocation) for the job|
13| ActualMaxTokens | Integer| maximum number of tokens used by the job|
14| SubmitOffset | Integer| time offset (seconds) when job is submitted|
15| WaitTime | Integer| time in seconds between job start and job submit|
16| RunTime | Integer|  time in seconds between job end and job start|


Notes:
-----

* Each line in the datasets (after the header) describes characteristics of one job.
* Except HT1, the values of all the remaining fields are generated through simulation using the provided distributions. The HT1 values are comparable across the training and testing datasets within each trace, but not across traces.
* The HT1 value forms the primary signature referred to in the AutoToken paper. These are not the actual numeric hashes, but unique integers with a 1:1 mapping to the corresponding hash value.
* The HT2 and HT3 values correspond to the re-mapped hash values of the job inputs and root operator respectively. The secondary signature referred to in the AutoToken paper is constructed by combining the HT2 and HT3 numbers. Refer to the code in the src directory for details.
* Fields 5--11 are the compile-time characteristics used as features for the AutoToken models.
* Field 13 is known after the job has run and is used as the label for the AutoToken models.
* The time offsets within each trace are computed from a pre-determined point in time for the trace. Time offsets across traces are not comparable.
