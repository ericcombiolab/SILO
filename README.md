# SILO
**SILO** is a Python based commond line tool that identifies IBD segments. It could not only detect long IBD segments but also short IBD segments (length < 2 centimorgan) by incorporating rare variants. Additionally, SILO can identify potential disease susceptible variants by detecting IBD segments among affected individuals.

Citation: Chonghao Wang, Werner Pieter Veldsman, Lu Zhang. Detection of short identity by descent segments using low-frequency variants. *bioRxiv*, https://doi.org/10.1101/2023.09.26.559464.
## Usage
```
python3 run_SILO.py --winsize INT --length FLOAT --error_common FLOAT --error_rare FLOAT --threshold FLOAT --mode STR --union INT  
--density FLOAT --bin_length FLOAT --N_simIBD INT --N_simnonIBD INT --negative_ratio_thres FLOAT --gative_count_thres INT  
--chunk_size INT --include_clean INT --seed_score_thres FLOAT --negative_thres FLOAT --pseudo FLOAT --threads INT
--train_common STR --train_rare STR --test_common STR --test_rare STR --output STR  

optional arguments:
  -w, --winsize: the number of common variants in a window. The default value is 14.  
  -l, --length: the length (centimorgan) of a block. The default value is 3.  
  -ec, --error_common: the genotype error of common variants. The default value is 0.005.  
  -er, --error_rare: the genotype error of rare variants. The default value is 1e-3.  
  -t, --threshold: the threshold for LLR score. The default value is 3.  
  --mode: input 'inner' or 'outer', which stands for computing inner log-likelihood ratio or computing outer log-likelihood ratio,  
  respectively. The default value is 'outer'.  
  --union: calculate scores for rare variants of a pair of individuals that 1. either one carries minor alleles or 2.  
  Both two individuals carry minor alleles. Input 1 for case 1, or 2 for case 2. The default value is 2.  
  --density: the minimum number of common SNPs in a block length of 1 cM. The default value is 125.  
  --bin_length: the length of bins. The default value is 2.  
  --N_simIBD: the number of simulated IBD pairs. The default value is 800.  
  --N_simnonIBD: the number of simulated non-IBD pairs. The default value is 800.  
  --negative_ratio_thres: the threshold for the negative ratio in generating IBD regions based on rare variants. The  
  default value is 0.1.  
  --negative_count_thres: the threshold for the negative count in generating IBD regions based on rare variants. The  
  default value is 1.  
  --pseudo: the pseudo-count used in empirical distribution for smoothing. The default value is 0.01.
  --include_clean: whether include the cleaning procedure or not. include_mean==0 means no including; include_mean==1 means including. By default include_mean==1.
  --threads: the number of threads used. The default value is 10.
  --chunk_size: the chunk size for multiprocessing. The default value is 600.
  --seed_score_thres: the threshold score of the seed (a window that contain the rare variant). The default value is -0.2.
  --negative_thres: the threshold of window common scores for negative count and negative ratio. The default value is -0.3.
  
required arguments
  --train_common: the path and filename (prefix) of common variants of the training set (i.e. population set).  
  --train_rare: the path and filename (prefix) of rare variants of the training set.  
  --test_common: the path and filename (prefix) of common variants of the test set.  
  --test_rare: the path and filename (prefix) of rare variants of the test set.  
  --output: the path and filename (prefix) of output.  
```

## Quick start
- Download the IBD_functions.py and run_SILO.py and put them in one folder. run_SILO.py will load the functions defined in IBD_functions.py.  
- SILO requires Python packages **scipy**, **numpy**, and **pandas**.
- Download the zipped file sample_data.zip and put it in the same folder with run_SILO.py. Unzip it and run the following command:
```
python3 run_SILO.py \
--winsize 14 \
--length 3 \
--error_common 0.005 \
--error_rare 0.001 \
--threads 10 \
--train_common ./sample_data/common_variants/training_common \
--train_rare ./sample_data/rare_variants/training_rare \
--test_common ./sample_data/common_variants/testing_common \
--test_rare ./sample_data/rare_variants/testing_rare \
--output ./output  
```

## File format
### Input
Common variants of the training set: .hap, .map, .frq  
Common variants of the testing set: .tped, .tfam  
Rare variants of the training set: .map, .frq, .frq.counts  
Rare variants of the testing set: .tped  
All file formats except .hap could be found at https://www.cog-genomics.org/plink/2.0/formats#haps.  
.hap, .tped, .tfam, .frq, .frq.counts are space-delimited, while .map is tab-delimited.  
The file format of .hap is as follow:  
```
haplotype1.1 haplotype1.2 haplotype2.1 haplotype2.2
1 0 0 1
1 1 1 0
0 0 -1 1
```
'0' stands for the minor allele, and '1' stands for the major allele. '-1' stands for the missing value. Each column in the .hap is a haplotype. The first two columns are the haplotypes for the first person, and the third and fourth columns are the haplotypes for the second person. The first row is the haplotype name, and the remaining rows stand for SNPs. The corresponding order of these remaining rows is the same as the order in .map file. Each column is seperated by a space. 
### Output
#### Inferred IBD regions (<OUTPUT_PREFIX>_inferred_IBD_regions.txt)
The output file is a tab-delimited text file. Each row stands for an IBD region between two individuals. There are 5 columns in total. The first column is the ID of one individual; the second column is the ID of another individual; the third column is the starting position (base-pair coordinates) of each IBD region; the fourth column is the ending position (base-pair coordinates) of each IBD region; the fifth column is the length (centimorgan) of each IBD region.  
```
indi_1  indi_2  start_pos  end_pos  length(cM)
```
#### Window score (<OUTPUT_PREFIX>_window_scores.txt)
The output file is a tab-delimited text file. Each row stands for a window between two individuals. There are 6 columns in total. The first column is the ID of one individual; the second column is the ID of another individual; the third column is the index of each window; the fourth column is the log-likelihood ratio (LLR) of common variants of each window; the fifth column is the LLR ratio of the rare variant (if there are more than one rare variants, then it is the summation of their LLRs) in each window; the sixth column is the summation of LLRs of the fourth and fifth column.  
```
indi_1  indi_2  window_index  commmon_score  rare_score  LLR
```
#### Block score (<OUTPUT_PREFIX>_block_scores.txt)
The output file is a tab-delimited text file. Each row stands for a block between two individuals. There are 5 columns in total. The first column is the ID of one individual; the second column is the ID of another individual; the third column is the index of each block; the fourth column is the indices of windows that are included in each block; the fifth column is the LLR of common variants of each block.  
```
indi_1  indi_2  block_index  window_index  common_score
```
