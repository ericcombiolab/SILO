import pandas as pd
import numpy as np
import random
from IBD_functions import *
import time
import itertools
import multiprocessing
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-w","--winsize",type=int,default=10,help="the window size")
parser.add_argument("-l","--length",type=float,default=3,help="the length of block")
parser.add_argument("-ec","--error_common",type=float,default=0.005,help="the genotype error of common variants")
parser.add_argument("-el","--error_lowf",type=float,default=1e-5,help="the genotype error of low-frequency variants")
parser.add_argument("-t","--threshold",type=float,default=3,help="the threshold for LLR score")
parser.add_argument("--mode",type=str,help="Input 'inner' or 'outer', which stands for computing inner log-likelihood ratio or computing outer log-likelihood ratio, respectively",default='outer')
parser.add_argument("--union",type=int,help="calculate scores for low frequency variants of a pair of individuals that 1. either one carries minor alleles or 2. both two individuals carry minor alleles. Input 1 for case 1, or 2 for case 2",default=2)
parser.add_argument("--density",type=float,help="the minimum number of common SNPs in a block length of 1 cM",default=125)
parser.add_argument("--bin_length",type=float,help="the length of bins",default=2)
parser.add_argument("--N_simIBD",type=int,default=800,help="the number of simulated IBD pairs")
parser.add_argument("--N_simnonIBD",type=int,default=800,help="the number of simulated non-IBD pairs")
parser.add_argument("--negative_ratio_thres",type=float,default=0.1,help="the threshold for the negative ratio in generating IBD regions based on low frequency variants")
parser.add_argument("--negative_count_thres",type=int,default=1,help="the threshold for the negative count in generating IBD regions based on low frequency variants")
parser.add_argument("--pseudo",type=float,default=0.01,help="the pseudo-count used in empirical distribution for smoothing")
parser.add_argument("--threads",type=int,default=10,help="the number of threads used")
parser.add_argument("--include_clean",type=int,default=1,help="whether include the cleaning procedure or not. include_clean==0 means no including; include_clean==1 means including. By default include_clean == 1")
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--train_common",type=str,help="the path and filename (prefix) of common variants of the training set",required=True)
requiredNamed.add_argument("--train_lowf",type=str,help="the path and filename (prefix) of low-frequency variants of the training set",required=True)
requiredNamed.add_argument("--test_common",type=str,help="the path and filename (prefix) of common variants of the test set",required=True)
requiredNamed.add_argument("--test_lowf",type=str,help="the path and filename (prefix) of low-frequency variants of the test set",required=True)
requiredNamed.add_argument("--output",type=str,help="the path and filename (prefix) of output",required=True)
args = parser.parse_args()


start_time = time.time()

union = args.union
if union not in [1,2]:
    sys.exit("Error: the hyperparameter 'union' must be 1 or 2")

error_common = args.error_common
error_lowf = args.error_lowf

hap = pd.read_csv('{}.hap'.format(args.train_common),sep=' ')
hap.replace(-1,1,inplace=True) # convert missing values to 1 (major allele)
common_tped = pd.read_csv('{}.tped'.format(args.test_common),sep=' +',engine='python',header=None)
tfam = pd.read_csv('{}.tfam'.format(args.test_common),sep=' ',header=None)
common_frq = pd.read_csv('{}.frq'.format(args.train_common),sep=' +',engine='python')
lowf_tped = pd.read_csv('{}.tped'.format(args.test_lowf),sep=' +',engine='python',header=None)
lowf_frq = pd.read_csv('{}.frq'.format(args.train_lowf),sep=' +',engine='python')
lowf_frq_counts = pd.read_csv('{}.frq.counts'.format(args.train_lowf),sep=' +',engine='python')
df_map = pd.read_csv('{}.map'.format(args.train_common),sep='\t',header=None)
lowf_map = pd.read_csv('{}.map'.format(args.train_lowf),sep='\t',header=None)

win_size = args.winsize
num_nodes = args.threads
epsilon_common = float(error_common)
epsilon_lowf = float(error_lowf)
LLR_threshold = args.threshold
length = args.length # block length
num_markers = len(common_tped.index)
print('Generating windows')
windows_set = generate_windows(win_size,num_markers)
inter_time6 = time.time()
diff6 = inter_time6-start_time
print('Time spent for generating windows: ',diff6)

print('Generating blocks')
whole_blocks_set = generate_blocks(windows_set,length,common_tped,args.density)
inter_time7 = time.time()
diff7 = inter_time7-inter_time6
print('Time spent for generating blocks: ',diff7)

print('Computing frequencies')
freq_dic,df_pos = compute_freq(hap,windows_set,num_nodes,common_tped)
inter_time1 = time.time()
diff1 = inter_time1-inter_time7
print('Time spent for computing frequencies: ',diff1)

print('Encoding')
tmp_res_dic = {}
tmp_tped_set = [common_tped,lowf_tped]
tmp_frq_set = [common_frq,lowf_frq]
judge_set = [0,1]
df_geno_common = encoding(tmp_tped_set[0],tfam,tmp_frq_set[0],judge_set[0],num_nodes)[0] # dataframe of genotypes for each sample in the test set
df_geno_lowf,index_minor_dic = encoding(tmp_tped_set[1],tfam,tmp_frq_set[1],judge_set[1],num_nodes)

inter_time2 = time.time()
diff2 = inter_time2-inter_time1
print('Time spent for encoding: ',diff2)

if args.mode == 'outer':
    print('Doing internal simulation')
    geno_indi_IBD,geno_indi_nonIBD = internal_simulation(hap,args.N_simIBD,args.N_simnonIBD,df_map,length)
    if geno_indi_IBD.isnull().values.any():
        print(geno_indi_IBD.head(10))
        print(geno_indi_IBD.tail(10))
        sys.exit('geno_indi_IBD has NaN element(s)')
    elif geno_indi_nonIBD.isnull().values.any():
        print(geno_indi_nonIBD.head(10))
        print(geno_indi_nonIBD.tail(10))
        sys.exit('geno_indi_nonIBD has NaN element(s)')
    inter_time_outer1 = time.time()
    diff_outer1 = inter_time_outer1-inter_time2
    print('Time spent for doing internal simulation: ',diff_outer1)
    print('Generating empirical distribution')
    tmp_res_set = []
    pool = multiprocessing.Pool(num_nodes)
    for i in range(len(windows_set)):
        win_indi_IBD = geno_indi_IBD.iloc[windows_set[i],:]
        win_indi_nonIBD = geno_indi_nonIBD.iloc[windows_set[i],:]
        win_freq = freq_dic[i]
        num_hap_training = len(hap.columns)
        result = pool.apply_async(empirical_distribution,(win_indi_IBD,win_indi_nonIBD,args.bin_length,win_freq,epsilon_common,num_hap_training,args.pseudo,))
        tmp_res_set.append(result)
    pool.close()
    pool.join()

    df_empi_IBD_whole = pd.DataFrame([])
    df_empi_nonIBD_whole = pd.DataFrame([])
    for i in range(len(tmp_res_set)):
        async_res = tmp_res_set[i]
        df_empi_IBD_whole = pd.concat([df_empi_IBD_whole,async_res.get()[0]],ignore_index=True)
        df_empi_nonIBD_whole = pd.concat([df_empi_nonIBD_whole,async_res.get()[1]],ignore_index=True)
        if i == 0:
            num_bin = len(df_empi_IBD_whole.index) # get the number of bins
    inter_time_outer2 = time.time()
    diff_outer2 = inter_time_outer2-inter_time_outer1
    print('Time spent for generating empirical distribution: ',diff_outer2)

index_sample = list(range(len(tfam.index))) # the column indexes of all samples in the tped file
pair_set = list(itertools.combinations(index_sample,2)) # the list of all possible pairs

print('Computing score')
pool = multiprocessing.Pool(num_nodes)
indi_1_set = []
indi_2_set = []
window_index_set = []
common_score_set = []
lowf_score_set = []
score_set = []
tmp_res_set = []
num_hap_training = len(hap.columns)
for i in range(len(pair_set)):
    indi_1_index = pair_set[i][0]
    indi_2_index = pair_set[i][1]
    win_lowf_dic = allocate_rare2win(indi_1_index,indi_2_index,index_minor_dic,lowf_tped,df_pos,union)
    for j in range(len(windows_set)):
        common_geno_1_set = list(df_geno_common.iloc[windows_set[j],indi_1_index].values)
        common_geno_2_set = list(df_geno_common.iloc[windows_set[j],indi_2_index].values)
        win_freq = freq_dic[j]
        judge_lowf = 0
        if j in list(win_lowf_dic.keys()):
            judge_lowf = 1
            lowf_SNP_set = win_lowf_dic[j]
            lowf_geno_1_set = list(df_geno_lowf.iloc[lowf_SNP_set,indi_1_index].values)
            lowf_geno_2_set = list(df_geno_lowf.iloc[lowf_SNP_set,indi_2_index].values)
            minor_set_train,major_set_train,minor_set_test,major_set_test = compute_allele_num(lowf_frq_counts,lowf_tped,lowf_SNP_set)
        else:
            lowf_SNP_set = []
            lowf_geno_1_set = []
            lowf_geno_2_set = []
            minor_set_train = []
            major_set_train = []
            minor_set_test = []
            major_set_test = []
        if args.mode == 'inner':
            result = pool.apply_async(compute_score_v1,(common_geno_1_set,common_geno_2_set,win_freq,minor_set_train,major_set_train,minor_set_test,major_set_test,epsilon_common,epsilon_lowf,num_hap_training,lowf_SNP_set,lowf_geno_1_set,lowf_geno_2_set,judge_lowf,))
        elif args.mode == 'outer':
            df_empi_IBD = df_empi_IBD_whole.iloc[j*num_bin:(j+1)*num_bin,:]
            df_empi_nonIBD = df_empi_nonIBD_whole.iloc[j*num_bin:(j+1)*num_bin,:]
            result = pool.apply_async(compute_score_v2,(df_empi_IBD,df_empi_nonIBD,common_geno_1_set,common_geno_2_set,win_freq,minor_set_train,major_set_train,minor_set_test,major_set_test,epsilon_common,epsilon_lowf,num_hap_training,lowf_SNP_set,lowf_geno_1_set,lowf_geno_2_set,judge_lowf,))
        indi_1_set.append(tfam.iloc[indi_1_index,0])
        indi_2_set.append(tfam.iloc[indi_2_index,0])
        window_index_set.append(j)
        tmp_res_set.append(result)
pool.close()
pool.join()
for async_res in tmp_res_set:
    common_score_set.append(async_res.get()[0])
    lowf_score_set.append(async_res.get()[1])
    score_set.append(async_res.get()[2])

df = pd.DataFrame(indi_1_set)
df[1] = indi_2_set
df[2] = window_index_set
df[3] = common_score_set
df[4] = lowf_score_set
df[5] = score_set
df.columns = ['indi_1','indi_2','window_index','common_score','lowf_score','LLR']
df.to_csv('{}_window_scores.txt'.format(args.output),sep='\t',index=False)

inter_time3 = time.time()
if args.mode == 'inner':
    diff3 = inter_time3-inter_time2
elif args.mode == 'outer':
    diff3 = inter_time3-inter_time_outer2
print('Time spent for computing window scores: ',diff3)

indi_1_set = []
indi_2_set = []
block_index_set = []
block_windows_set = []
block_common_score_set = []
tmp_result_set = []
num_pairs = len(pair_set)
pool = multiprocessing.Pool(min(num_pairs,num_nodes))

for i in range(num_pairs):
    start_index = int(i*len(df.index)/num_pairs)
    indi_1 = df.loc[start_index,'indi_1']
    indi_2 = df.loc[start_index,'indi_2']
    score_set = df.loc[:,'common_score'].values
    result = pool.apply_async(generate_block_scores,(indi_1,indi_2,whole_blocks_set,start_index,score_set,))
    tmp_result_set.append(result)
pool.close()
pool.join()

for async_res in tmp_result_set:
    indi_1_set += async_res.get()[0]
    indi_2_set += async_res.get()[1]
    block_index_set += async_res.get()[2]
    block_windows_set += async_res.get()[3]
    block_common_score_set += async_res.get()[4]
new_df = pd.DataFrame(indi_1_set)
new_df[1] = indi_2_set
new_df[2] = block_index_set
new_df[3] = block_windows_set
new_df[4] = block_common_score_set
new_df.columns = ['indi_1','indi_2','block_index','window_index','common_score']
new_df.to_csv('{}_block_scores.txt'.format(args.output),sep='\t',index=False)

inter_time4 = time.time()
diff4 = inter_time4-inter_time3
print('Time spent for generating block scores: ',diff4)

gen_IBD_nodes = min(num_nodes,num_pairs)
pool = multiprocessing.Pool(gen_IBD_nodes)
tmp_res_set = []
for i in range(len(pair_set)):
    indi_1 = tfam.iloc[pair_set[i][0],0]
    indi_2 = tfam.iloc[pair_set[i][1],0]
    index1 = new_df.index[new_df.loc[:,'indi_1']==indi_1]
    index2 = new_df.index[new_df.loc[:,'indi_2']==indi_2]
    index = sorted(list(set(index1).intersection(set(index2))))
    sub_new_df = new_df.loc[index,:]
    sub_new_df.index = list(range(len(sub_new_df.index)))
    index3 = df.index[df.loc[:,'indi_1']==indi_1]
    index4 = df.index[df.loc[:,'indi_2']==indi_2]
    index_inter = sorted(list(set(index3).intersection(set(index4))))
    sub_df_window = df.loc[index_inter,:]
    result = pool.apply_async(generate_IBD_region,(sub_new_df,sub_df_window,windows_set,df_map,lowf_map,LLR_threshold,args.negative_ratio_thres,args.negative_count_thres,args.include_clean,))
    tmp_res_set.append(result)
pool.close()
pool.join()

indi_1_set = []
indi_2_set = []
lowf_start_pos_set_non = []
lowf_end_pos_set_non = []
all_start_pos_set = []
all_end_pos_set = []
for i in range(len(tmp_res_set)):
    async_res = tmp_res_set[i]
    check_IBD = async_res.get()[4]
    if check_IBD == 1:
        indi_1_set = indi_1_set + [async_res.get()[2]]*len(async_res.get()[0])
        indi_2_set = indi_2_set + [async_res.get()[3]]*len(async_res.get()[0])
        all_start_pos_set += async_res.get()[0]
        all_end_pos_set += async_res.get()[1]
        lowf_start_pos_set_non += async_res.get()[5]
        lowf_end_pos_set_non += async_res.get()[6]

if len(lowf_start_pos_set_non) > 0:
    lowf_start_pos_set_non = sorted(list(set(lowf_start_pos_set_non)))
    lowf_end_pos_set_non = sorted(list(set(lowf_end_pos_set_non)))

if len(all_start_pos_set) > 0:
    df_IBD = pd.DataFrame(indi_1_set)
    df_IBD[1] = indi_2_set
    df_IBD[2] = all_start_pos_set
    df_IBD[3] = all_end_pos_set
    df_map_combine = pd.concat([df_map,lowf_map],ignore_index=True)
    df_map_combine.index = df_map_combine.iloc[:,3].values
    chro = df_map_combine.iloc[0,0]
    for tmp_pos in lowf_start_pos_set_non:
        if tmp_pos-1 not in df_map_combine.iloc[:,3].values:
            df_map_combine.loc['{}_{}'.format(chro,tmp_pos-1)] = [chro,'{}_{}'.format(chro,tmp_pos-1),df_map_combine.loc[tmp_pos,2],tmp_pos-1]
    for tmp_pos in lowf_end_pos_set_non:
        if tmp_pos+1 not in df_map_combine.iloc[:,3].values:
            df_map_combine.loc['{}_{}'.format(chro,tmp_pos+1)] = [chro,'{}_{}'.format(chro,tmp_pos+1),df_map_combine.loc[tmp_pos,2],tmp_pos+1]
    df_map_combine.index = df_map_combine.iloc[:,3].values
    df_map_combine.sort_values(by=[3],inplace=True)
    all_start_cm_pos_set = df_map_combine.loc[all_start_pos_set,2].values
    all_end_cm_pos_set = df_map_combine.loc[all_end_pos_set,2].values
    df_IBD[4] = all_end_cm_pos_set-all_start_cm_pos_set
    df_IBD.columns = ['indi_1','indi_2','start_pos','end_pos','length(cM)']
    df_IBD.to_csv('{}_inferred_IBD_regions.txt'.format(args.output),sep='\t',index=False)
else:
    df_IBD = pd.DataFrame(columns=['indi_1','indi_2','start_pos','end_pos','length(cM)'])
    df_IBD.to_csv('{}_inferred_IBD_regions.txt'.format(args.output),sep='\t',index=False)

inter_time5 = time.time()
diff5 = inter_time5-inter_time4
print('Time spent for generating IBD regions: ',diff5)

end_time = time.time()
diff = end_time-start_time
print('Overall time spent: ',diff)


