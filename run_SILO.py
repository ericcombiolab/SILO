import pandas as pd
import numpy as np
from IBD_functions import *
import time
import itertools
import multiprocessing
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-w","--winsize",type=int,default=14,help="the window size")
parser.add_argument("-l","--length",type=float,default=4,help="the length of block")
parser.add_argument("-ec","--error_common",type=float,default=0.005,help="the genotype error of common variants")
parser.add_argument("-er","--error_rare",type=float,default=1e-3,help="the genotype error of rare variants")
parser.add_argument("-t","--threshold",type=float,default=3,help="the threshold for LLR score")
parser.add_argument("--mode",type=str,help="Input 'inner' or 'outer', which stands for computing inner log-likelihood ratio or computing outer log-likelihood ratio, respectively",default='outer')
parser.add_argument("--union",type=int,help="calculate scores for rare variants of a pair of individuals that 1. either one carries minor alleles or 2. both two individuals carry minor alleles. Input 1 for case 1, or 2 for case 2",default=2)
parser.add_argument("--density",type=float,help="the minimum number of common SNPs in a block length of 1 cM",default=125)
parser.add_argument("--bin_length",type=float,help="the length of bins",default=2)
parser.add_argument("--N_simIBD",type=int,default=800,help="the number of simulated IBD pairs")
parser.add_argument("--N_simnonIBD",type=int,default=800,help="the number of simulated non-IBD pairs")
parser.add_argument("--negative_ratio_thres",type=float,default=0.1,help="the threshold for the negative ratio in generating IBD regions based on rare variants")
parser.add_argument("--negative_count_thres",type=int,default=1,help="the threshold for the negative count in generating IBD regions based on rare variants")
parser.add_argument("--pseudo",type=float,default=0.01,help="the pseudo-count used in empirical distribution for smoothing")
parser.add_argument("--threads",type=int,default=10,help="the number of threads used")
parser.add_argument("--chunk_size",type=int,default=600,help="the chunk size for multiprocessing")
parser.add_argument("--include_clean",type=int,default=1,help="whether include the cleaning procedure or not. include_clean==0 means no including; include_clean==1 means including. By default include_clean == 1")
parser.add_argument("--seed_score_thres",type=float,default=-0.2,help="the threshold score of the seed (a window that contain the rare variant)")
parser.add_argument("--negative_thres",type=float,default=-0.3,help="the threshold of window common scores for negative count and negative ratio")
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--train_common",type=str,help="the path and filename (prefix) of common variants of the training set",required=True)
requiredNamed.add_argument("--train_rare",type=str,help="the path and filename (prefix) of rare variants of the training set",required=True)
requiredNamed.add_argument("--test_common",type=str,help="the path and filename (prefix) of common variants of the test set",required=True)
requiredNamed.add_argument("--test_rare",type=str,help="the path and filename (prefix) of rare variants of the test set",required=True)
requiredNamed.add_argument("--output",type=str,help="the path and filename (prefix) of output",required=True)
args = parser.parse_args()

start_time = time.time()

union = args.union
if union not in [1,2]:
    sys.exit("Error: the hyperparameter 'union' must be 1 or 2")

error_common = args.error_common
error_rare = args.error_rare

hap = pd.read_csv('{}.hap'.format(args.train_common),sep=' ')
hap.replace(-1,1,inplace=True) # convert missing values to 1 (major allele)
common_tped = pd.read_csv('{}.tped'.format(args.test_common),sep=' +',engine='python',header=None)
tfam = pd.read_csv('{}.tfam'.format(args.test_common),sep=' ',header=None)
common_frq = pd.read_csv('{}.frq'.format(args.train_common),sep=' +',engine='python')
rare_tped = pd.read_csv('{}.tped'.format(args.test_rare),sep=' +',engine='python',header=None)
rare_frq = pd.read_csv('{}.frq'.format(args.train_rare),sep=' +',engine='python')
rare_frq_counts = pd.read_csv('{}.frq.counts'.format(args.train_rare),sep=' +',engine='python')
df_map = pd.read_csv('{}.map'.format(args.train_common),sep='\t',header=None)
rare_map = pd.read_csv('{}.map'.format(args.train_rare),sep='\t',header=None)

win_size = args.winsize
num_nodes = args.threads
mode = args.mode
epsilon_common = float(error_common)
epsilon_rare = float(error_rare)
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
tmp_tped_set = [common_tped,rare_tped]
tmp_frq_set = [common_frq,rare_frq]
judge_set = [0,1]
df_geno_common = encoding(tmp_tped_set[0],tfam,tmp_frq_set[0],judge_set[0],num_nodes)[0] # dataframe of genotypes for each sample in the test set
df_geno_rare,index_minor_dic = encoding(tmp_tped_set[1],tfam,tmp_frq_set[1],judge_set[1],num_nodes)

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

    for i in range(len(tmp_res_set)):
        async_res = tmp_res_set[i]
        if i == 0:
            arr_empi_IBD_whole = async_res.get()[0]
            arr_empi_nonIBD_whole = async_res.get()[1]
            num_bin = arr_empi_IBD_whole.shape[0] # get the number of bins
        else:
            arr_empi_IBD_whole = np.append(arr_empi_IBD_whole,async_res.get()[0],axis=0)
            arr_empi_nonIBD_whole = np.append(arr_empi_nonIBD_whole,async_res.get()[1],axis=0)
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
rare_score_set = []
score_set = []
tmp_res_set = []
num_hap_training = len(hap.columns)
chunk_size = args.chunk_size
for i in range(len(pair_set)):
    indi_1_index = pair_set[i][0]
    indi_2_index = pair_set[i][1]
    win_rare_dic = allocate_rare2win(indi_1_index,indi_2_index,index_minor_dic,rare_tped,df_pos,union)
    num_chunk = max(1,len(windows_set)//chunk_size)
    for chunk_index in range(num_chunk):
        start_index = chunk_index*chunk_size
        if chunk_index == num_chunk-1:
            chunk_windows_set = windows_set[chunk_index*chunk_size:]
            end_index = len(windows_set)
        else:
            chunk_windows_set = windows_set[chunk_index*chunk_size:(chunk_index+1)*chunk_size]
            end_index = (chunk_index+1)*chunk_size
        indi_1_set.extend([tfam.iloc[indi_1_index,1]]*len(chunk_windows_set))
        indi_2_set.extend([tfam.iloc[indi_2_index,1]]*len(chunk_windows_set))
        window_index_set.extend(list(range(start_index,end_index)))
        flat_chunk_windows_set = list(itertools.chain.from_iterable(chunk_windows_set))
        df_geno_common_sub = df_geno_common.loc[flat_chunk_windows_set,[indi_1_index,indi_2_index]]
        df_geno_rare_sub = df_geno_rare.loc[:,[indi_1_index,indi_2_index]]
        arr_empi_IBD_sub = arr_empi_IBD_whole[start_index*(num_bin):end_index*(num_bin)]
        arr_empi_nonIBD_sub = arr_empi_nonIBD_whole[start_index*(num_bin):end_index*(num_bin)]
        result = pool.apply_async(compute_score_multiprocess,(chunk_windows_set,start_index,indi_1_index,indi_2_index,df_geno_common_sub,df_geno_rare_sub,
                               freq_dic,win_rare_dic,rare_frq_counts,rare_tped,mode,arr_empi_IBD_sub,arr_empi_nonIBD_sub,num_bin,
                               epsilon_common,epsilon_rare,num_hap_training,))
        tmp_res_set.append(result)
pool.close()
pool.join()
for async_res in tmp_res_set:
    for ele in async_res.get():
        common_score_set.append(ele[0])
        rare_score_set.append(ele[1])
        score_set.append(ele[2])

df = pd.DataFrame(indi_1_set)
df[1] = indi_2_set
df[2] = window_index_set
df[3] = common_score_set
df[4] = rare_score_set
df[5] = score_set
df.columns = ['indi_1','indi_2','window_index','common_score','rare_score','LLR']
df.to_csv('{}_window_scores.txt'.format(args.output),sep='\t',index=False)

inter_time3 = time.time()
if mode == 'inner':
    diff3 = inter_time3-inter_time2
else:
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
    indi_1 = tfam.iloc[pair_set[i][0],1]
    indi_2 = tfam.iloc[pair_set[i][1],1]
    index1 = new_df.index[new_df.loc[:,'indi_1']==indi_1]
    index2 = new_df.index[new_df.loc[:,'indi_2']==indi_2]
    index = sorted(list(set(index1).intersection(set(index2))))
    sub_new_df = new_df.loc[index,:]
    sub_new_df.index = list(range(len(sub_new_df.index)))
    index3 = df.index[df.loc[:,'indi_1']==indi_1]
    index4 = df.index[df.loc[:,'indi_2']==indi_2]
    index_inter = sorted(list(set(index3).intersection(set(index4))))
    sub_df_window = df.loc[index_inter,:]
    result = pool.apply_async(generate_IBD_region,(sub_new_df,sub_df_window,windows_set,df_map,rare_map,LLR_threshold,args.negative_ratio_thres,args.negative_count_thres,args.include_clean,args.seed_score_thres,args.negative_thres,))
    tmp_res_set.append(result)
pool.close()
pool.join()

indi_1_set = []
indi_2_set = []
rare_start_pos_set_non = []
rare_end_pos_set_non = []
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
        rare_start_pos_set_non += async_res.get()[5]
        rare_end_pos_set_non += async_res.get()[6]

if len(rare_start_pos_set_non) > 0:
    rare_start_pos_set_non = sorted(list(set(rare_start_pos_set_non)))
    rare_end_pos_set_non = sorted(list(set(rare_end_pos_set_non)))

if len(all_start_pos_set) > 0:
    df_IBD = pd.DataFrame(indi_1_set)
    df_IBD[1] = indi_2_set
    df_IBD[2] = all_start_pos_set
    df_IBD[3] = all_end_pos_set
    df_map_combine = pd.concat([df_map,rare_map],ignore_index=True)
    df_map_combine.index = df_map_combine.iloc[:,3].values
    chro = df_map_combine.iloc[0,0]
    for tmp_pos in rare_start_pos_set_non:
        if tmp_pos-1 not in df_map_combine.iloc[:,3].values:
            df_map_combine.loc['{}_{}'.format(chro,tmp_pos-1)] = [chro,'{}_{}'.format(chro,tmp_pos-1),df_map_combine.loc[tmp_pos,2],tmp_pos-1]
    for tmp_pos in rare_end_pos_set_non:
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


