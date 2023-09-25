import numpy as np
import pandas as pd
import ast
import multiprocessing
from itertools import product
from itertools import chain
import scipy.integrate as integrate
import scipy
import math
import sys
import warnings
from scipy.stats import norm 

warnings.filterwarnings("error")

########################################################################################
### This part is for common SNPs

def generate_windows(win_size,num_markers):
    # win_size: the size of a window
    # num_markers: the total number of commonm SNPs in the file
    num_windows = float(num_markers/win_size)
    if num_windows-int(num_windows) >= 0.5:
        num_windows = int(num_windows)+1
    else:
        num_windows = int(num_windows)
    windows_set = []
    for i in range(num_windows):
        if i != num_windows-1:
            window = list(range(i*win_size,(i+1)*win_size))
            windows_set.append(window)
        else:
            window = list(range(i*win_size,num_markers))
            windows_set.append(window)
    return windows_set

def generate_blocks(windows_set,length,common_tped,density):
    # windows_set: the set of windows
    # length: the length of blocks (in centiMorgan)
    # common_tped: the dataframe of .tped file of common SNPs
    # density: SNP density threshold of a block, i.e. the minimum number of common SNPs in a block of length 1 cM
    judge_process = 0 # This means our building blocks process should continue. If judge_process == 1, then we need to stop
    # find the index of first marker in the theoretical last block
    index = list(common_tped.index[common_tped.iloc[:,2]<=common_tped.iloc[-1,2]-length])[-1]
    window_index = int(index/len(windows_set[0])) # the corresponding index of the window that the marker belongs to
    start_snp_index_set = []
    end_snp_index_set = []
    for i in range(len(windows_set)):
        if i <= window_index:
            start_snp_index_set.append(windows_set[i][0])
        end_snp_index_set.append(windows_set[i][-1])
    start_snp_cm_set = common_tped.iloc[start_snp_index_set,2].values # the list of starting positions (cM) of each window
    end_snp_cm_set = common_tped.iloc[end_snp_index_set,2].values # the list of ending positions (cM) of each window
    min_end_snp_cm_set = start_snp_cm_set+length # the list of minimum ending positions (cM) of each block. Here, 'minimum' means that the final ending positions should be equal or larger to these minimum values
    df = pd.DataFrame(end_snp_cm_set)
    whole_blocks_set = []
    for i in range(len(min_end_snp_cm_set)):
        min_end = min_end_snp_cm_set[i]
        index1 = df.index[df.loc[:,0]>=min_end][0]
        if len(windows_set[0])*len(list(range(i,index1+1)))/(df.loc[index1,0]-(min_end-length)) >= density: # check SNP density of a block
            whole_blocks_set.append(list(range(i,index1+1)))
    whole_blocks_set = list(map(list,sorted(set(map(tuple,whole_blocks_set)))))
    return whole_blocks_set

def tmp_compute_freq(i,sub_df_hap):
    # This function is for multiprocessing in compute_freq function
    sub_freq_dic = {}
    for j in range(len(sub_df_hap.columns)):
        hap = list(sub_df_hap.iloc[:,j].values)
        if str(hap) not in list(sub_freq_dic.keys()):
            sub_freq_dic[str(hap)] = 1/len(sub_df_hap.columns)
        else:
            sub_freq_dic[str(hap)] += 1/len(sub_df_hap.columns)
    # remove haplotypes that have very low frequencies (here we remove frq < 4/len(sub_df_hap.columns))
    tmp_sub_freq_dic = {key:val for key, val in sub_freq_dic.items() if val >= 4/len(sub_df_hap.columns)}
    if len(tmp_sub_freq_dic.keys())==0: # In case we do not have haplotypes with frequency larger than 4/len(sub_df_hap.columns) 
        sub_freq_dic = {key:val for key, val in sub_freq_dic.items() if val >= 2/len(sub_df_hap.columns)}
    else:
        sub_freq_dic = {key:val for key, val in sub_freq_dic.items() if val >= 4/len(sub_df_hap.columns)}
    return sub_freq_dic

def compute_freq(df_hap,windows_set,num_nodes,common_tped):
    # df_hap: the dataframe of haplotypes
    # windows_set: the set of indexs of SNPs for each window
    # num_nodes: the number of nodes for multiprocessing
    # common_tped: the dataframe of .tped file for common SNPs
    freq_dic = {}
    if num_nodes > len(windows_set):
        num_nodes = len(windows_set)
    pool = multiprocessing.Pool(num_nodes)

    start_pos_set = []
    end_pos_set = []
    for i in range(len(windows_set)):
        start_pos = common_tped.iloc[windows_set[i][0],3]
        end_pos = common_tped.iloc[windows_set[i][-1],3]
        start_pos_set.append(start_pos)
        end_pos_set.append(end_pos)
        #freq_dic[i] = tmp_compute_freq(i,df_hap,windows_set)
        freq_dic[i] = pool.apply_async(tmp_compute_freq,(i,df_hap.iloc[windows_set[i],:],))
    pool.close()
    pool.join()
    for i in range(len(windows_set)):
        freq_dic[i] = freq_dic[i].get()
    df_pos = pd.DataFrame(start_pos_set)
    df_pos[1] = end_pos_set

    # change two places, which does not affect the final result and could be helpful in assigning rare SNPs to windows (e.g. the first rare SNP's location is before the first window)
    df_pos.iloc[0,0] = 0
    df_pos.iloc[len(df_pos.index)-1,1] = 2*df_pos.iloc[len(df_pos.index)-1,1] # times 2 is just to make this large enough to include the rare SNPs that are beyond domain of the last window
    return freq_dic,df_pos

def tmp_encoding(j,tped,frq,judge):
    # This is a function for multiprocessing in encoding function

    index_minor = [] # the index of minor alleles (if any) for rare SNPs
    if judge == 0: # This is for common SNPs
        index1 = tped.index[tped.iloc[:,4+2*j]==frq.loc[:,'A1'].values]
        index2 = tped.index[tped.iloc[:,4+2*j]==frq.loc[:,'A2'].values]
        index3 = sorted(list(set(list(range(len(tped.index))))-set(list(index1)+list(index2)))) # the indices of missing values
        tped.iloc[index1,4+2*j] = 0
        tped.iloc[index2,4+2*j] = 1
        tped.iloc[index3,4+2*j] = -4.5

        index1 = tped.index[tped.iloc[:,4+2*j+1]==frq.loc[:,'A1'].values]
        index2 = tped.index[tped.iloc[:,4+2*j+1]==frq.loc[:,'A2'].values]
        index3 = sorted(list(set(list(range(len(tped.index))))-set(list(index1)+list(index2)))) # the indices of missing values
        tped.iloc[index1,4+2*j+1] = 0
        tped.iloc[index2,4+2*j+1] = 1
        tped.iloc[index3,4+2*j+1] = -4.5

        geno = tped.iloc[:,4+2*j].values+tped.iloc[:,4+2*j+1].values

    elif judge == 1: # This is for rare SNPs
        index1 = tped.index[tped.iloc[:,4+2*j]==frq.loc[:,'A1'].values]
        index2 = tped.index[tped.iloc[:,4+2*j]==frq.loc[:,'A2'].values]
        index3 = sorted(list(set(list(range(len(tped.index))))-set(list(index1)+list(index2)))) # the indices of missing values
        tped.iloc[index1,4+2*j] = 0
        tped.iloc[index2,4+2*j] = 1
        tped.iloc[index3,4+2*j] = -4.5

        index4 = tped.index[tped.iloc[:,4+2*j+1]==frq.loc[:,'A1'].values]
        index5 = tped.index[tped.iloc[:,4+2*j+1]==frq.loc[:,'A2'].values]
        index6 = sorted(list(set(list(range(len(tped.index))))-set(list(index4)+list(index5)))) # the indices of missing values
        tped.iloc[index4,4+2*j+1] = 0
        tped.iloc[index5,4+2*j+1] = 1
        tped.iloc[index6,4+2*j+1] = -4.5

        geno = tped.iloc[:,4+2*j].values+tped.iloc[:,4+2*j+1].values
        index_minor = sorted(list(set(list(index1)+list(index4))))

    result = [geno,index_minor]
    return result

def encoding(tped,tfam,frq,judge,num_nodes):
    # tped: dataframe of .tped file of the test set
    # tfam: dataframe of .tfam file of the test set
    # frq: dataframe of .frq file of the training set
    # judge: binary value (0 or 1). If it's 0, we are dealing with common SNPs. If it's 1, we are dealing with rare SNPs
    # num_nodes: the number of nodes for multiprocessing

    tmp_dic = {}
    if num_nodes > len(tfam.index):
        num_nodes = len(tfam.index)
    pool = multiprocessing.Pool(min(num_nodes,len(tfam.index)))
    num_samples = len(tfam.index) # the number of individuals in the test set
    tped_copy = tped.copy()
    for j in range(num_samples):
        tmp_dic[j] = pool.apply_async(tmp_encoding,(j,tped_copy,frq,judge,))
    pool.close()
    pool.join()
    list1 = []
    index_minor_dic = {} # a dictionary links the person and its minor allele distribution in rare SNPs
    for i in range(num_samples):
        list1.append(tmp_dic[i].get()[0])
        index_minor_dic[i] = tmp_dic[i].get()[1]
    df = pd.DataFrame(list1)
    df = df.T
    df.columns = tfam.index
    result = [df,index_minor_dic]
    return result

def allocate_rare2win(indi_1,indi_2,index_minor_dic,rare_tped,df_pos,union):
    # This function allocates rare SNPs in which at least one individual has minor alleles to windows of common SNPs
    # Input:
    # indi_1: the index of individual 1
    # indi_2: the index of individual 2
    # index_minor_dic: the dictionary which links the individual and the index of rare SNPs in which the individual has minor alleles
    # rare_tped: the dataframe of .tped file for rare SNPs
    # df_pos: the dataframe of start and end position (base-pair coordinates) for each window
    # union: a hyperparameter
    # Output:
    # win_rare_dic: the dictionary that links the window and its associated rare SNPs (if any)

    snp_index_1 = index_minor_dic[indi_1] # the list of indexes of the rare SNPs
    snp_index_2 = index_minor_dic[indi_2] # the list of indexes of the rare SNPs
    if union == 1:
        rare_set = sorted(list(set(snp_index_1).union(set(snp_index_2)))) # the union of snp_index_1 and snp_index_2
    elif union == 2:
        rare_set = sorted(list(set(snp_index_1).intersection(set(snp_index_2))))
    tmp_rare_snp_set = [] # the list consists of indices of low frequency variants that indi_1 or indi_2 have minor allele(s)
    tmp_window_set = [] # the list consists of indices of windows that have low frequency variants
    win_rare_dic = {}
    if len(rare_set) > 0:
        for index in rare_set:
            pos = rare_tped.iloc[index,3] # the base-pair coordinate of the SNP 
            index1 = list(df_pos.index[df_pos.iloc[:,0]<pos])
            index2 = list(df_pos.index[df_pos.iloc[:,1]>pos])
            intersection = list(set(index1).intersection(set(index2))) # the index of the window that is supposed to include the rare SNP
            if len(intersection) == 0: # In this case, the rare SNP is outside the window
                index1 = list(df_pos.index[df_pos.iloc[:,0]>pos])[0]-1 # the index of window that is just before the rare SNP
                index2 = index1+1
                if np.abs(df_pos.iloc[index1,1]-pos) >= np.abs(df_pos.iloc[index2,0]-pos):
                    intersection = [index2]
                else:
                    intersection = [index1]
            intersection = intersection[0]
            tmp_window_set.append(intersection)
            tmp_rare_snp_set.append(index)
        # create the dictionary
        check_duplicates_set = [] # the list store the indices of windows
        store_set = []
        for i in range(len(tmp_window_set)):
            if i == 0:
                check_duplicates_set.append(tmp_window_set[i])
                store_set.append(tmp_rare_snp_set[i])
                if len(tmp_window_set) == 1:
                    win_rare_dic[tmp_window_set[i]] = [tmp_rare_snp_set[i]]
            elif i < len(tmp_window_set)-1:
                if tmp_window_set[i] in check_duplicates_set:
                    store_set.append(tmp_rare_snp_set[i])
                else:
                    check_duplicates_set.append(tmp_window_set[i])
                    win_rare_dic[tmp_window_set[i-1]] = store_set
                    store_set = [tmp_rare_snp_set[i]]
            else:
                if tmp_window_set[i] in check_duplicates_set:
                    store_set.append(tmp_rare_snp_set[i])
                    win_rare_dic[tmp_window_set[i]] = store_set
                    store_set = []
                else:
                    if len(store_set) != 0:
                        win_rare_dic[tmp_window_set[i-1]] = store_set
                        store_set = []
                    check_duplicates_set.append(tmp_window_set[i])
                    win_rare_dic[tmp_window_set[i]] = [tmp_rare_snp_set[i]]
    return win_rare_dic

def haplotype_listing(geno_1,geno_2,win_freq):
    # This function is to list all possible haplotypes based on genotypes
    # geno_1: the genotypes of person 1 in a window
    # geno_2: the genotypes of person 2 in a window
    # win_freq: a dictionary, it is the frequencies of haplotypes for a window in the training data
    count_1 = 0 # the count of heterozygous genotypes in the window for person 1
    count_2 = 0 # the count of heterozygous genotypes in the window for person 2
    non_heter_set_1 = [] # the index of locus of non heterozygous for person 1
    non_heter_set_2 = []
    heter_set_1 = [] # the index of locus of heterozygous for person 1
    heter_set_2 = []
    for k in range(len(geno_1)):
        if geno_1[k] == 1:
            count_1 = count_1+1
            heter_set_1.append(k)
        else:
            non_heter_set_1.append(k)
        if geno_2[k] == 1:
            count_2 = count_2+1
            heter_set_2.append(k)
        else:
            non_heter_set_2.append(k)
    keys = sorted(win_freq,key=win_freq.get,reverse=True)
    keys = [ast.literal_eval(x) for x in keys]

    # generate hap_1 and hap_2
    hap_1 = [] # the list of all potential haplotypes infered from geno_1
    hap_2 = []
    tmp_memory_list_1 = [] # This list is used to make sure no same haptylotype pairs occurs the second time in hap_1
    tmp_memory_list_2 = []
    if len(keys) == 0:
        sys.exit('len(keys) == 0')
    for i in range(len(keys)):
        if len(hap_1) >= 50 and len(hap_2) >= 50:
            break
        else:
            hap_train = keys[i] # a haplotype in the training set
            one_hap_1 = [] # one of the haplotype infered from geno_1
            alter_hap_1 = [] # the other haplotype infered from geno_1
            alter_hap_1_ = [] # this is a backup to deal with the missing values in geno_1 (if any)
            one_hap_2 = []
            alter_hap_2 = []
            alter_hap_2_ = []
            for j in range(len(hap_train)):
                if geno_1[j] == 0 or geno_1[j] == 2:
                    if geno_1[j]/2 != hap_train[j]: # This judges whether we could get the allele (hap_train[j]) from geno_1[j]. For example, if geno_1[j] == 2 and hap_train[j] == 0, then we failed. But we still include this into hap_1 to avoid len(hap_1) == 0. Corresponding adjustments for this is in function of computing common scores, where we make the such haplotype frequency very low.
                        if i == len(keys)-1 and len(hap_1) == 0:
                            one_hap_1.append(int(geno_1[j]/2))
                            alter_hap_1.append(int(geno_1[j]/2))
                            alter_hap_1_.append(int(geno_1[j]/2))
                        else:
                            break
                    else:
                        one_hap_1.append(hap_train[j])
                        alter_hap_1.append(hap_train[j])
                        alter_hap_1_.append(hap_train[j])
                elif geno_1[j] == 1:
                    one_hap_1.append(hap_train[j])
                    alter_hap_1.append(int(1-hap_train[j]))
                    alter_hap_1_.append(int(1-hap_train[j]))
                else: # missing value
                    one_hap_1.append(hap_train[j])
                    alter_hap_1.append(hap_train[j])
                    alter_hap_1_.append(int(1-hap_train[j])) # note that we may create haplotypes that do not exist in the training set, but it's OK as their frequencies will be set to a very small value

                if geno_2[j] != 1:
                    if geno_2[j]/2 != hap_train[j]:
                        if i == len(keys)-1 and len(hap_2) == 0:
                            one_hap_2.append(int(geno_2[j]/2))
                            alter_hap_2.append(int(geno_2[j]/2))
                            alter_hap_2_.append(int(geno_2[j]/2))
                        else:
                            break
                    else:
                        one_hap_2.append(hap_train[j])
                        alter_hap_2.append(hap_train[j])
                        alter_hap_2_.append(hap_train[j])
                elif geno_2[j] == 1:
                    one_hap_2.append(hap_train[j])
                    alter_hap_2.append(int(1-hap_train[j]))
                    alter_hap_2_.append(int(1-hap_train[j]))
                else:
                    one_hap_2.append(hap_train[j])
                    alter_hap_2.append(hap_train[j])
                    alter_hap_2_.append(int(1-hap_train[j]))

            if alter_hap_1 not in tmp_memory_list_1 and alter_hap_1_ not in tmp_memory_list_1 and len(alter_hap_1) == len(hap_train):
                hap_1.append(one_hap_1)
                hap_1.append(alter_hap_1)
                tmp_memory_list_1.append(one_hap_1)
                tmp_memory_list_1.append(alter_hap_1)
                if alter_hap_1 != alter_hap_1_: # missing values occur in geno_1
                    hap_1.append(one_hap_1)
                    hap_1.append(alter_hap_1_)
                    tmp_memory_list_1.append(alter_hap_1_)

            if alter_hap_2 not in tmp_memory_list_2 and alter_hap_2_ not in tmp_memory_list_2 and len(alter_hap_2) == len(hap_train):
                hap_2.append(one_hap_2)
                hap_2.append(alter_hap_2)
                tmp_memory_list_2.append(one_hap_2)
                tmp_memory_list_2.append(alter_hap_2)
                if alter_hap_2 != alter_hap_2_:
                    hap_2.append(one_hap_2)
                    hap_2.append(alter_hap_2_)
                    tmp_memory_list_2.append(alter_hap_2_)

    return hap_1,hap_2

def compute_IBD_score(geno_1,geno_2,h1,h2,h3,win_freq,epsilon,num_hap_training):
    # This function is to compute the probability of IBD for common SNPs scores
    # geno_1: the genotypes of person 1 in a window
    # geno_2: the genotypes of person 2 in a window
    # h1: the non-shared haplotype of person 1
    # h2: the shared haplotype between person 1 and person 2
    # h3: the non-shared haplotype of person 2
    # win_freq: the dictionary of frequencies of a window in the training dataset
    # epsilon: genotype error
    # num_hap_training: the number of haplotypes in the training dataset

    win_freq_copy = win_freq.copy() # Creating this copy is to aviod changing win_freq itself in this function

    p_1 = 1 # p_1 refers to the probability of genotypes given two haplotypes for person 1
    for i in range(len(geno_1)):
        if geno_1[i] == h1[i]+h2[i]:
            p_1 = p_1*(1-epsilon)
        else:
            p_1 = p_1*epsilon/2

    p_2 = 1
    for i in range(len(geno_2)):
        if geno_2[i] == h2[i]+h3[i]:
            p_2 = p_2*(1-epsilon)
        else:
            p_2 = p_2*epsilon/2    

    tmp_freq = 1/(100*num_hap_training)
    if str(h1) not in list(win_freq_copy.keys()) or win_freq_copy[str(h1)] == 0: # In case that h1 is not in the haplotypes list of training set
        win_freq_copy[str(h1)] = tmp_freq
    if str(h2) not in list(win_freq_copy.keys()) or win_freq_copy[str(h2)] == 0:
        win_freq_copy[str(h2)] = tmp_freq
    if str(h3) not in list(win_freq_copy.keys()) or win_freq_copy[str(h3)] == 0:
        win_freq_copy[str(h3)] = tmp_freq

    p_IBD = p_1*p_2*win_freq_copy[str(h1)]*win_freq_copy[str(h2)]*win_freq_copy[str(h3)]
    return p_IBD

def compute_nonIBD_score(geno_1,geno_2,h1,h2,h3,h4,win_freq,epsilon,num_hap_training):
    # This function is to compute the probability of non-IBD for common SNPs scores
    # geno_1: the genotypes of person 1 in a window
    # geno_2: the genotypes of person 2 in a window
    # h1: the haplotype of person 1
    # h2: the haplotype of person 1
    # h3: the haplotype of person 2
    # h4: the haplotype of person 2
    # win_freq: the dictionary of frequencies of a window in the training set
    # epsilon: genotype error

    win_freq_copy = win_freq.copy()

    p_1 = 1 # p_1 refers to the probability of genotypes given two haplotypes for person 1
    for i in range(len(geno_1)):
        if geno_1[i] == h1[i]+h2[i]:
            p_1 = p_1*(1-epsilon)
        else:
            p_1 = p_1*epsilon/2

    p_2 = 1
    for i in range(len(geno_2)):
        if geno_2[i] == h3[i]+h4[i]:
            p_2 = p_2*(1-epsilon)
        else:
            p_2 = p_2*epsilon/2
    
    # assign a small value to those not in win_freq_copy or those with frequency being 0
    tmp_freq = 1/(100*num_hap_training)
    if str(h1) not in list(win_freq_copy.keys()) or win_freq_copy[str(h1)] == 0:
        win_freq_copy[str(h1)] = tmp_freq
    if str(h2) not in list(win_freq_copy.keys()) or win_freq_copy[str(h2)] == 0:
        win_freq_copy[str(h2)] = tmp_freq
    if str(h3) not in list(win_freq_copy.keys()) or win_freq_copy[str(h3)] == 0:
        win_freq_copy[str(h3)] = tmp_freq
    if str(h4) not in list(win_freq_copy.keys()) or win_freq_copy[str(h4)] == 0:
        win_freq_copy[str(h4)] = tmp_freq

    p_nonIBD = p_1*p_2*win_freq_copy[str(h1)]*win_freq_copy[str(h2)]*win_freq_copy[str(h3)]*win_freq_copy[str(h4)]
    if p_nonIBD == 0:
        print('p_1: ',p_1)
        print('p_2: ',p_2)
        print('freq_h1: ',win_freq_copy[str(h1)])
        print('freq_h2: ',win_freq_copy[str(h2)])
        print('freq_h3: ',win_freq_copy[str(h3)])
        print('freq_h4: ',win_freq_copy[str(h4)])
        print('tmp_freq: ',tmp_freq)
        print('win_freq_copy: ',win_freq_copy)
        sys.exit('p_nonIBD = 0')
    return p_nonIBD

def compute_common_inner_score(geno_1,geno_2,win_freq,epsilon,num_hap_training):
    # This function is to compute the common SNPs scores for a pair of people
    # geno_1: the genotypes of person 1 in a window
    # geno_2: the genotypes of person 2 in a window
    # hap_1: the potential haplotypes given geno_1 and the shown haplotypes in the training set
    # hap_2: the potential haplotypes given geno_2 and the shown haplotypes in the training set
    # epsilon: genotype error
    
    hap_1,hap_2 = haplotype_listing(geno_1,geno_2,win_freq)
    p_IBD = 0 # the sum of probabilities of all potential haplotypes
    p_nonIBD = 0
    if int(len(hap_1)/2) != len(hap_1)/2 or int(len(hap_2)/2) != len(hap_2)/2:
        sys.exit('hap_1.shape[0] or hap_2.shape[0] is not even')
    for i in range(int(len(hap_1)/2)):
        for j in range(int(len(hap_2)/2)):
            # IBD assumption part
            ## For example, assume the first haplotype in i-th haplotype pair and the first haplotype in j-th pair to be h2
            tmp_set = []
            for k1 in range(2):
                for k2 in range(2):
                    for k3 in range(2):
                        if k1 == 0: # when k1 == 0, we take the haplotype from hap_1 as h2. The reason that we seperately take hap_1 and hap_2 as h2 is we could thus minimise the effects of genotype errors. Suppose if geno_1 has genotype error (true genotype 2 to false genotype 0 or true genotype 0 to false genotye 2), then hap_1 could not contain the correct haplotype. If we try hap_2 as well rather than hap_1 as h2, then we could be able to minimize the effects caused by genotype error (as the chance that two genotypes geno_1 and geno_2 both have error is relatively smaller)
                            h1 = list(hap_1[2*i+k2])
                            h2 = list(hap_1[2*i+(1-k2)])
                            h3 = list(hap_2[2*j+k3])
                        elif k1 == 1: # when k1 == 1, we take the haplotype from hap_2 as h2.
                            h1 = list(hap_1[2*i+k2])
                            h2 = list(hap_2[2*j+(1-k3)])
                            h3 = list(hap_2[2*j+k3])
                        p_IBD_inter = compute_IBD_score(geno_1,geno_2,h1,h2,h3,win_freq,epsilon,num_hap_training)
                        tmp_set.append(p_IBD_inter)
            p_IBD += max(tmp_set) # we pick the largest value in tmp_set as p_IBD_inter. Because we do not know which haplotype should be the actual h2, we take all possible choices and choose the one with the largest value 
            # non-IBD assumption part
            h1 = list(hap_1[2*i])
            h2 = list(hap_1[2*i+1])
            h3 = list(hap_2[2*j])
            h4 = list(hap_2[2*j+1])
            p_nonIBD_inter = compute_nonIBD_score(geno_1,geno_2,h1,h2,h3,h4,win_freq,epsilon,num_hap_training)
            if p_nonIBD_inter == 0:
                sys.exit('p_nonIBD_inter == 0')
            p_nonIBD += p_nonIBD_inter
    if p_IBD != 0 and p_nonIBD != 0:
        common_score = np.log(p_IBD/p_nonIBD)
    elif p_IBD == 0 and p_nonIBD != 0:
        common_score = -100 # -100 is used to represent -inf. Any reasonablely negative values could be picked as long as it is far from 0
    else:
        common_score = 0
    result = [p_IBD,p_nonIBD,common_score]
    return result

###############################################################################
# This part is for rare SNPs

def compute_allele_num(train_counts,test_tped,rare_snp_set):
    # Input:
    # train_counts: the dataframe of .frq.counts file of rare SNPs in training dataset
    # test_tped: the dataframe of .tped file of test dataset of rare SNPs
    # rare_snp_set: the list consists of indexes of rare SNPs in which both individuals have minor allele
    # Output:
    # minor_set_train: the list of the count of minor alleles (rare SNPs) in the training set

    minor_set_train = []
    major_set_train = []
    minor_set_test = []
    major_set_test = []
    for i in rare_snp_set:
        minor_allele = train_counts.loc[i,'A1']
        major_allele = train_counts.loc[i,'A2']
        count_minor_train = int(train_counts.loc[i,'C1'])
        count_major_train = int(train_counts.loc[i,'C2'])
        minor_set_train.append(count_minor_train)
        major_set_train.append(count_major_train)

        count_minor_test = 0
        count_major_test = 0
        for j in range(4,len(test_tped.columns)):
            if test_tped.iloc[i,j] == minor_allele:
                count_minor_test += 1
            elif test_tped.iloc[i,j] == major_allele:
                count_major_test += 1
        minor_set_test.append(count_minor_test)
        major_set_test.append(count_major_test)
    return minor_set_train,major_set_train,minor_set_test,major_set_test

def distribution(theta,alpha,beta,constant_value,epsilon):
    # this function is for beta distribution or normal distribution (approximation to beta distribution)
    if constant_value == np.inf: # the overflow occurs
        normal_mean = alpha/(alpha+beta)
        normal_std = (alpha*beta/((alpha+beta)**2*(alpha+beta+1)))**0.5
        res = 1/(normal_std*(2*np.pi)**0.5)*np.exp(-0.5*((theta-normal_mean)/normal_std)**2)
    else:
        res = theta**(alpha-1)*(1-theta)**(beta-1)*constant_value
    return res

def p_3(h_1,h_2,h_3,alpha,beta,constant_value,epsilon):
    # this is for the case of IBD
    res = integrate.quad(lambda theta: theta**h_1*(1-theta)**(1-h_1)*theta**h_2*(1-theta)**(1-h_2)*theta**h_3*(1-theta)**(1-h_3)*distribution(theta,alpha,beta,constant_value,epsilon),0,1,epsabs=1e-15)
    return res[0]

def p_4(h_1,h_2,h_3,h_4,alpha,beta,constant_value,epsilon):
    # this is for the case of non-IBD
    # Here h_1, h_2, h_3 and h_4 are different from those in p_3 function
    res = integrate.quad(lambda theta: theta**h_1*(1-theta)**(1-h_1)*theta**h_2*(1-theta)**(1-h_2)*theta**h_3*(1-theta)**(1-h_3)*theta**h_4*(1-theta)**(1-h_4)*distribution(theta,alpha,beta,constant_value,epsilon),0,1,epsabs=1e-15) # Note that if we encounter integration warnings (e.g. report divergence), we could change the parameter 'epsabs' to solve the problem 
    return res[0]

def p_g_h_IBD(h_1,h_2,h_3,g_1,g_2,epsilon):
    # This function is to compute p(g_1|h1,h2)*p(g_2|h2,h3)
    # h_1: the genotype of person 1
    # h_2: the genotype of person 1 and person 2
    # h_3: the genotype of person 2
    # g_1: the genotype of person 1
    # g_2: the genotype of person 2
    # epsilon: the genotyping error
    p = 1
    if g_1 == h_1+h_2:
        p = p*(1-epsilon)
    else:
        p = p*epsilon/2

    if g_2 == h_2+h_3:
        p = p*(1-epsilon)
    else:
        p = p*epsilon/2
    return p

def p_g_h_nonIBD(h_1,h_2,h_3,h_4,g_1,g_2,epsilon):
    # This function is to compute p(g_1|h1,h2)*p(g_2|h3,h4)
    # h_1: the genotype of person 1
    # h_2: the genotype of person 1
    # h_3: the genotype of person 2
    # h_4: the genotype of person 2
    # g_1: the genotype of person 1
    # g_2: the genotype of person 2
    # epsilon: the genotyping error
    p = 1
    if g_1 == h_1+h_2:
        p = p*(1-epsilon)
    else:
        p = p*epsilon/2

    if g_2 == h_3+h_4:
        p = p*(1-epsilon)
    else:
        p = p*epsilon/2
    return p

def single_marker(rare_geno_1,rare_geno_2,minor_count_train,major_count_train,minor_count_test,major_count_test,epsilon):
    # This sub-function is to compute the score of a pair of individuals at a single marker
    # ped_1 is the first individual's genotype at the single marker
    # ped_2 is the second individual's genotype at the single marker
    alpha = major_count_train+major_count_test
    beta = minor_count_train+minor_count_test
    try:
        constant_value = np.exp(np.sum(np.log(np.arange(1,alpha+beta)))-np.sum(np.log(np.arange(1,alpha)))-np.sum(np.log(np.arange(1,beta)))) # a constant term in beta function
    except RuntimeWarning:
        constant_value = np.inf
    else:
        constant_value = np.exp(np.sum(np.log(np.arange(1,alpha+beta)))-np.sum(np.log(np.arange(1,alpha)))-np.sum(np.log(np.arange(1,beta))))
    g_1 = rare_geno_1
    g_2 = rare_geno_2

    p_IBD = 0
    p_nonIBD = 0
    for h_1 in [0,1]:
        for h_2 in [0,1]:
            for h_3 in [0,1]:
                p_IBD += p_g_h_IBD(h_1,h_2,h_3,g_1,g_2,epsilon)*p_3(h_1,h_2,h_3,alpha,beta,constant_value,epsilon)
                for h_4 in [0,1]:
                    if (h_3 == 1 and h_4 == 0) or (h_1 == 1 and h_2 == 0):
                        judge = 1
                    else:
                        judge = 0
                    if judge == 0: # remove the double counting when genotype is 1
                        p_nonIBD += p_g_h_nonIBD(h_1,h_2,h_3,h_4,g_1,g_2,epsilon)*p_4(h_1,h_2,h_3,h_4,alpha,beta,constant_value,epsilon)

    return p_IBD,p_nonIBD

def compute_rare_score(minor_set_train,major_set_train,minor_set_test,major_set_test,rare_snp_set,rare_geno_1_set,rare_geno_2_set,epsilon):
    # rare_snp_set: the list consists of indexes of rare SNPs in which both individuals have minor allele
    # rare_tped is the dataframe of .tped file of rare SNPs of test dataset
    # frq_count is the dataframe of .frq.counts of rare SNPs of training dataset
    p_IBD = 1
    p_nonIBD = 1
    for i in range(len(rare_snp_set)):
        rare_geno_1 = rare_geno_1_set[i]
        rare_geno_2 = rare_geno_2_set[i]
        p_IBD_inter,p_nonIBD_inter = single_marker(rare_geno_1,rare_geno_2,minor_set_train[i],major_set_train[i],minor_set_test[i],major_set_test[i],epsilon)
        p_IBD = p_IBD*p_IBD_inter
        p_nonIBD = p_nonIBD*p_nonIBD_inter
    if p_IBD != 0 and p_nonIBD != 0:
        rare_score = np.log(p_IBD/p_nonIBD)
    else:
        rare_score = 0
    result = [p_IBD,p_nonIBD,rare_score]
    return result

#########################################
# Outer log likelihood

def internal_simulation(training_hap,N_simIBD,N_simnonIBD,common_map,block_length):
    # This function is to simulate individuals sharing IBD along the entire chromosome and individuals without any IBD segments
    # Input:
    # training_hap: the dataframe of the haplotype file of training dataset
    # N_simIBD: the number of pairs of individuals to be simulated as IBD
    # N_simnonIBD: the number of pairs of individuals to be simulated as non-IBD
    # common_map: the dataframe of .map file for common variants
    # block_length: the minimum length of a block (a hyperparameter)
    

    # Part 1: simulate IBD pairs
    if N_simIBD <= len(training_hap.columns):
        index_hap_IBD = np.random.choice(len(training_hap.columns),N_simIBD,replace=False) # the index of individuals in training_hap to be simulated
        index_hap_1 = np.random.choice(len(training_hap.columns),N_simIBD,replace=False)
        index_hap_2 = np.random.choice(len(training_hap.columns),N_simIBD,replace=False)
        index_hap_IBD = list(index_hap_IBD)
        index_hap_1 = list(index_hap_1)
        index_hap_2 = list(index_hap_2)
        geno_indi_IBD = pd.DataFrame(training_hap.iloc[:,index_hap_IBD].values+training_hap.iloc[:,index_hap_1].values,columns=list(range(len(index_hap_IBD)))) # the genotypes of individuals picked
        geno_indi_IBD_tmp = pd.DataFrame(training_hap.iloc[:,index_hap_IBD].values+training_hap.iloc[:,index_hap_2].values,columns=list(range(len(index_hap_IBD))))
        geno_indi_IBD = pd.concat([geno_indi_IBD,geno_indi_IBD_tmp],axis=1)
        geno_indi_IBD.columns = list(range(len(geno_indi_IBD.columns)))
    else:
        index_hap_IBD = np.random.choice(len(training_hap.columns),len(training_hap.columns),replace=False) # the index of individuals in training_hap to be simulated
        index_hap_1 = np.random.choice(len(training_hap.columns),len(training_hap.columns),replace=False)
        index_hap_2 = np.random.choice(len(training_hap.columns),len(training_hap.columns),replace=False)
        index_hap_IBD = list(index_hap_IBD)
        index_hap_1 = list(index_hap_1)
        index_hap_2 = list(index_hap_2)
        geno_indi_IBD = pd.DataFrame(training_hap.iloc[:,index_hap_IBD].values+training_hap.iloc[:,index_hap_1].values,columns=list(range(len(index_hap_IBD)))) # the genotypes of individuals picked
        geno_indi_IBD_tmp = pd.DataFrame(training_hap.iloc[:,index_hap_IBD].values+training_hap.iloc[:,index_hap_2].values,columns=list(range(len(index_hap_IBD))))
        geno_indi_IBD = pd.concat([geno_indi_IBD,geno_indi_IBD_tmp],axis=1)
        geno_indi_IBD.columns = list(range(len(geno_indi_IBD.columns)))
        
        num_indi = N_simIBD-len(training_hap.columns)
        chr_start_pos = common_map.iloc[0,2]
        chr_end_pos = common_map.iloc[-1,2]
        chr_length = chr_end_pos-chr_start_pos
        region_length = chr_length/10
        num_regions = 10
        geno_indi_IBD_tmp = pd.DataFrame([])
        for i in range(num_regions):
            if i == 0:
                region_start_pos = chr_start_pos-0.1
                region_end_pos = chr_start_pos+(i+1)*region_length
            elif i > 0 and i < num_regions-1:
                region_start_pos = chr_start_pos+i*region_length
                region_end_pos = chr_start_pos+(i+1)*region_length
            else:
                region_start_pos = chr_start_pos+i*region_length
                region_end_pos = chr_end_pos+0.1 # here, 0.1 is not a must, any number greater than zero (not too small) is OK
            tmp_index1 = common_map.index[common_map.iloc[:,2]>=region_start_pos] # the index of markers that their positions (cM) are 'larger' than the start position of the region
            tmp_index2 = common_map.index[common_map.iloc[:,2]<region_end_pos]
            index_inter = sorted(list(set(tmp_index1).intersection(set(tmp_index2)))) # the indices of markers that are in the region
            index_hap_IBD = np.random.choice(len(training_hap.columns),num_indi,replace=True) 
            index_hap_1 = np.random.choice(len(training_hap.columns),num_indi,replace=True)
            index_hap_2 = np.random.choice(len(training_hap.columns),num_indi,replace=True)
            geno_region = pd.DataFrame(training_hap.iloc[index_inter,index_hap_IBD].values+training_hap.iloc[index_inter,index_hap_1].values,columns=list(range(len(index_hap_IBD))))
            geno_region_tmp = pd.DataFrame(training_hap.iloc[index_inter,index_hap_IBD].values+training_hap.iloc[index_inter,index_hap_2].values,columns=list(range(len(index_hap_IBD))))
            geno_region = pd.concat([geno_region,geno_region_tmp],axis=1)
            geno_region.columns = list(range(len(geno_region.columns)))
            geno_indi_IBD_tmp = pd.concat([geno_indi_IBD_tmp,geno_region],ignore_index=True)
        geno_indi_IBD = pd.concat([geno_indi_IBD,geno_indi_IBD_tmp],axis=1)
        geno_indi_IBD.columns = list(range(len(geno_indi_IBD.columns)))
        # change the order of columns such that the i-th column and (i+N_simIBD)-th column are IBD
        index1 = list(range(len(training_hap.columns)))+list(range(len(training_hap.columns)*2,len(training_hap.columns)*2+num_indi))
        index2 = list(range(len(training_hap.columns),len(training_hap.columns)*2))+list(range(len(training_hap.columns)*2+num_indi,len(training_hap.columns)*2+num_indi*2))
        index_union = index1+index2
        geno_indi_IBD = geno_indi_IBD.loc[:,index_union]
        geno_indi_IBD.columns = list(range(len(geno_indi_IBD.columns)))
        

    # Part 2: simulate non-IBD pairs
    # We will generate new individuals by randomly select different regions of the chromosome from several individuals and combine them into one
    # create regions that will be drawn from multiple individuals
    chr_start_pos = common_map.iloc[0,2]
    chr_end_pos = common_map.iloc[-1,2]
    chr_length = chr_end_pos-chr_start_pos
    if block_length > 0.2:
        region_length = 0.2 # the length of each region
    else:
        region_length = 0.5*block_length
    num_regions = int(chr_length/region_length) # the number of regions
    geno_indi_nonIBD = pd.DataFrame([])
    for i in range(num_regions):
        if i == 0:
            region_start_pos = chr_start_pos-0.1
            region_end_pos = chr_start_pos+(i+1)*region_length
        elif i > 0 and i < num_regions-1:
            region_start_pos = chr_start_pos+i*region_length
            region_end_pos = chr_start_pos+(i+1)*region_length
        else:
            region_start_pos = chr_start_pos+i*region_length
            region_end_pos = chr_end_pos+0.1 # here, 0.1 is not a must, any number greater than zero (not too small) is OK
        tmp_index1 = common_map.index[common_map.iloc[:,2]>=region_start_pos] # the index of markers that their positions (cM) are 'larger' than the start position of the region
        tmp_index2 = common_map.index[common_map.iloc[:,2]<region_end_pos]
        index_inter = sorted(list(set(tmp_index1).intersection(set(tmp_index2)))) # the indices of markers that are in the region
        index_hap_1 = np.random.choice(len(training_hap.columns),N_simnonIBD,replace=True)
        index_hap_2 = np.random.choice(len(training_hap.columns),N_simnonIBD,replace=True)
        index_hap_3 = np.random.choice(len(training_hap.columns),N_simnonIBD,replace=True)
        index_hap_4 = np.random.choice(len(training_hap.columns),N_simnonIBD,replace=True)
        # make sure index_hap_3 and index_hap_4 do not have the same value with index_hap_1 or index_hap_2 in the same position, i.e. we get them from different haplotypes to avoid IBD
        arr1 = (index_hap_3-index_hap_1 == 0)
        arr2 = (index_hap_3-index_hap_2 == 0)
        if sum(arr1) > 0:
            index_hap_3[arr1] = index_hap_3[arr1]+1
            arr3 = (index_hap_3 >= len(training_hap.columns))
            if sum(arr3) > 0:
                index_hap_3[arr3] = index_hap_3[arr3] - np.random.randint(len(training_hap.columns)-3) # here len(training_hap.columns)-3 is to avoid we generate the same value with index_hap_1
        if sum(arr2) > 0:
            index_hap_3[arr2] = index_hap_3[arr2]+1
            arr3 = (index_hap_3 >= len(training_hap.columns))
            if sum(arr3) > 0:
                index_hap_3[arr3] = index_hap_3[arr3] - np.random.randint(len(training_hap.columns)-3)
        
        arr1 = (index_hap_4-index_hap_1 == 0)
        arr2 = (index_hap_4-index_hap_2 == 0)
        if sum(arr1) > 0:
            index_hap_4[arr1] = index_hap_4[arr1]+1
            arr3 = (index_hap_4 >= len(training_hap.columns))
            if sum(arr3) > 0:
                index_hap_4[arr3] = index_hap_4[arr3] - np.random.randint(len(training_hap.columns)-3)
        if sum(arr2) > 0:
            index_hap_4[arr2] = index_hap_4[arr2]+1
            arr3 = (index_hap_4 >= len(training_hap.columns))
            if sum(arr3) > 0:
                index_hap_4[arr3] = index_hap_4[arr3] - np.random.randint(len(training_hap.columns)-3)
        geno_region = pd.DataFrame(training_hap.iloc[index_inter,index_hap_1].values+training_hap.iloc[index_inter,index_hap_2].values,columns=list(range(len(index_hap_1))))
        geno_region_tmp = pd.DataFrame(training_hap.iloc[index_inter,index_hap_3].values+training_hap.iloc[index_inter,index_hap_4].values,columns=list(range(len(index_hap_3))))
        geno_region = pd.concat([geno_region,geno_region_tmp],axis=1)
        geno_region.columns = list(range(len(geno_region.columns)))
        geno_indi_nonIBD = pd.concat([geno_indi_nonIBD,geno_region],ignore_index=True)
    geno_indi_IBD.columns = list(range(len(geno_indi_IBD.columns)))
    geno_indi_nonIBD.columns = list(range(len(geno_indi_nonIBD.columns)))
    return geno_indi_IBD,geno_indi_nonIBD

def empirical_distribution(win_indi_IBD,win_indi_nonIBD,bin_length,win_freq,epsilon_common,num_hap_training,pseudo_count):
    # This function is used to generate empirical distribution
    # Input:
    # win_indi_IBD: the dataframe of simulated IBD individuals in a window
    # win_indi_nonIBD: the dataframe of simulated non-IBD individuals in a window
    # bin_length: the length of bins
    # pseudo_count: Used in Laplace smoothing. A technique used to smooth categorical data.
    IBD_score_set = []
    nonIBD_score_set = []

    # compute scores from IBD pairs
    ibd_col = len(win_indi_IBD.columns)
    nonibd_col = len(win_indi_nonIBD.columns)
    for i in range(max(int(ibd_col/2),int(nonibd_col/2))):
        if i < int(ibd_col/2):
            common_geno_1 = list(win_indi_IBD.iloc[:,i].values)
            noise_array = np.random.choice(3,len(win_indi_IBD),p=[1-epsilon_common,epsilon_common/2,epsilon_common/2])
            common_geno_2 = list(np.abs(win_indi_IBD.iloc[:,i+int(ibd_col/2)].values-noise_array)) # common_geno_1 and common_geno_2 are the basically the same (we introduce some noises to make the model more robust) as we simulate two identical sequensenses along the entire chromosome
            common_inner_result = compute_common_inner_score(common_geno_1,common_geno_2,win_freq,epsilon_common,num_hap_training)
            IBD_score_set.append(common_inner_result[2])

    # compute scores from non-IBD pairs
        if i < int(nonibd_col/2):
            common_geno_1 = list(win_indi_nonIBD.iloc[:,i].values)
            common_geno_2 = list(win_indi_nonIBD.iloc[:,i+int(nonibd_col/2)].values)
            common_inner_result = compute_common_inner_score(common_geno_1,common_geno_2,win_freq,epsilon_common,num_hap_training)
            nonIBD_score_set.append(common_inner_result[2])

    IBD_score_set = sorted(IBD_score_set)
    nonIBD_score_set = sorted(nonIBD_score_set)
    IBD_score_array = np.array(IBD_score_set)
    nonIBD_score_array = np.array(nonIBD_score_set)
    left_bound = -18 # the left boundary of bins excluding -inf
    right_bound = 18
    num_bin = int((right_bound-left_bound)/bin_length)+2 # we have '+2' as we have two more bins, which are (-inf,left_bound) and (right_bound,inf)
    boundary_score_left_IBD_set = []
    boundary_score_right_IBD_set = []
    boundary_score_left_nonIBD_set = []
    boundary_score_right_nonIBD_set = []
    bin_frequency_IBD_set = []
    bin_frequency_nonIBD_set = []
    for i in range(num_bin):
        # IBD part
        if i == 0:
            boundary_score_left = -np.inf
            boundary_score_right = left_bound
            index1 = np.where(IBD_score_array>boundary_score_left)[0] # here we do not have >= but >
            index2 = np.where(IBD_score_array<boundary_score_right)[0] # here we do not have <= but <
            index_inter = list(set(index1).intersection(set(index2)))
            bin_frequency = (len(index_inter)+pseudo_count)/(len(IBD_score_set)+pseudo_count*num_bin) # smoothed version of empirical probability. Note that the empirical probability in this bin is 0 in the training data set, and we add a pseudo count here to avoid the empirical probability to be 0.
        elif i > 0 and i < num_bin-2:
            boundary_score_left = left_bound+(i-1)*bin_length # the left boundary score of the bin
            boundary_score_right = left_bound+i*bin_length
            index1 = np.where(IBD_score_array>=boundary_score_left)[0]
            index2 = np.where(IBD_score_array<boundary_score_right)[0]
            index_inter = list(set(index1).intersection(set(index2)))
            bin_frequency = (len(index_inter)+pseudo_count)/(len(IBD_score_set)+pseudo_count*num_bin) # smoothed version of empirical probability
        elif i == num_bin-2:
            boundary_score_left = left_bound+(i-1)*bin_length # the left boundary score of the bin
            boundary_score_right = right_bound
            index1 = np.where(IBD_score_array>=boundary_score_left)[0]
            index2 = np.where(IBD_score_array<=boundary_score_right)[0] # note that here we have '<=' instead of '<' as we need to count the value at the right boundary here
            index_inter = list(set(index1).intersection(set(index2)))
            bin_frequency = (len(index_inter)+pseudo_count)/(len(IBD_score_set)+pseudo_count*num_bin)
        else:
            boundary_score_left = right_bound
            boundary_score_right = np.inf
            index1 = np.where(IBD_score_array>boundary_score_left)[0] # here we do not have >= but >
            index2 = np.where(IBD_score_array<boundary_score_right)[0] # here we do not have <= but <
            index_inter = list(set(index1).intersection(set(index2)))
            bin_frequency = (len(index_inter)+pseudo_count)/(len(IBD_score_set)+pseudo_count*num_bin)
        boundary_score_left_IBD_set.append(boundary_score_left)
        boundary_score_right_IBD_set.append(boundary_score_right)
        bin_frequency_IBD_set.append(bin_frequency)

        # non-IBD part
        if i == 0:
            boundary_score_left = -np.inf
            boundary_score_right = left_bound
            index1 = np.where(nonIBD_score_array>boundary_score_left)[0]
            index2 = np.where(nonIBD_score_array<boundary_score_right)[0]
            index_inter = list(set(index1).intersection(set(index2)))
            bin_frequency = (len(index_inter)+pseudo_count)/(len(nonIBD_score_set)+pseudo_count*num_bin)
        elif i > 0 and i < num_bin-2:
            boundary_score_left = left_bound+(i-1)*bin_length # the left boundary score of the bin
            boundary_score_right = left_bound+i*bin_length
            index1 = np.where(nonIBD_score_array>=boundary_score_left)[0]
            index2 = np.where(nonIBD_score_array<boundary_score_right)[0]
            index_inter = list(set(index1).intersection(set(index2)))
            bin_frequency = (len(index_inter)+pseudo_count)/(len(nonIBD_score_set)+pseudo_count*num_bin)
        elif i == num_bin-2:
            boundary_score_left = left_bound+(i-1)*bin_length # the left boundary score of the bin
            boundary_score_right = right_bound
            index1 = np.where(nonIBD_score_array>=boundary_score_left)[0]
            index2 = np.where(nonIBD_score_array<=boundary_score_right)[0] # note that here we have '<=' instead of '<' as we need to count the value at the right boundary here
            index_inter = list(set(index1).intersection(set(index2)))
            bin_frequency = (len(index_inter)+pseudo_count)/(len(nonIBD_score_set)+pseudo_count*num_bin)
        else:
            boundary_score_left = right_bound
            boundary_score_right = np.inf
            index1 = np.where(nonIBD_score_array>boundary_score_left)[0]
            index2 = np.where(nonIBD_score_array<boundary_score_right)[0]
            index_inter = list(set(index1).intersection(set(index2)))
            bin_frequency = (len(index_inter)+pseudo_count)/(len(nonIBD_score_set)+pseudo_count*num_bin)
        boundary_score_left_nonIBD_set.append(boundary_score_left)
        boundary_score_right_nonIBD_set.append(boundary_score_right)
        bin_frequency_nonIBD_set.append(bin_frequency)
    df_empi_IBD = pd.DataFrame(boundary_score_left_IBD_set)
    df_empi_IBD[1] = boundary_score_right_IBD_set
    df_empi_IBD[2] = bin_frequency_IBD_set

    df_empi_nonIBD = pd.DataFrame(boundary_score_left_nonIBD_set)
    df_empi_nonIBD[1] = boundary_score_right_nonIBD_set
    df_empi_nonIBD[2] = bin_frequency_nonIBD_set

    return [df_empi_IBD,df_empi_nonIBD]

def compute_common_outer_score(df_empi_IBD,df_empi_nonIBD,common_geno_1_set,common_geno_2_set,win_freq,epsilon_common,num_hap_training):
    # This function is used to compute the outer log-likelihood ratio
    # Input:
    # df_empi_IBD: the dataframe of data on empirical distribution of IBD pairs for a window
    # df_empi_nonIBD: the dataframe of data on empirical distribution of non-IBD pairs for a window
    # common_geno_1_set: the list of genotypes of individual 1 in the testing data set for a window
    # common_geno_2_set: the list of genotypes of individual 2 in the testing data set for a window
    # 

    df_empi_IBD.index = list(range(len(df_empi_IBD.index)))
    df_empi_nonIBD.index = list(range(len(df_empi_nonIBD.index)))
    common_result = compute_common_inner_score(common_geno_1_set,common_geno_2_set,win_freq,epsilon_common,num_hap_training)
    common_inner_score = common_result[2]

    # IBD part
    index1 = df_empi_IBD.index[df_empi_IBD.loc[:,0]<=common_inner_score]
    index2 = df_empi_IBD.index[df_empi_IBD.loc[:,1]>common_inner_score]
    if len(list(set(index1).intersection(set(index2)))) == 0:
        print('df_empi_IBD:\n',df_empi_IBD)
        print('common_inner_score: ',common_inner_score)
        sys.exit('the intersection is empty for common_inner_score in df_empi_IBD')
    index_inter = list(set(index1).intersection(set(index2)))
    if len(index_inter) == 1:
        index_inter = index_inter[0]
        if common_inner_score == df_empi_IBD.iloc[-2,1]: # the case that common_inner_score equals to the right boundary of the second bin counting from the right-hand side
            index_inter = index_inter-1 # we allocate the score to the left bin of the original bin as the score actually appeared in the training data set and thus should be allocated in the left bin
    else:
        sys.exit('len(index_inter) != 1 in IBD case of function compute_common_outer_score')
    common_outer_score_IBD = df_empi_IBD.loc[index_inter,2]

    # non-IBD part
    index1 = df_empi_nonIBD.index[df_empi_nonIBD.loc[:,0]<=common_inner_score]
    index2 = df_empi_nonIBD.index[df_empi_nonIBD.loc[:,1]>common_inner_score]
    if len(list(set(index1).intersection(set(index2)))) == 0:
        print('df_empi_nonIBD:\n',df_empi_nonIBD)
        print('common_inner_score: ',common_inner_score)
        sys.exit('the intersection is empty for common_inner_score in df_empi_nonIBD')
    index_inter = list(set(index1).intersection(set(index2)))
    if len(index_inter) == 1:
        index_inter = index_inter[0]
        if common_inner_score == df_empi_nonIBD.iloc[-2,1]:
            index_inter = index_inter-1
    else:
        sys.exit('len(index_inter) != 1 in non-IBD case of function compute_common_outer_score')
    common_outer_score_nonIBD = df_empi_nonIBD.loc[index_inter,2]

    common_outer_score = np.log(common_outer_score_IBD/common_outer_score_nonIBD)
    
    return common_outer_score

####################################################

def compute_score_v1(common_geno_1_set,common_geno_2_set,win_freq,minor_set_train,major_set_train,minor_set_test,major_set_test,epsilon_common,epsilon_rare,num_hap_training,rare_SNP_set,rare_geno_1_set,rare_geno_2_set,judge_rare):
    # This function computes inner log-likelihood ratio of common variants and scores of low-frequency variants, and combine them into a single score
    # Input:
    # win_index: the index of the window
    epsilon_common = 0.01*epsilon_common # during inference, we multiply a scaling factor (0.01 here) to epsilon_common to reduce FPR
    common_result = compute_common_inner_score(common_geno_1_set,common_geno_2_set,win_freq,epsilon_common,num_hap_training)
    if judge_rare == 1:
        rare_result = compute_rare_score(minor_set_train,major_set_train,minor_set_test,major_set_test,rare_SNP_set,rare_geno_1_set,rare_geno_2_set,epsilon_rare)
    if judge_rare == 1:
        p_IBD = common_result[0]*rare_result[0]
        p_nonIBD = common_result[1]*rare_result[1]
        common_score = common_result[2]
        rare_score = rare_result[2]
    else:
        p_IBD = common_result[0]
        p_nonIBD = common_result[1]
        common_score = common_result[2]
        rare_score = 0
    if p_IBD != 0 and p_nonIBD != 0:
        score = np.log(p_IBD/p_nonIBD)
    else:
        score = 0
    result = [common_score,rare_score,score]
    return result

def compute_score_v2(df_empi_IBD,df_empi_nonIBD,common_geno_1_set,common_geno_2_set,win_freq,minor_set_train,major_set_train,minor_set_test,major_set_test,epsilon_common,epsilon_rare,num_hap_training,rare_SNP_set,rare_geno_1_set,rare_geno_2_set,judge_rare):
    # This function computes outer log-likelihood ratio of common variants and scores of low-frequency variants, and combine them into a single score
    epsilon_common = 0.01*epsilon_common # during inference, we multiply a scaling factor (0.01 here) to epsilon_common to reduce FPR
    common_result = compute_common_outer_score(df_empi_IBD,df_empi_nonIBD,common_geno_1_set,common_geno_2_set,win_freq,epsilon_common,num_hap_training)
    if judge_rare == 1:
        rare_result = compute_rare_score(minor_set_train,major_set_train,minor_set_test,major_set_test,rare_SNP_set,rare_geno_1_set,rare_geno_2_set,epsilon_rare)
        common_score = common_result
        rare_score = rare_result[2]
    else:
        common_score = common_result
        rare_score = 0
    score = common_score+rare_score
    result = [common_score,rare_score,score]
    return result

def combine_pos_set(start_pos_set_1,end_pos_set_1,start_pos_set_2,end_pos_set_2):
    # This function is used in function generate_IBD_region to combine IBD regions from common variants and low frequency variants into a final IBD regions
    df_pos = pd.DataFrame(start_pos_set_1)
    df_pos[1] = end_pos_set_1

    for i in range(len(start_pos_set_2)):
        start_pos = start_pos_set_2[i]
        end_pos = end_pos_set_2[i]
        if len(df_pos.index) == 0:
            df_pos = pd.DataFrame([start_pos,end_pos])
            df_pos = df_pos.T
            start_pos_set = [start_pos]
            end_pos_set = [end_pos]
            continue
        index1_set = list(df_pos.index[df_pos.iloc[:,0]<=start_pos])
        index2_set = list(df_pos.index[df_pos.iloc[:,1]>=end_pos])
        if len(index1_set) == 0 and len(index2_set) == 0:
            start_pos_set = [start_pos]
            end_pos_set = [end_pos]
        elif len(index1_set) > 0 and len(index2_set) == 0:
            index1 = list(index1_set)[-1]
            if start_pos > df_pos.iloc[index1,1]:
                start_pos_set = list(df_pos.loc[:index1,0].values)
                end_pos_set = list(df_pos.loc[:index1,1].values)
                start_pos_set.append(start_pos)
                end_pos_set.append(end_pos)
            else:
                start_pos_set = list(df_pos.loc[:index1,0].values)
                end_pos_set = list(df_pos.loc[:index1,1].values)
                end_pos_set[-1] = end_pos
        elif len(index1_set) == 0 and len(index2_set) > 0:
            index2 = list(index2_set)[0]
            if end_pos < df_pos.iloc[index2,0]:
                start_pos_set = list(df_pos.loc[index2:,0].values)
                end_pos_set = list(df_pos.loc[index2:,1].values)
                start_pos_set = [start_pos]+start_pos_set
                end_pos_set = [end_pos]+end_pos_set
            else:
                start_pos_set = list(df_pos.loc[index2:,0].values)
                end_pos_set = list(df_pos.loc[index2:,1].values)
                start_pos_set[0] = start_pos
        else:
            index1 = list(index1_set)[-1]
            index2 = list(index2_set)[0]
            if start_pos <= df_pos.iloc[index1,1]:
                if end_pos >= df_pos.iloc[index2,0]:
                    if index1 == index2:
                        start_pos_set = list(df_pos.iloc[:,0].values)
                        end_pos_set = list(df_pos.iloc[:,1].values)
                    else:
                        start_pos_set = list(df_pos.loc[:index1,0].values)
                        end_pos_set = list(df_pos.loc[:index1,1].values)
                        end_pos_set[-1] = df_pos.loc[index2,1]
                        if index2 < len(df_pos.index):
                            start_pos_set = start_pos_set + list(df_pos.loc[(index2+1):,0].values)
                            end_pos_set = end_pos_set + list(df_pos.loc[(index2+1):,1].values)
                else:
                    start_pos_set = list(df_pos.loc[:index1,0].values)
                    end_pos_set = list(df_pos.loc[:index1,1].values)
                    end_pos_set[-1] = end_pos
                    start_pos_set = start_pos_set + list(df_pos.loc[index2:,0].values)
                    end_pos_set = end_pos_set + list(df_pos.loc[index2:,1].values)
            else:
                if end_pos >= df_pos.iloc[index2,0]:
                    start_pos_set = list(df_pos.loc[:index1,0].values)
                    end_pos_set = list(df_pos.loc[:index1,1].values)
                    start_pos_set.append(start_pos)
                    end_pos_set.append(df_pos.loc[index2,1])
                    if index2 < len(df_pos.index):
                        start_pos_set = start_pos_set + list(df_pos.loc[(index2+1):,0].values)
                        end_pos_set = end_pos_set + list(df_pos.loc[(index2+1):,1].values)
                else:
                    start_pos_set = list(df_pos.loc[:index1,0].values)
                    end_pos_set = list(df_pos.loc[:index1,1].values)
                    start_pos_set.append(start_pos)
                    end_pos_set.append(end_pos)
                    start_pos_set = start_pos_set + list(df_pos.loc[index2:,0].values)
                    end_pos_set = end_pos_set + list(df_pos.loc[index2:,1].values)

        df_pos = pd.DataFrame(start_pos_set)
        df_pos[1] = end_pos_set
        if len(start_pos_set) == 0:
            break
    
    return start_pos_set,end_pos_set 

def remove_pos_set(start_pos_set_1,end_pos_set_1,start_pos_set_2_non,end_pos_set_2_non):
    # This function is used in function generate_IBD_region to remove regions in IBD segments that have overlaps with non-IBD regions calculated from low-frequency variants
    df_pos = pd.DataFrame(start_pos_set_1)
    df_pos[1] = end_pos_set_1

    for i in range(len(start_pos_set_2_non)):
        start_pos = start_pos_set_2_non[i]
        end_pos = end_pos_set_2_non[i]
        index1_set = list(df_pos.index[df_pos.iloc[:,0]<=start_pos])
        index2_set = list(df_pos.index[df_pos.iloc[:,1]>=end_pos])
        if len(index1_set) == 0 and len(index2_set) == 0:
            start_pos_set = []
            end_pos_set = []
            break
        elif len(index1_set) > 0 and len(index2_set) == 0:
            index1 = list(index1_set)[-1]
            if start_pos > df_pos.iloc[index1,1]:
                start_pos_set = list(df_pos.loc[:index1,0].values)
                end_pos_set = list(df_pos.loc[:index1,1].values)
            else:
                if start_pos != df_pos.iloc[index1,0]:
                    start_pos_set = list(df_pos.loc[:index1,0].values)
                    end_pos_set = list(df_pos.loc[:index1,1].values)
                    end_pos_set[-1] = start_pos-1
                else:
                    if index1 == 0:
                        start_pos_set = []
                        end_pos_set = []
                    else:
                        start_pos_set = list(df_pos.loc[:(index1-1),0].values)
                        end_pos_set = list(df_pos.loc[:(index1-1),1].values)
        elif len(index1_set) == 0 and len(index2_set) > 0:
            index2 = list(index2_set)[0]
            if end_pos < df_pos.iloc[index2,0]:
                start_pos_set = list(df_pos.loc[index2:,0].values)
                end_pos_set = list(df_pos.loc[index2:,1].values)
            else:
                if end_pos != df_pos.iloc[index2,1]:
                    start_pos_set = list(df_pos.loc[index2:,0].values)
                    end_pos_set = list(df_pos.loc[index2:,1].values)
                    start_pos_set[0] = end_pos+1
                else:
                    if index2 == len(df_pos.index):
                        start_pos_set = []
                        end_pos_set = []
                    else:
                        start_pos_set = list(df_pos.loc[(index2+1):,0].values)
                        end_pos_set = list(df_pos.loc[(index2+1):,1].values)
        else:
            index1 = list(index1_set)[-1]
            index2 = list(index2_set)[0]
            if start_pos <= df_pos.iloc[index1,1]:
                if end_pos >= df_pos.iloc[index2,0]:
                    if index1 > 0 and index2 < len(df_pos.index):
                        if start_pos == df_pos.iloc[index1,0]:
                            start_pos_set = list(df_pos.iloc[:index1,0].values)
                            end_pos_set = list(df_pos.iloc[:index1,1].values)
                        else:
                            start_pos_set = list(df_pos.iloc[:index1,0].values)
                            end_pos_set = list(df_pos.iloc[:index1,1].values)
                            start_pos_set = start_pos_set + [df_pos.iloc[index1,0]]
                            end_pos_set = end_pos_set + [start_pos-1]
                        if end_pos == df_pos.iloc[index2,1]:
                            start_pos_set = start_pos_set + list(df_pos.iloc[(index2+1):,0].values)
                            end_pos_set = end_pos_set + list(df_pos.iloc[(index2+1):,1].values)
                        else:
                            start_pos_set = start_pos_set + [end_pos+1]
                            end_pos_set = end_pos_set + [df_pos.iloc[index2,1]]
                            start_pos_set = start_pos_set + list(df_pos.iloc[(index2+1):,0].values)
                            end_pos_set = end_pos_set + list(df_pos.iloc[(index2+1):,1].values)
                    elif index1 == 0 and index2 < len(df_pos.index):
                        if start_pos == df_pos.iloc[index1,0]:
                            start_pos_set = []
                            end_pos_set = []
                        else:
                            start_pos_set = [df_pos.iloc[index1,0]]
                            end_pos_set = [start_pos-1]
                        if end_pos == df_pos.iloc[index2,1]:
                            start_pos_set = start_pos_set + list(df_pos.iloc[(index2+1):,0].values)
                            end_pos_set = end_pos_set + list(df_pos.iloc[(index2+1):,1].values)
                        else:
                            start_pos_set = start_pos_set + [end_pos+1]
                            end_pos_set = end_pos_set + [df_pos.iloc[index2,1]]
                            start_pos_set = start_pos_set + list(df_pos.iloc[(index2+1):,0].values)
                            end_pos_set = end_pos_set + list(df_pos.iloc[(index2+1):,1].values)
                    elif index1 > 0 and index2 == len(df_pos.index):
                        if start_pos == df_pos.iloc[index1,0]:
                            start_pos_set = list(df_pos.iloc[:index1,0].values)
                            end_pos_set = list(df_pos.iloc[:index1,1].values)
                        else:
                            start_pos_set = list(df_pos.iloc[:index1,0].values)
                            end_pos_set = list(df_pos.iloc[:index1,1].values)
                            start_pos_set = start_pos_set + [df_pos.iloc[index1,0]]
                            end_pos_set = end_pos_set + [start_pos-1]
                        if end_pos < df_pos.iloc[index2,1]:
                            start_pos_set = start_pos_set + [end_pos+1]
                            end_pos_set = end_pos_set + [df_pos.iloc[index2,1]]
                    else:
                        if start_pos == df_pos.iloc[index1,0]:
                            start_pos_set = []
                            end_pos_set = []
                        else:
                            start_pos_set = [df_pos.iloc[index1,0]]
                            end_pos_set = [start_pos-1]
                        if end_pos < df_pos.iloc[index2,1]:
                            start_pos_set = start_pos_set + [end_pos+1]
                            end_pos_set = end_pos_set + [df_pos.iloc[index2,1]]
                        
                else:
                    if index1 == 0:
                        if start_pos == df_pos.iloc[index1,0]:
                            start_pos_set = []
                            end_pos_set = []
                        else:
                            start_pos_set = [df_pos.iloc[index1,0]]
                            end_pos_set = [start_pos-1]
                    else:
                        start_pos_set = list(df_pos.iloc[:index1,0].values)
                        end_pos_set = list(df_pos.iloc[:index1,1].values)
                        if start_pos > df_pos.iloc[index1,0]:
                            start_pos_set = start_pos_set + [df_pos.iloc[index1,0]]
                            end_pos_set = end_pos_set + [start_pos-1]
                    start_pos_set = start_pos_set + list(df_pos.iloc[index2:,0].values)
                    end_pos_set = end_pos_set + list(df_pos.iloc[index2:,1].values)    

            else:
                start_pos_set = list(df_pos.loc[:index1,0].values)
                end_pos_set = list(df_pos.loc[:index1,1].values)
                if end_pos >= df_pos.iloc[index2,0]:
                    if index2 == len(df_pos.index):
                        if end_pos < df_pos.iloc[index2,1]:
                            start_pos_set = start_pos_set + [end_pos+1]
                            end_pos_set = end_pos_set + [df_pos.iloc[index2,1]]
                    else:
                        if end_pos < df_pos.iloc[index2,1]:
                            start_pos_set = start_pos_set + [end_pos+1]
                            end_pos_set = end_pos_set + [df_pos.iloc[index2,1]]
                        start_pos_set = start_pos_set + list(df_pos.loc[(index2+1):,0].values)
                        end_pos_set = end_pos_set + list(df_pos.loc[(index2+1):,1].values)
                else:
                    start_pos_set = list(df_pos.loc[:index1,0].values)
                    end_pos_set = list(df_pos.loc[:index1,1].values)
                    start_pos_set = start_pos_set + list(df_pos.loc[index2:,0].values)
                    end_pos_set = end_pos_set + list(df_pos.iloc[index2:,1].values)
        df_pos = pd.DataFrame(start_pos_set)
        df_pos[1] = end_pos_set
        if len(start_pos_set) == 0:
            break
    
    return start_pos_set,end_pos_set

def generate_block_scores(indi_1,indi_2,whole_blocks_set,start_index,score_set):
    # This function is to generate block scores for one pair of individuals
    # Input:
    # index_pair: the index of the pair in the pair_set that this function is working on
    
    indi_1_set = [indi_1]*len(whole_blocks_set)
    indi_2_set = [indi_2]*len(whole_blocks_set)
    block_index_set = []
    block_windows_set = []
    block_common_score_set = []
    for block_index in range(len(whole_blocks_set)):
        block_array = np.array(whole_blocks_set[block_index])
        block_common_score = score_set[block_array+start_index].sum()
        block_index_set.append(block_index)
        block_windows_set.append(whole_blocks_set[block_index])
        block_common_score_set.append(block_common_score)

    list_combined = [indi_1_set,indi_2_set,block_index_set,block_windows_set,block_common_score_set]
    return list_combined
    

def generate_IBD_region(df_block_score,df_window_score,windows_set,common_map,rare_map,threshold,negative_ratio_thres,negative_count_thres,include_clean):
    # Input:
    # df_block_score: the dataframe of block score file of a pair of individuals
    # df_window_score: the dataframe of window score file of a pair of individuals
    # windows_set: the list of windows in which associated markers are
    # common_map: the .map file of common SNPs
    # rare_map: the .map file of rare SNPs
    # threshold: the threshold for LLR score
    # negative_ratio_thres: the threshold for the negative_ratio
    # negative_count_thres: the threshold for the negative_count
    # include_clean: whether include the cleaning procedure or not (include_clean == 1 if it is included)
    # Output:
    #

    df_block_score.index = list(range(len(df_block_score.index)))
    df_window_score.index = list(range(len(df_window_score.index)))
    index = df_block_score.index[df_block_score.loc[:,'common_score']>=threshold]
    if len(index) == 0:
        check_IBD = 0 # check_IBD == 0 means that there is no IBD region
        start_pos_set = []
        end_pos_set = []
    else:
        check_IBD = 1
        windows_index_IBD_set = list(df_block_score.loc[index,'window_index'].values)
        windows_index_IBD_set = np.array(sorted(list(set(chain(*windows_index_IBD_set))))) # the list of windows that are in IBD regions
        # get the boundary of each consecutive windows region
        tmp_windows_index = windows_index_IBD_set[1:]
        tmp_diff = tmp_windows_index-windows_index_IBD_set[:-1]
        windows_index_IBD_set = list(windows_index_IBD_set)
        tmp_diff = pd.DataFrame(tmp_diff)
        boundary_index = list(tmp_diff.index[tmp_diff.loc[:,0]>1])
        start_pos_set = []
        end_pos_set = []
        if len(boundary_index) == 0: # In this case, we have only one IBD region
            start_index = windows_index_IBD_set[0]
            end_index = windows_index_IBD_set[-1]
            if start_index == 0:
                previous_common_end_pos = common_map.iloc[windows_set[start_index][0],3]-1
            else:
                previous_common_end_pos = common_map.iloc[windows_set[start_index-1][-1],3]
            if end_index == len(windows_set)-1:
                next_common_start_pos = common_map.iloc[windows_set[end_index][-1],3]+1
            else:
                next_common_start_pos = common_map.iloc[windows_set[end_index+1][0],3]
            common_start_pos = common_map.iloc[windows_set[start_index][0],3]
            common_end_pos = common_map.iloc[windows_set[end_index][-1],3]
            index1 = rare_map.index[rare_map.loc[:,3]<common_start_pos]
            index2 = rare_map.index[rare_map.loc[:,3]>previous_common_end_pos]
            index_inter1 = sorted(list(set(index1).intersection(set(index2))))
            index3 = rare_map.index[rare_map.loc[:,3]>common_end_pos]
            index4 = rare_map.index[rare_map.loc[:,3]<next_common_start_pos]
            index_inter2 = sorted(list(set(index3).intersection(set(index4))))
            if len(index_inter1) == 0 and len(index_inter2) == 0:
                start_pos = common_start_pos
                end_pos = common_end_pos
            elif len(index_inter1) > 0 and len(index_inter2) == 0:
                tmp_pos_set = []
                for j in range(len(index_inter1)):
                    if np.abs(rare_map.loc[index_inter1[j],3]-previous_common_end_pos) >= np.abs(rare_map.loc[index_inter1[j],3]-common_start_pos):
                        tmp_pos_set.append(rare_map.loc[index_inter1[j],3])
                if len(tmp_pos_set) == 0:
                    start_pos = common_start_pos
                else:
                    start_pos = min(tmp_pos_set) # get the minimum in the tmp_pos_set
                end_pos = common_end_pos
            elif len(index_inter1) == 0 and len(index_inter2) > 0:
                tmp_pos_set = []
                for j in range(len(index_inter2)):
                    if np.abs(rare_map.loc[index_inter2[j],3]-next_common_start_pos) > np.abs(rare_map.loc[index_inter2[j],3]-common_end_pos): # here it's not '>=' but just '>', because for consistency, we put rare_SNPs that are in the case of '=' to windows that they have the rare SNPs before them but not behind them
                        tmp_pos_set.append(rare_map.loc[index_inter2[j],3])
                if len(tmp_pos_set) == 0:
                    end_pos = common_end_pos
                else:
                    end_pos = max(tmp_pos_set) # get the maximum in the tmp_pos_set, note that here is not minimum
                start_pos = common_start_pos
            else:
                tmp_pos_set_1 = []
                tmp_pos_set_2 = []
                for j in range(len(index_inter1)):
                    if np.abs(rare_map.loc[index_inter1[j],3]-previous_common_end_pos) >= np.abs(rare_map.loc[index_inter1[j],3]-common_start_pos):
                        tmp_pos_set_1.append(rare_map.loc[index_inter1[j],3])
                if len(tmp_pos_set_1) == 0:
                    start_pos = common_start_pos
                else:
                    start_pos = min(tmp_pos_set_1) # get the minimum in the tmp_pos_set
                for k in range(len(index_inter2)):
                    if np.abs(rare_map.loc[index_inter2[k],3]-next_common_start_pos) > np.abs(rare_map.loc[index_inter2[k],3]-common_end_pos):
                        tmp_pos_set_2.append(rare_map.loc[index_inter2[k],3])
                if len(tmp_pos_set_2) == 0:
                    end_pos = common_end_pos
                else:
                    end_pos = max(tmp_pos_set_2) # get the maximum in the tmp_pos_set, note that here is not minimum
            start_pos_set.append(start_pos)
            end_pos_set.append(end_pos)

        else:
            for i in range(len(boundary_index)+1):
                # get the index of first and last windows in a consecutive window cluster
                if i == 0:
                    start_index = windows_index_IBD_set[0]
                    end_index = windows_index_IBD_set[boundary_index[i]]
                elif i > 0 and i < len(boundary_index):
                    start_index = windows_index_IBD_set[boundary_index[i-1]+1]
                    end_index = windows_index_IBD_set[boundary_index[i]]
                else:
                    start_index = windows_index_IBD_set[boundary_index[i-1]+1]
                    end_index = windows_index_IBD_set[-1]
                # get the start and end position
                if i == 0:
                    if start_index == 0:
                        previous_common_end_pos = common_map.iloc[windows_set[start_index][0],3]-1
                    else:
                        previous_common_end_pos = common_map.iloc[windows_set[start_index-1][-1],3] # the position of the last marker in the previous neighbor window
                    next_common_start_pos = common_map.iloc[windows_set[end_index+1][0],3] # the position of the first marker in the next window
                elif i > 0 and i < len(boundary_index):
                    previous_common_end_pos = common_map.iloc[windows_set[start_index-1][-1],3]
                    next_common_start_pos = common_map.iloc[windows_set[end_index+1][0],3]
                else:
                    previous_common_end_pos = common_map.iloc[windows_set[start_index-1][-1],3]
                    if end_index == len(windows_set)-1:
                        next_common_start_pos = common_map.iloc[windows_set[end_index][-1],3]+1
                    else:
                        next_common_start_pos = common_map.iloc[windows_set[end_index+1][0],3]
                common_start_pos = common_map.iloc[windows_set[start_index][0],3]
                common_end_pos = common_map.iloc[windows_set[end_index][-1],3]
                index1 = rare_map.index[rare_map.loc[:,3]<common_start_pos]
                index2 = rare_map.index[rare_map.loc[:,3]>previous_common_end_pos]
                index_inter1 = sorted(list(set(index1).intersection(set(index2))))
                index3 = rare_map.index[rare_map.loc[:,3]>common_end_pos]
                index4 = rare_map.index[rare_map.loc[:,3]<next_common_start_pos]
                index_inter2 = sorted(list(set(index3).intersection(set(index4))))
                if len(index_inter1) == 0 and len(index_inter2) == 0:
                    start_pos = common_start_pos
                    end_pos = common_end_pos
                elif len(index_inter1) > 0 and len(index_inter2) == 0:
                    tmp_pos_set = []
                    for j in range(len(index_inter1)):
                        if np.abs(rare_map.loc[index_inter1[j],3]-previous_common_end_pos) >= np.abs(rare_map.loc[index_inter1[j],3]-common_start_pos):
                            tmp_pos_set.append(rare_map.loc[index_inter1[j],3])
                    if len(tmp_pos_set) == 0:
                        start_pos = common_start_pos
                    else:
                        start_pos = min(tmp_pos_set) # get the minimum in the tmp_pos_set
                    end_pos = common_end_pos
                elif len(index_inter1) == 0 and len(index_inter2) > 0:
                    tmp_pos_set = []
                    for j in range(len(index_inter2)):
                        if np.abs(rare_map.loc[index_inter2[j],3]-next_common_start_pos) > np.abs(rare_map.loc[index_inter2[j],3]-common_end_pos): # here it's not '>=' but just '>', because for consistency, we put rare_SNPs that are in the case of '=' to windows that they have the rare SNPs before them but not behind them
                            tmp_pos_set.append(rare_map.loc[index_inter2[j],3])
                    if len(tmp_pos_set) == 0:
                        end_pos = common_end_pos
                    else:
                        end_pos = max(tmp_pos_set) # get the maximum in the tmp_pos_set, note that here is not minimum
                    start_pos = common_start_pos
                else:
                    tmp_pos_set_1 = []
                    tmp_pos_set_2 = []
                    for j in range(len(index_inter1)):
                        if np.abs(rare_map.loc[index_inter1[j],3]-previous_common_end_pos) >= np.abs(rare_map.loc[index_inter1[j],3]-common_start_pos):
                            tmp_pos_set_1.append(rare_map.loc[index_inter1[j],3])
                    if len(tmp_pos_set_1) == 0:
                        start_pos = common_start_pos
                    else:
                        start_pos = min(tmp_pos_set_1) # get the minimum in the tmp_pos_set
                    for k in range(len(index_inter2)):
                        if np.abs(rare_map.loc[index_inter2[k],3]-next_common_start_pos) > np.abs(rare_map.loc[index_inter2[k],3]-common_end_pos):
                            tmp_pos_set_2.append(rare_map.loc[index_inter2[k],3])
                    if len(tmp_pos_set_2) == 0:
                        end_pos = common_end_pos
                    else:
                        end_pos = max(tmp_pos_set_2) # get the maximum in the tmp_pos_set, note that here is not minimum
                start_pos_set.append(start_pos)
                end_pos_set.append(end_pos)

    # get the IBD regions based on low-frequency variants
    index_set = df_window_score.index[df_window_score.loc[:,'lowf_score']>1] # get the indices of low frequency variants that two individuals share minor alleles. We use >1 instead of >0 due to that genotypes of 1 and 2 of two individuals could have lowf_score slighly larger than 0 and we do not want to include this case.
    lowf_start_pos_set = []
    lowf_end_pos_set = []
    index_store_set = [] # This list is used to store the indices of windows that have been included in the IBD regions for low frequency variants

    if len(index_set) != 0:
        for i in range(len(index_set)):
            index = list(index_set)[i]
            if index not in index_store_set:
                llr = df_window_score.loc[index,'LLR']
                index_left = index
                index_right = index
                index_store_set.append(index)
                if df_window_score.loc[index,'common_score'] < 0:
                    negative_count = 1 # count of negative window scores (common score)
                else:
                    negative_count = 0
                all_count = 1
                while llr >= threshold:
                    region_start_cm_pos = common_map.iloc[windows_set[index_left][0],2]
                    region_end_cm_pos = common_map.iloc[windows_set[index_right][-1],2]
                    region_length = region_end_cm_pos-region_start_cm_pos
                    negative_ratio = negative_count/all_count
                    if region_length <= 2 and (negative_ratio <= negative_ratio_thres or negative_count <= negative_count_thres): # low frequency variants are used for finding short IBD regions. We do not extend the region if it is longer than 2 cM
                        if index_left > 0 and index_right < len(df_window_score.index)-1:
                            index_left = index_left-1
                            index_right = index_right+1
                            left_llr = df_window_score.loc[index_left,'LLR']
                            right_llr = df_window_score.loc[index_right,'LLR']
                            new_llr = llr+left_llr+right_llr
                            if new_llr >= threshold and left_llr >= -3 and right_llr >= -3:
                                llr = new_llr
                                index_store_set.append(index_left)
                                index_store_set.append(index_right)
                                if df_window_score.loc[index_left,'common_score'] < 0:
                                    negative_count += 1
                                if df_window_score.loc[index_right,'common_score'] < 0:
                                    negative_count += 1
                                all_count += 2
                            else:
                                new_llr = llr+left_llr
                                if new_llr >= threshold and left_llr >= -3:
                                    llr = new_llr
                                    index_store_set.append(index_left)
                                    if df_window_score.loc[index_left,'common_score'] < 0:
                                        negative_count += 1
                                    all_count += 1
                                    index_right = index_right-1 # here it is to avoid the expansion of the right direction when adding the llr score of the right window does not exceed the threshold
                                else:
                                    new_llr = llr+right_llr
                                    if new_llr >= threshold and right_llr >= -3:
                                        llr = new_llr
                                        index_store_set.append(index_right)
                                        if df_window_score.loc[index_right,'common_score'] < 0:
                                            negative_count += 1
                                        all_count += 1
                                        index_left = index_left+1
                                    else: # the case that expansion makes LLR score smaller than the threshold or both left_llr and right_llr are smaller than -3
                                        index_left = index_left+1
                                        index_right = index_right-1
                                        break
                        elif index_left <= 0 and index_right < len(df_window_score.index)-1:
                            index_right = index_right+1
                            right_llr = df_window_score.loc[index_right,'LLR']
                            new_llr = llr+right_llr
                            if new_llr >= threshold and right_llr >= -3:
                                llr = new_llr
                                if df_window_score.loc[index_right,'common_score'] < 0:
                                    negative_count += 1
                                all_count += 1
                                index_store_set.append(index_right)
                            else:
                                index_right = index_right-1
                                break
                        elif index_left > 0 and index_right >= len(df_window_score.index)-1:
                            index_left = index_left-1
                            left_llr = df_window_score.loc[index_left,'LLR']
                            new_llr = llr+left_llr
                            if new_llr >= threshold and left_llr >= -3:
                                llr = new_llr
                                index_store_set.append(index_left)
                                if df_window_score.loc[index_left,'common_score'] < 0:
                                    negative_count += 1
                                all_count += 1
                            else:
                                index_left = index_left+1
                                break
                        else:
                            break
                    else:
                        if index_left != index and index_right != index:
                            index_left = index_left+1
                            index_right = index_right-1
                        else:
                            llr = threshold-1 # we do not report this region as an IBD region
                        break
                if llr >= threshold:
                    if index_left == 0:
                        previous_common_end_pos = common_map.iloc[windows_set[index_left][0],3]-1
                    else:
                        previous_common_end_pos = common_map.iloc[windows_set[index_left-1][0],3]

                    if index_right == len(windows_set)-1:
                        next_common_start_pos = common_map.iloc[windows_set[index_right][-1],3]+1
                    else:
                        next_common_start_pos = common_map.iloc[windows_set[index_right+1][0],3]
                    common_start_pos = common_map.iloc[windows_set[index_left][0],3]
                    common_end_pos = common_map.iloc[windows_set[index_right][-1],3]
                    index1 = rare_map.index[rare_map.loc[:,3]<common_start_pos]
                    index2 = rare_map.index[rare_map.loc[:,3]>previous_common_end_pos]
                    index_inter1 = sorted(list(set(index1).intersection(set(index2))))
                    index3 = rare_map.index[rare_map.loc[:,3]>common_end_pos]
                    index4 = rare_map.index[rare_map.loc[:,3]<next_common_start_pos]
                    index_inter2 = sorted(list(set(index3).intersection(set(index4))))
                    if len(index_inter1) == 0 and len(index_inter2) == 0:
                        start_pos = common_start_pos
                        end_pos = common_end_pos
                    elif len(index_inter1) > 0 and len(index_inter2) == 0:
                        tmp_pos_set = []
                        for j in range(len(index_inter1)):
                            if np.abs(rare_map.loc[index_inter1[j],3]-previous_common_end_pos) >= np.abs(rare_map.loc[index_inter1[j],3]-common_start_pos):
                                tmp_pos_set.append(rare_map.loc[index_inter1[j],3])
                        if len(tmp_pos_set) == 0:
                            start_pos = common_start_pos
                        else:
                            start_pos = min(tmp_pos_set) # get the minimum in the tmp_pos_set
                        end_pos = common_end_pos
                    elif len(index_inter1) == 0 and len(index_inter2) > 0:
                        tmp_pos_set = []
                        for j in range(len(index_inter2)):
                            if np.abs(rare_map.loc[index_inter2[j],3]-next_common_start_pos) > np.abs(rare_map.loc[index_inter2[j],3]-common_end_pos): # here it's not '>=' but just '>', because for consistency, we put rare_SNPs that are in the case of '=' to windows that they have the rare SNPs before them but not behind them
                                tmp_pos_set.append(rare_map.loc[index_inter2[j],3])
                        if len(tmp_pos_set) == 0:
                            end_pos = common_end_pos
                        else:
                            end_pos = max(tmp_pos_set) # get the maximum in the tmp_pos_set, note that here is not minimum
                        start_pos = common_start_pos
                    else:
                        tmp_pos_set_1 = []
                        tmp_pos_set_2 = []
                        for j in range(len(index_inter1)):
                            if np.abs(rare_map.loc[index_inter1[j],3]-previous_common_end_pos) >= np.abs(rare_map.loc[index_inter1[j],3]-common_start_pos):
                                tmp_pos_set_1.append(rare_map.loc[index_inter1[j],3])
                        if len(tmp_pos_set_1) == 0:
                            start_pos = common_start_pos
                        else:
                            start_pos = min(tmp_pos_set_1) # get the minimum in the tmp_pos_set
                        for k in range(len(index_inter2)):
                            if np.abs(rare_map.loc[index_inter2[k],3]-next_common_start_pos) > np.abs(rare_map.loc[index_inter2[k],3]-common_end_pos):
                                tmp_pos_set_2.append(rare_map.loc[index_inter2[k],3])
                        if len(tmp_pos_set_2) == 0:
                            end_pos = common_end_pos
                        else:
                            end_pos = max(tmp_pos_set_2) # get the maximum in the tmp_pos_set, note that here is not minimum
                    if len(lowf_start_pos_set) > 0:
                        if start_pos > max(lowf_end_pos_set):
                            lowf_start_pos_set.append(start_pos)
                            lowf_end_pos_set.append(end_pos)
                        elif start_pos == max(lowf_end_pos_set):
                            lowf_end_pos_set[-1] = end_pos
                        else:
                            if start_pos <= min(lowf_start_pos_set):
                                lowf_start_pos_set = [start_pos]
                                if end_pos > max(lowf_end_pos_set):
                                    lowf_end_pos_set = [end_pos]
                                else:
                                    lowf_end_pos_set = [max(lowf_end_pos_set)]
                            else:
                                for m1 in range(len(lowf_end_pos_set)-1):
                                    if start_pos <= lowf_end_pos_set[m1]:
                                        lowf_start_pos_set = lowf_start_pos_set[:(m1+1)]
                                        lowf_end_pos_set = lowf_end_pos_set[:(m1+1)]
                                        lowf_end_pos_set[-1] = end_pos
                                        break
                                    else:
                                        if start_pos < lowf_start_pos_set[m1+1]:
                                            lowf_start_pos_set = lowf_start_pos_set[:(m1+1)]
                                            lowf_end_pos_set = lowf_end_pos_set[:(m1+1)]
                                            lowf_start_pos_set.append(start_pos)
                                            lowf_end_pos_set.append(end_pos)
                                            break
                    else:
                        lowf_start_pos_set.append(start_pos)
                        lowf_end_pos_set.append(end_pos)

    # get the non-IBD regions based on low frquency variants
    lowf_start_pos_set_non = []
    lowf_end_pos_set_non = []
    index_store_set_non = []
    if include_clean == 1:
        index_set = df_window_score.index[df_window_score.loc[:,'lowf_score']<0]
        if len(index_set) != 0:
            for i in range(len(index_set)):
                index = list(index_set)[i]
                if index not in index_store_set_non:
                    llr = df_window_score.loc[index,'LLR']
                    index_left = index
                    index_right = index
                    index_store_set_non.append(index)
                    while llr < threshold:
                        region_start_cm_pos = common_map.iloc[windows_set[index_left][0],2]
                        region_end_cm_pos = common_map.iloc[windows_set[index_right][-1],2]
                        region_length = region_end_cm_pos-region_start_cm_pos
                        if region_length <= 1 and df_window_score.loc[index_left,'common_score']>-1 and df_window_score.loc[index_right,'common_score']>-1: # we only care about low frequency variants that stay in the regions of common scores larger than -1 (IBS regions)
                            if index_left > 0 and index_right < len(df_window_score.index)-1:
                                index_left = index_left-1
                                index_right = index_right+1
                                left_llr = df_window_score.loc[index_left,'LLR']
                                right_llr = df_window_score.loc[index_right,'LLR']
                                new_llr = llr+left_llr+right_llr
                                if new_llr < threshold:
                                    llr = new_llr
                                    index_store_set_non.append(index_left)
                                    index_store_set_non.append(index_right)
                                else:
                                    new_llr = llr+left_llr
                                    if new_llr < threshold:
                                        llr = new_llr
                                        index_store_set_non.append(index_left)
                                        index_right = index_right-1 
                                    else:
                                        new_llr = llr+right_llr
                                        if new_llr < threshold:
                                            llr = new_llr
                                            index_store_set_non.append(index_right)
                                            index_left = index_left+1
                                        else:
                                            index_left = index_left+1
                                            index_right = index_right-1
                                            break
                            elif index_left <= 0 and index_right < len(df_window_score.index)-1:
                                index_right = index_right+1
                                right_llr = df_window_score.loc[index_right,'LLR']
                                new_llr = llr+right_llr
                                if new_llr < threshold:
                                    llr = new_llr
                                    index_store_set_non.append(index_right)
                                else:
                                    index_right = index_right-1
                                    break
                            elif index_left > 0 and index_right >= len(df_window_score.index)-1:
                                index_left = index_left-1
                                left_llr = df_window_score.loc[index_left,'LLR']
                                new_llr = llr+left_llr
                                if new_llr < threshold:
                                    llr = new_llr
                                    index_store_set_non.append(index_left)
                                else:
                                    index_left = index_left+1
                                    break
                            else:
                                break
                        else:
                            if index_right != index and index_left != index:
                                index_left = index_left+1
                                index_right = index_right-1
                            break
                    if llr < threshold:
                        if index_left == 0:
                            previous_common_end_pos = common_map.iloc[windows_set[index_left][0],3]-1
                        else:
                            previous_common_end_pos = common_map.iloc[windows_set[index_left-1][0],3]
    
                        if index_right == len(windows_set)-1:
                            next_common_start_pos = common_map.iloc[windows_set[index_right][-1],3]+1
                        else:
                            next_common_start_pos = common_map.iloc[windows_set[index_right+1][0],3]
                        common_start_pos = common_map.iloc[windows_set[index_left][0],3]
                        common_end_pos = common_map.iloc[windows_set[index_right][-1],3]
                        index1 = rare_map.index[rare_map.loc[:,3]<common_start_pos]
                        index2 = rare_map.index[rare_map.loc[:,3]>previous_common_end_pos]
                        index_inter1 = sorted(list(set(index1).intersection(set(index2))))
                        index3 = rare_map.index[rare_map.loc[:,3]>common_end_pos]
                        index4 = rare_map.index[rare_map.loc[:,3]<next_common_start_pos]
                        index_inter2 = sorted(list(set(index3).intersection(set(index4))))
                        if len(index_inter1) == 0 and len(index_inter2) == 0:
                            start_pos = common_start_pos
                            end_pos = common_end_pos
                        elif len(index_inter1) > 0 and len(index_inter2) == 0:
                            tmp_pos_set = []
                            for j in range(len(index_inter1)):
                                if np.abs(rare_map.loc[index_inter1[j],3]-previous_common_end_pos) >= np.abs(rare_map.loc[index_inter1[j],3]-common_start_pos):
                                    tmp_pos_set.append(rare_map.loc[index_inter1[j],3])
                            if len(tmp_pos_set) == 0:
                                start_pos = common_start_pos
                            else:
                                start_pos = min(tmp_pos_set) # get the minimum in the tmp_pos_set
                            end_pos = common_end_pos
                        elif len(index_inter1) == 0 and len(index_inter2) > 0:
                            tmp_pos_set = []
                            for j in range(len(index_inter2)):
                                if np.abs(rare_map.loc[index_inter2[j],3]-next_common_start_pos) > np.abs(rare_map.loc[index_inter2[j],3]-common_end_pos): # here it's not '>=' but just '>', because for consistency, we put rare_SNPs that are in the case of '=' to windows that they have the rare SNPs before them but not behind them
                                    tmp_pos_set.append(rare_map.loc[index_inter2[j],3])
                            if len(tmp_pos_set) == 0:
                                end_pos = common_end_pos
                            else:
                                end_pos = max(tmp_pos_set) # get the maximum in the tmp_pos_set, note that here is not minimum
                            start_pos = common_start_pos
                        else:
                            tmp_pos_set_1 = []
                            tmp_pos_set_2 = []
                            for j in range(len(index_inter1)):
                                if np.abs(rare_map.loc[index_inter1[j],3]-previous_common_end_pos) >= np.abs(rare_map.loc[index_inter1[j],3]-common_start_pos):
                                    tmp_pos_set_1.append(rare_map.loc[index_inter1[j],3])
                            if len(tmp_pos_set_1) == 0:
                                start_pos = common_start_pos
                            else:
                                start_pos = min(tmp_pos_set_1) # get the minimum in the tmp_pos_set
                            for k in range(len(index_inter2)):
                                if np.abs(rare_map.loc[index_inter2[k],3]-next_common_start_pos) > np.abs(rare_map.loc[index_inter2[k],3]-common_end_pos):
                                    tmp_pos_set_2.append(rare_map.loc[index_inter2[k],3])
                            if len(tmp_pos_set_2) == 0:
                                end_pos = common_end_pos
                            else:
                                end_pos = max(tmp_pos_set_2) # get the maximum in the tmp_pos_set, note that here is not minimum
                        lowf_start_pos_set_non.append(start_pos)
                        lowf_end_pos_set_non.append(end_pos)

    if len(lowf_start_pos_set) > 0:
        if len(start_pos_set) > 0:
            start_pos_set,end_pos_set = combine_pos_set(start_pos_set,end_pos_set,lowf_start_pos_set,lowf_end_pos_set)
        else:
            start_pos_set = lowf_start_pos_set
            end_pos_set = lowf_end_pos_set
            check_IBD = 1
    
    if len(lowf_start_pos_set_non) > 0 and len(start_pos_set) > 0:
        start_pos_set,end_pos_set = remove_pos_set(start_pos_set,end_pos_set,lowf_start_pos_set_non,lowf_end_pos_set_non)
        if len(start_pos_set) == 0:
            check_IBD = 0

    indi_1 = df_block_score.iloc[0,0]
    indi_2 = df_block_score.iloc[0,1]
    list1 = [start_pos_set,end_pos_set,indi_1,indi_2,check_IBD,lowf_start_pos_set_non,lowf_end_pos_set_non]
    return list1



