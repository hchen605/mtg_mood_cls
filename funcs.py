import numpy as np
import pickle
import random
import pandas as pd
from matplotlib import pyplot as plt

def generate_train_aug_csv(train_csv, aug_csv, feat_path, aug_path, experiments):
    lines = open(train_csv, 'r').readlines()
    title = lines[0]
    lines = lines[1:]
    for idx, elem in enumerate(lines):
        lines[idx] = lines[idx].split('\t')
        lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

    lines_aug = open(aug_csv, 'r').readlines()
    title_aug = lines_aug[0]
    lines_aug = lines_aug[1:]
    for idx, elem in enumerate(lines_aug):
        lines_aug[idx] = lines_aug[idx].split('\t')
        lines_aug[idx][0] = lines_aug[idx][0].split('/')[-1].split('.')[0]

    temp = aug_path.split('/')
    temp_aug = feat_path.split('/')
    name = temp[-1]
    name_aug = temp_aug[-1]
    if name == '':
        name = temp[-2]
    if name_aug == '':
        name_aug = temp_aug[-2]

    fw_csv = experiments + '/' + name_aug + '_' + name + '.csv'
    fw = open(fw_csv, 'w')
    fw.write(title)
    for i in range(len(lines)):
        fw.write(feat_path + '/' + lines[i][0] + '.logmel')
        fw.write('\t' + lines[i][1])

    for i in range(len(lines_aug)):
        fw.write(aug_path + '/' + lines_aug[i][0] + '.logmel')
        fw.write('\t' + lines_aug[i][1])

    fw.close()

    return fw_csv
        
def generate_train_aug_csv_2(train_csv, aug_csv, feat_path, aug_path, experiments):
    lines = open(train_csv, 'r').readlines()
    title = lines[0]
    lines = lines[1:]

    lines_aug = open(aug_csv, 'r').readlines()
    title_aug = lines_aug[0]
    lines_aug = lines_aug[1:]
    for idx, elem in enumerate(lines_aug):
        lines_aug[idx] = lines_aug[idx].split('\t')
        lines_aug[idx][0] = lines_aug[idx][0].split('/')[-1].split('.')[0]

    temp = aug_path.split('/')
    temp_aug = feat_path.split('/')
    name = temp[-1]
    name_aug = temp_aug[-1]
    if name == '':
        name = temp[-2]
    if name_aug == '':
        name_aug = temp_aug[-2]

    fw_csv = experiments + '/' +  name_aug + '_' + name + '.csv'
    fw = open(fw_csv, 'w')
    fw.write(title)
    for i in range(len(lines)):
        fw.write(lines[i])
        #fw.write('\t' + lines[i][1])

    for i in range(len(lines_aug)):
        fw.write(aug_path + '/' + lines_aug[i][0] + '.logmel')
        fw.write('\t' + lines_aug[i][1])

    fw.close()

    return fw_csv
        

def frequency_masking(mel_spectrogram, frequency_masking_para=13, frequency_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(frequency_mask_num):
        f = random.randrange(0, frequency_masking_para)
        f0 = random.randrange(0, fbank_size[0] - f)

        if (f0 == f0 + f):
            continue

        mel_spectrogram[f0:(f0+f),:] = 0
    return mel_spectrogram
   

   
def time_masking(mel_spectrogram, time_masking_para=40, time_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(time_mask_num):
        t = random.randrange(0, time_masking_para)
        t0 = random.randrange(0, fbank_size[1] - t)

        if (t0 == t0 + t):
            continue

        mel_spectrogram[:, t0:(t0+t)] = 0
    return mel_spectrogram


def cmvn(data):
    shape = data.shape
    eps = 2**-30
    for i in range(shape[0]):
        utt = data[i].squeeze().T
        mean = np.mean(utt, axis=0)
        utt = utt - mean
        std = np.std(utt, axis=0)
        utt = utt / (std + eps)
        utt = utt.T
        data[i] = utt.reshape((utt.shape[0], utt.shape[1], 1))
    return data


def frequency_label(num_sample, num_frequencybins, num_timebins):

    data = np.arange(num_frequencybins, dtype='float32').reshape(num_frequencybins, 1) / num_frequencybins
    data = np.broadcast_to(data, (num_frequencybins, num_timebins))
    data = np.broadcast_to(data, (num_sample, num_frequencybins, num_timebins))
    data = np.expand_dims(data, -1)
    
    return data




       
from sklearn.metrics import roc_auc_score

def maha_distance(xs,cov_inv_in,mean_in,norm_type=None):
  diffs = xs - mean_in.reshape([1,-1])

  second_powers = np.matmul(diffs,cov_inv_in)*diffs

  if norm_type in [None,"L2"]:
    return np.sum(second_powers,axis=1)
  elif norm_type in ["L1"]:
    return np.sum(np.sqrt(np.abs(second_powers)),axis=1)
  elif norm_type in ["Linfty"]:
    return np.max(second_powers,axis=1)

def get_scores(
    indist_train_embeds_in,
    indist_train_labels_in,
    indist_test_embeds_in,
    outdist_test_embeds_in,
    subtract_mean = True,
    normalize_to_unity = True,
    subtract_train_distance = True,
    indist_classes = 100,
    norm_name = "L2",
    ):
    # storing the replication results
    maha_intermediate_dict = dict()
  
    description = ""
    
    all_train_mean = np.mean(indist_train_embeds_in,axis=0,keepdims=True)

    indist_train_embeds_in_touse = indist_train_embeds_in
    indist_test_embeds_in_touse = indist_test_embeds_in
    outdist_test_embeds_in_touse = outdist_test_embeds_in

    if subtract_mean:
        indist_train_embeds_in_touse -= all_train_mean
        indist_test_embeds_in_touse -= all_train_mean
        outdist_test_embeds_in_touse -= all_train_mean
        description = description+" subtract mean,"

    if normalize_to_unity:
        indist_train_embeds_in_touse = indist_train_embeds_in_touse / np.linalg.norm(indist_train_embeds_in_touse,axis=1,keepdims=True)
        indist_test_embeds_in_touse = indist_test_embeds_in_touse / np.linalg.norm(indist_test_embeds_in_touse,axis=1,keepdims=True)
        outdist_test_embeds_in_touse = outdist_test_embeds_in_touse / np.linalg.norm(outdist_test_embeds_in_touse,axis=1,keepdims=True)
        description = description+" unit norm,"

    #full train single fit
    mean = np.mean(indist_train_embeds_in_touse,axis=0)
    cov = np.cov((indist_train_embeds_in_touse-(mean.reshape([1,-1]))).T)

    eps = 1e-8
    cov_inv = np.linalg.inv(cov)

    #getting per class means and covariances
    class_means = []
    class_cov_invs = []
    class_covs = []

    for c in range(indist_classes):

        mean_now = np.mean(indist_train_embeds_in_touse[indist_train_labels_in == c],axis=0)

        cov_now = np.cov((indist_train_embeds_in_touse[indist_train_labels_in == c]-(mean_now.reshape([1,-1]))).T)
        class_covs.append(cov_now)
        # print(c)

        eps = 1e-8
        cov_inv_now = np.linalg.inv(cov_now)

        class_cov_invs.append(cov_inv_now)
        class_means.append(mean_now)

    #the average covariance for class specific
    class_cov_invs = [np.linalg.inv(np.mean(np.stack(class_covs,axis=0),axis=0))]*len(class_covs)

    maha_intermediate_dict["class_cov_invs"] = class_cov_invs
    maha_intermediate_dict["class_means"] = class_means
    maha_intermediate_dict["cov_inv"] = cov_inv
    maha_intermediate_dict["mean"] = mean

    out_totrain = maha_distance(outdist_test_embeds_in_touse,cov_inv,mean,norm_name)
    in_totrain = maha_distance(indist_test_embeds_in_touse,cov_inv,mean,norm_name)

    out_totrainclasses = [maha_distance(outdist_test_embeds_in_touse,class_cov_invs[c],class_means[c],norm_name) for c in range(indist_classes)]
    in_totrainclasses = [maha_distance(indist_test_embeds_in_touse,class_cov_invs[c],class_means[c],norm_name) for c in range(indist_classes)]

    out_scores = np.min(np.stack(out_totrainclasses,axis=0),axis=0)
    in_scores = np.min(np.stack(in_totrainclasses,axis=0),axis=0)

    if subtract_train_distance:
        out_scores = out_scores - out_totrain
        in_scores = in_scores - in_totrain


    onehots = np.array([1]*len(out_scores) + [0]*len(in_scores))
    scores = np.concatenate([out_scores,in_scores],axis=0)

    return onehots, scores, description, maha_intermediate_dict



def get_auroc(onehots,scores,make_plot = True,add_to_title=None,swap_classes=False):

    auroc = roc_auc_score(onehots, scores)

    to_replot_dict = dict()

    if swap_classes == False:
        out_scores,in_scores = scores[onehots==0], scores[onehots==1] 
    else:
        out_scores,in_scores = scores[onehots==1], scores[onehots==0] 

    if make_plot:
        plt.figure(figsize = (5.5,3),dpi=100)

        if add_to_title is not None:
            plt.title(add_to_title+" AUROC="+str(float(auroc*100))[:6]+"%",fontsize=14)
        else:
            plt.title(" AUROC="+str(float(auroc*100))[:6]+"%",fontsize=14)


    vals,bins = np.histogram(out_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    if make_plot:
        plt.plot(bin_centers,vals,linewidth=4,color="navy",marker="",label="in test")
        plt.fill_between(bin_centers,vals,[0]*len(vals),color="navy",alpha=0.3)

    to_replot_dict["out_bin_centers"] = bin_centers
    to_replot_dict["out_vals"] = vals

    vals,bins = np.histogram(in_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    if make_plot:
        plt.plot(bin_centers,vals,linewidth=4,color="crimson",marker="",label="out test")
        plt.fill_between(bin_centers,vals,[0]*len(vals),color="crimson",alpha=0.3)

    to_replot_dict["in_bin_centers"] = bin_centers
    to_replot_dict["in_vals"] = vals

    vals,bins = np.histogram(in_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    if make_plot:
        plt.plot(bin_centers,vals,linewidth=4,color="crimson",marker="",label="out test")
        plt.fill_between(bin_centers,vals,[0]*len(vals),color="crimson",alpha=0.3)

    to_replot_dict["in_bin_centers"] = bin_centers
    to_replot_dict["in_vals"] = vals

    if make_plot:
        plt.xlabel("Score",fontsize=14)
        plt.ylabel("Count",fontsize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylim([0,None])

        plt.legend(fontsize = 14)

        plt.tight_layout()
        plt.savefig('../data/ood/auroc.pdf')
        plt.show()

    return auroc,to_replot_dict


