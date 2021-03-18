import os
import matplotlib as mpl
import logging
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import glob, os
import numpy as np
import time
import datetime as DT
import collections
import argparse

from matplotlib.dates import date2num
import pickle
import subprocess
import random
import pandas as pd
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import warnings
from sklearn import metrics
import sys
import re
import math


warnings.filterwarnings('always')


def load_data(file_name, convert2array=True):
    fdata = open(file_name, 'rb')
    txt_file_name = [line.rstrip() for line in fdata]
    values = [];
    for idx,row in enumerate(txt_file_name):
        row = row.decode('cp1252').strip('\r\n').split(' ')
        values.append(row)
    fdata.close()
    if convert2array:
        return np.array(values, dtype='f')
    else:
        return values

def date2weekday(dt):
    DayL = ['Mon', 'Tues', 'Wednes', 'Thurs', 'Fri', 'Satur', 'Sun']
    year, month, day = (int(x) for x in dt.split('-'))
    answer = DT.date(year, month, day).weekday()
    return DayL[answer]

def load_motif(file_dicts, featureDimension = 10, verbose = False, convolve = False): #generate image
    dataV = np.array(load_data(file_dicts['oheart']), dtype=int).reshape(-1)
    dataTimestamp = load_data(file_dicts['timestamp'], convert2array=False)
    mtx = {_[0]:np.zeros([24, 60]) for _ in dataTimestamp}
    for idx, _ in enumerate(dataTimestamp):
        mtx[_[0]][int(_[1][:2]), int(_[1][3:5])] = dataV[idx]
    return mtx

def npmask(batch_size, length, mask_mask_length):
    mask = np.ones((batch_size, length), np.float32)
    w = np.random.randint(mask_mask_length)
    start_w = np.random.randint(length-w)
    mask[:, start_w:start_w+w] = 0.
    return mask

def convert2idx(words, offset, is_set = False):
    dictionary = {}
    for word in words:
        dictionary[word] = len(dictionary) + offset
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


def gen_batchs(central_id, batch_size, mtx, dic_dv):
    np.random.shuffle(central_id)
    xs_batches = len(central_id) // batch_size
    for idx in range(xs_batches):
        idx_begin = idx * batch_size
        idx_end = (idx + 1) * batch_size
        batch_central_Image = []
        batch_central_date = []

        'each batch'
        for id_d in central_id[idx_begin:idx_end]:
            tmp = []; tmp_d = []; tmp_gt = {}
            'each instance'
            for(id, d) in id_d:
                tmp.append(mtx[id][d]);
                tmp_d.append(dic_dv[d])


            batch_central_Image.append(np.array(tmp))
            tmp_max_d = max(tmp_d)
            batch_central_date.append([tmp_max_d-_ for _ in tmp_d])

        yield (np.array(batch_central_Image), np.array(batch_central_date))

def eval_label(alg, eval_mask, train_phase, mtx, dict_dv, start_date, end_date, num_input):
    # apply mask to each time series
    eval_mse_list, eval_mae_list, eval_mape_list, avail_vs_list = [], [], [], []
    for id in mtx:
        'gen input'
        support_inputs, support_ds, query_inputs, query_ds, total_inputs, total_ds, ds = [], [], [], [], [], [], []
        for d_idx, d in enumerate(sorted(list(mtx[id].keys()))):
            if len(mtx[id][d])!=0:
                total_ds.append(dict_dv[d])
                total_inputs.append(mtx[id][d])
            if dict_dv[d]>=start_date and dict_dv[d]<end_date and len(total_inputs)>=num_input+1:
                ds.append(d)
                support_inputs.append(total_inputs[-num_input - 1:-1])
                support_ds.append([total_ds[-1]-_ for _ in total_ds[-num_input - 1:-1]])
                query_inputs.append([total_inputs[-1]])
                query_ds.append([0])
        if len(support_inputs)>0:
            eval_mse, eval_mae, eval_mape, avail_vs = alg.getMissDailyLabel_MissValue(support_inputs, support_ds, query_inputs, query_ds, eval_mask)
            'load output'
            eval_mse_list.append(eval_mse); eval_mae_list.append(eval_mae); eval_mape_list.append(eval_mape); avail_vs_list.append(avail_vs)

    eval_mse = sum([mse * v for mse, v in zip(eval_mse_list, avail_vs_list)
                    if not math.isnan(mse) and not math.isnan(v)]) / sum(avail_vs_list)
    eval_mae = sum([mae * v for mae, v in zip(eval_mae_list, avail_vs_list)
                    if not math.isnan(mae) and not math.isnan(v)]) / sum(avail_vs_list)
    eval_mape = sum([mape * v for mape, v in zip(eval_mape_list, avail_vs_list)
                     if not math.isnan(mape) and not math.isnan(v)]) / sum(avail_vs_list)
    eval_rmse = np.sqrt(eval_mse)

    results = '%s: rmse %.2f, mae %.2f, eval_mape %.4f' % (train_phase, eval_rmse, eval_mae, eval_mape)
    return ([eval_mse, eval_mae, eval_mape, eval_rmse], results)

def eval3_label(alg, best_performance, best_valid, eval_mask, mtx, dict_dv, valid_start, tst_start, tst_end, num_input):
    valid_rslt = eval_label(alg, eval_mask, 'valid', mtx, dict_dv, valid_start, tst_start, num_input)
    test_rslt = eval_label(alg, eval_mask, 'test', mtx, dict_dv, tst_start, tst_end, num_input)
    if valid_rslt[0][1] <= best_valid[0][1]:
        best_valid= valid_rslt
        best_performance = test_rslt
    return best_valid, best_performance


def gen_epochs(n, central_id, batch_size, mtx, dic_dv):
    for i in range(n):
        yield (gen_batchs(central_id, batch_size, mtx, dic_dv))

def alys_histogram(image):
    _ = np.int32(np.array(image).reshape(-1)/10); len_ = len(_)
    dict_ = collections.Counter(_)
    dict_= {k:round(dict_[k]/len_,2) for k in dict_}
    return dict_
'###############################'
DateFormat = lambda v: DT.datetime.fromordinal(int(v)).strftime("%Y-%m-%d")
Date2v = lambda date: int(date2num(DT.datetime.strptime(date.split(' ')[0], "%Y-%m-%d").replace(minute=0).replace(second=0)))
'###############################'

def load_heart_rate_data(process, loss_type, train_ratio, num_input, num_pos, tp,gap=4):
    '###############################'
    'timeframe to infer missing values'
    # an example input for time period 1
    train_start, valid_start, tst_start, tst_end = '2017-07-01', '2018-05-15', '2018-06-01', '2018-07-01'
    
    data_path = "./motif_suwen/pcdata/heart_per_format_Motif"
    if process:
        files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        logging.info('time_stamp {}'.format(len(files)))
        file_set = list(set([f.split('_')[0] for f in files]))
        id_set = list(set([f.split('_')[-1].replace('.txt', '') for f in files if 'repeat' in f]))
        logging.info('id_set {}'.format(id_set))#'lheart', 'res', 'repeat', 'oheart', 'timestamp
        logging.info('id_set {}'.format(len(id_set)))

        idx = 0
        id_files = {}; mtx = {}
        for idx, id in enumerate(id_set):#tranverse ID
            id_files[id] = {}
            for file in file_set:
                cfile = [f for f in listdir(data_path) if isfile(join(data_path, f)) if file in f and id in f]
                try:
                    id_files[id][file] = join(data_path, cfile[0])
                except:
                    id_files[id][file] = []

            if id_files[id]['repeat']!=[]:
                mtx[id] = load_motif(id_files[id])
                logging.info("idx {}, id {}, len(mtx[id]) {}".format(idx, id, len(mtx[id])))
        with open('./tmp_data/participant_hr.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([id_set, mtx], f)
    else:

        with open('./tmp_data/participant_hr.pkl', 'rb') as f:
            [id_set, mtx] = pickle.load(f)

    '***evaluate mask***'
    eval_mask = np.ones((1, 24*60), np.float32)
    eval_mask[:, 16*60:16*60 + gap*60] = 0.

    dates = list(set([d for id in mtx for d in mtx[id]]))
    dic_dv = {d:Date2v(d) for d in dates}
    dict_user, reverse_dict_user = convert2idx(id_set, 1, is_set=False)
    dict_user['nan'] = 0; reverse_dict_user[0] = 'nan'
    # print (sorted(list(dic_dv.keys())))
    '*******************generate train data************************************'
    train_start, valid_start, tst_start, tst_end = dic_dv[train_start], dic_dv[valid_start], dic_dv[tst_start], dic_dv[tst_end]
    plus = '' if num_input<=6 else '_'+str(num_input)
    if not os.path.isfile( './tmp_data/central_id_tp_'+str(tp)+plus+'.pkl'):
        central_id = []
        for idx, id in enumerate(mtx):
            ds = list(mtx[id].keys())
            ds = [d for d in ds if dic_dv[d]>=train_start and dic_dv[d]<valid_start]
            if len(ds)>10:
                for _ in range(100):
                    sampled_keys = np.random.choice(ds, num_pos+num_input)
                    sampled_keys = sorted(sampled_keys)
                    central_id.append([(id, d) for d in sampled_keys])
        with open('./tmp_data/central_id_tp_'+str(tp)+plus+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(central_id, f)
    else:
        with open('./tmp_data/central_id_tp_'+str(tp)+plus+'.pkl', 'rb') as f:
            central_id = pickle.load(f)

    '******************generate exp instance******************************'
    exp_central_Image = []; exp_central_date = []; exp_date = []; exp_weekday = []
    for _ in central_id[:1]:
        tmp = [];
        tmp_d = [];
        tmp_date = [];
        tmp_week = []
        for (id, d) in _:
            tmp.append(mtx[id][d]);
            tmp_d.append(dic_dv[d])
            tmp_date.append(d)
            tmp_week.append(date2weekday(d))
        tmp_max_d = max(tmp_d)
        exp_central_Image.append(np.array(tmp))
        exp_central_date.append([tmp_max_d - _ for _ in tmp_d])
        exp_date.append(tmp_date)
        exp_weekday.append(tmp_week)
    return id_set, mtx, eval_mask, dict_user, reverse_dict_user, train_start, valid_start, \
           tst_start, tst_end, central_id, exp_central_Image, exp_central_date, exp_date, exp_weekday, dic_dv

    
def train_and_eval(alg, model_dir, output_file, n_epoch, batch_size, num_input, verbose, loss_type, train_ratio,
               id_set, mtx, eval_mask, dict_user, reverse_dict_user, train_start, valid_start,
               tst_start, tst_end, central_id,exp_central_Image, exp_central_date, exp_date, exp_weekday, dic_dv, log_file = ""):
    '*******************Train and evaluate************************************'
    label_micros = []
    label_macros = []
    label_weights = []

    pretrain_flag=0
    avg_loss = 0
    loss_list = [];

    alg.create(pretrain_flag=pretrain_flag, save_file=model_dir)
    for epoch_idx, epoch in enumerate(gen_epochs(n_epoch, central_id, batch_size, mtx, dic_dv)):
        new_log = create_file_log(file_name=log_file, filemode='a')
        for batch_central_Image, batch_central_date in epoch:
            mask = npmask(1, 60 * 24, 10 * 24)
            step, loss_, loss_d, loss_m = alg.feedbatch_daily(batch_central_Image, batch_central_date, mask, {})
            new_log.logger.debug("n_id {}, epoch {}, step{}, {}, {}".format(
                len(central_id), epoch_idx, step, batch_central_Image.shape, batch_central_date.shape))
            avg_loss += loss_
            new_log.logger.debug('step %s, loss %.3f, loss_d %.3f, loss_m %.3f '% (step, loss_, loss_d, loss_m))
            if step % 10 == 0:

                avg_loss /= 10
                loss_list.append(np.round(avg_loss, 3))
                avg_loss = 0
                if step %200 == 0:
                    if step == 200 and pretrain_flag == 0:
                        best_valid = eval_label(alg, eval_mask, 'train', mtx, dic_dv, valid_start, tst_start, num_input)
                        best_performance = eval_label(alg, eval_mask, 'test', mtx, dic_dv, tst_start, tst_end, num_input)
                        pretrain_flag = 2
                    else:
                        best_valid, best_performance = eval3_label(alg, best_performance, best_valid, eval_mask,
                                                                   mtx, dic_dv, valid_start, tst_start, tst_end, num_input)
                    

        new_log.logger.info('epoch  %s'%(epoch_idx))
        new_log.logger.info('valid {}'.format(best_valid[1]))
        new_log.logger.info('test {}'.format(best_performance[1]))
        new_log.reset_log()
        
        alg.save_weight(model_dir)
    return best_performance


class create_file_log():
    def __init__(self, file_name = 'my_log1', filemode = 'w', level = logging.DEBUG):
        formatter = logging.Formatter(fmt='%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s',
                                      datefmt='%Y/%m/%d %H:%M:%S')  # %I:%M:%S %p AM|PM format
        logging.basicConfig(filename='%s.log' % (file_name),
                            format='%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S', filemode=filemode, level=level)

        # console printer
        self.logger = logging.getLogger()
        self.screen_handler = logging.StreamHandler(sys.stdout)  # stream=sys.stdout is similar to normal print
        self.screen_handler.setFormatter(formatter)
        self.screen_handler.flush = sys.stdout.flush
        self.logger.addHandler(self.screen_handler)

    def flush_hander(self):
        self.screen_handler.flush()

    def clear_handler(self):
        self.logger.removeHandler(self.screen_handler)

    def reset_log(self):
        self.flush_hander()
        self.clear_handler()
        logging.shutdown()
