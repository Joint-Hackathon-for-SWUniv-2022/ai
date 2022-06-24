import time
import torch
import numpy as np
from dataset import *
from utils import *
import json
import pandas as pd
from torch import nn
import csv

import os
import warnings
warnings.filterwarnings(action='ignore')

## 각종 파라미터 세팅
from config import *

# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)

def preprocessing_all(raw_data):
    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    device = torch.device("cuda" if args.cuda else "cpu")

    print("Load and Preprocess BOJ dataset")
    # Load Data
    # Filter Data
    raw_data, user_activity, item_popularity = filter_triplets(raw_data, 5, 10)
    #---------------------------
    # 여기에 tag == None인 아이템은 제외
    df_problems = pd.read_csv('data/problem_data.csv')

    for i in range(len(df_problems)):
        df_problems['problem_id'][i] = int(df_problems['problem_id'][i])
        df_problems['problem_level'][i] = int(df_problems['problem_level'][i])
        df_problems['problem_average_tries'][i] = int(df_problems['problem_average_tries'][i])
        df_problems['problem_level'][i] = int(df_problems['problem_level'][i])
        if(df_problems['problem_tag'][i] == "[]"):
            df_problems['problem_tag'][i] = None
        else:
            df_problems['problem_tag'][i] = df_problems['problem_tag'][i].strip('[').strip(']').split(", ")

    df_problems.drop(df_problems[df_problems['problem_average_tries'] == 7340].index, axis=0, inplace=True)
    # level 0에 해당하는 문제 제거
    df_problems = df_problems[df_problems['problem_level'] != 0]
    # not_solvable == False만
    df_problems = df_problems[df_problems['problem_is_solvable'] == True]
    # tag가 nan인 문제 제거
    df_problems = df_problems[~df_problems['problem_tag'].isnull()]

    #---------------------------

    # Shuffle User Indices
    unique_uid = pd.unique(raw_data['user_id'])
    print("(BEFORE) unique_uid:", unique_uid)
    np.random.seed(args.seed)
    idx_perm = np.random.permutation(unique_uid.size) # 해당 숫자까지의 인덱스를 무작위로 섞은 것을 arr로 반환
    unique_uid = unique_uid[idx_perm]
    print("(AFTER) unique_uid:", unique_uid) # 무작위로 item을 섞음

    n_users = unique_uid.size
    n_heldout_users = 2

    # Split Train/Validation/Test User Indices
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]

    #주의: 데이터의 수가 아닌 사용자의 수입니다!
    print("훈련 데이터에 사용될 사용자 수:", len(tr_users))
    print("검증 데이터에 사용될 사용자 수:", len(vd_users))
    print("테스트 데이터에 사용될 사용자 수:", len(te_users))

    ##훈련 데이터에 해당하는 아이템들
    #Train에는 전체 데이터를 사용합니다.
    train_plays = raw_data.loc[raw_data['user_id'].isin(tr_users)]

    ##문제 ID
    unique_pid = pd.unique(raw_data['problem_id'])

    pro_dir = os.path.join(args.dataset, 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    problem2id = dict((int(pid), int(i)) for (i, pid) in enumerate(unique_pid)) # item2idx dict
    user2id = dict((uid, int(i)) for (i, uid) in enumerate(unique_uid)) # user2idx dict

    with open(os.path.join(pro_dir, 'problem2id.json'), 'w', encoding="utf-8") as f:
        json.dump(problem2id, f, ensure_ascii=False, indent="\t")
        
    with open(os.path.join(pro_dir, 'user2id.json'), 'w', encoding="utf-8") as f:
        json.dump(user2id, f, ensure_ascii=False, indent="\t")

    with open(os.path.join(pro_dir, 'unique_pid.txt'), 'w', newline="") as f:
        i=0
        for pid in unique_pid:
            if(i== len(unique_pid) -1):
                f.write('%s' % pid)
            else:
                f.write('%s\n' % pid)
                i+=1

    with open(os.path.join(pro_dir, 'unique_uid.txt'), 'w') as f:
        i=0
        for uid in unique_uid:
            if(i== len(unique_uid)-1):
                f.write('%s' % uid)
            else:
                f.write('%s\n' % uid)
                i+=1

    #Validation과 Test에는 input으로 사용될 tr 데이터와 정답을 확인하기 위한 te 데이터로 분리되었습니다.
    print('Data Split Start!')
    vad_users = raw_data.loc[raw_data['user_id'].isin(vd_users)]
    vad_users = vad_users.loc[vad_users['problem_id'].isin(unique_pid)]
    vad_users_tr, vad_users_te = split_train_test_proportion(vad_users)

    test_users = raw_data.loc[raw_data['user_id'].isin(te_users)]
    test_users = test_users.loc[test_users['problem_id'].isin(unique_pid)]
    test_users_tr, test_users_te = split_train_test_proportion(test_users)

    train_data = numerize(train_plays, user2id, problem2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_users_tr, user2id, problem2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_users_te, user2id, problem2id)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_users_tr, user2id, problem2id)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_users_te, user2id, problem2id)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
    print("Data Split Done!")

    # item_tag_emb
    print("Tag emb start!")
    set_tags = set()
    for tags in df_problems['problem_tag'].dropna().values:
        for tag in tags:
            if(tag not in set_tags):
                set_tags.add(tag)


    df_tags = df_problems[['problem_id', 'problem_tag']]
    df_tags = df_tags.explode('problem_tag').dropna().reset_index(drop=True)
    df_tags = df_tags[df_tags['problem_id'].isin(unique_pid)].reset_index(drop=True)

    temp = nn.Embedding(len(set_tags), 300)
    tag_emb = pd.DataFrame(df_tags['problem_tag'].value_counts().index.values, columns=['problem_tag'])

    dict_tag_idx = dict()
    for i, j in enumerate(df_tags['problem_tag'].value_counts().index.values):
        dict_tag_idx[j] = i

    list_emb = []
    dict_tag_emb = dict()
    for i in df_tags['problem_tag'].value_counts().index.values:
        list_emb.append(temp(torch.tensor(dict_tag_idx[i])).detach().numpy())
        dict_tag_emb[i] = temp(torch.tensor(dict_tag_idx[i])).detach().numpy()

    df_tag_emb = pd.concat([tag_emb, pd.DataFrame(list_emb)], axis=1)
    df_tags2 = pd.merge(df_tags, df_tag_emb, on='problem_tag', how='left')
    tag2emb = df_tags2.iloc[:, 2:].values
    df_tags['emb'] = list(tag2emb) 

    total = []

    def item_problem_emb_mean(i):
        total.append(np.mean(df_tags[df_tags['problem_id'] == i].emb))

    print(len(df_tags['problem_id'].unique()))

    item_problem_emb_idx = pd.DataFrame(list(df_tags['problem_id'].unique()), columns=['problem_id'])
    item_problem_emb_idx.problem_id.apply(lambda x: item_problem_emb_mean(x))
    item_problem_emb = pd.DataFrame(total)
    item_problem_emb.index = df_tags['problem_id'].unique()

    item_problem_emb = item_problem_emb.reset_index()
    item_problem_emb['index'] = item_problem_emb['index'].apply(lambda x : problem2id[x])
    item_problem_emb = item_problem_emb.set_index('index')
    item_problem_emb = item_problem_emb.sort_index()

    item_problem_emb = item_problem_emb.T
    print(item_problem_emb.shape)

    item_problem_emb.to_csv(os.path.join(pro_dir, 'problem_tag_emb.csv'), index=False)

    print('Problem tag emb Done!')

    model_score = dict()
    model_score['vae'] = 0
    model_score['dae'] = 0
    with open(os.path.join(pro_dir, 'model_score.json'), 'w', encoding="utf-8") as f:
        json.dump(model_score, f, ensure_ascii=False, indent="\t")
    #--------------------------------------------