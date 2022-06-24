import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy import sparse
import os
from dataset import *
from utils import *
import json
from tqdm import tqdm
from model import *
import datetime
import warnings
warnings.filterwarnings(action = 'ignore')
import bottleneck as bn

## 각종 파라미터 세팅
from config import *

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

def infer_all(raw_data):
    ## 배치사이즈 포함
    ### 데이터 준비
    # Load Data
    pro_dir = os.path.join(args.dataset, 'pro_sg')

    with open(pro_dir + '/model_score.json', 'r', encoding="utf-8") as f:
        model_score = json.load(f)

    raw_data, _, _ = filter_triplets(raw_data, 5, 10)
    df_problems = pd.read_csv('data/problem_data.csv')
    df_problems.drop(df_problems[df_problems['problem_average_tries'] == 7340].index, axis=0, inplace=True)
    # level 0에 해당하는 문제 제거
    df_problems = df_problems[df_problems['problem_level'] != 0]
    # not_solvable == False만
    df_problems = df_problems[df_problems['problem_is_solvable'] == True]
    # tag가 nan인 문제 제거
    df_problems = df_problems[~df_problems['problem_tag'].isnull()]

    raw_data = raw_data[raw_data['problem_id'].isin(df_problems['problem_id'].values)].reset_index(drop=True)
    
    device = torch.device("cuda" if args.cuda else "cpu")

    # Import Data
    print("Inference Start!!")

    with open(pro_dir + '/problem2id.json', 'r', encoding="utf-8") as f:
        problem2id = json.load(f)

    with open(pro_dir + '/user2id.json', 'r', encoding="utf-8") as f:
        user2id = json.load(f)

    infer_df = numerize_for_infer(raw_data, user2id, problem2id)

    loader = DataLoader(args.dataset)
    n_items = loader.load_n_items()
    n_users = infer_df['uid'].max() + 1

    rows, cols = infer_df['uid'], infer_df['pid']
    data = sparse.csr_matrix((np.ones_like(rows),(rows, cols)), dtype='float64', shape=(n_users, n_items))

    num_data = data.shape[0]
    index_list = list(range(num_data))

    df_users = pd.read_csv('data/user_data.csv')
    df_users = df_users[['user_id', 'tier']]
    df_user_lv = df_users[df_users['user_id'].isin(raw_data['user_id'].unique())].reset_index(drop=True)

    dict_user_lv = dict()
    for i in range(len(df_user_lv)):
        dict_user_lv[df_user_lv.iloc[i,0]] = df_user_lv.iloc[i, 1]

    dict_problem_lv = dict()
    for i in range(len(df_problems)):
        dict_problem_lv[df_problems.iloc[i, 0]] = df_problems.iloc[i,1]

    id2user = dict((int(v), k) for k, v in user2id.items())
    id2problem = dict((int(v), k) for k, v in problem2id.items())

    model_kwargs = {
    'hidden_dim': args.hidden_dim,
    'latent_dim': args.latent_dim,
    'input_dim': data.shape[1]
    }

    model_name = "vae"
    #score = 0
    #for m, s in model_score.items():
    #    if s > score:
    #        model_name = m
    #        score = s
    #print(f"Best Model => {model_name} : {score:.4f}")

    if model_name == 'vae':
        p_dims = [200, 3000, n_items] # [200, 600, 6807]
        problem_tag_emb = pd.read_csv(pro_dir + '/problem_tag_emb.csv')

        model = MultiVAE(p_dims, tag_emb=problem_tag_emb)

        with open(os.path.join(args.save_dir, 'best_vae_' + args.save), 'rb') as f:
            model.load_state_dict(torch.load(f))
        model = model.to(device)

    #elif model_name == 'dae':
    #    p_dims = [200, 3000, n_items]  # [200, 600, 6807]
    #    problem_tag_emb = pd.read_csv(pro_dir + '/problem_tag_emb.csv')
#
    #    model = MultiDAE(p_dims, tag_emb=problem_tag_emb).to(device)
#
    #    with open(os.path.join(args.save_dir, 'best_dae_' + args.save), 'rb') as f:
    #        model.load_state_dict(torch.load(f))
    #    model = model.to(device)

    model.eval()

    # Infer
    pred_list = None

    with torch.no_grad():
        for start_index in tqdm(range(0, num_data, args.batch_size)):
            end_index = min(start_index + args.batch_size, num_data)
            data_batch = data[index_list[start_index:end_index]]
            data_tensor = naive_sparse2tensor(data_batch).to(device)

            if model_name == 'vae':
                recon_batch, _, _ = model(data_tensor)

            recon_batch = recon_batch.cpu().numpy()

            recon_batch[data_batch.nonzero()] = -np.inf

            idx = bn.argpartition(-recon_batch, 100, axis=1)[:, :100]
            out = np.array([])

            for user, problem in enumerate(idx):
                user += start_index
                user = id2user[user]

                infer_out = infer(user, problem, dict_user_lv, dict_problem_lv, id2problem, args.infer_cnt)
                out = np.append(out, infer_out)


            if start_index == 0:
                pred_list = out
            else:
                pred_list = np.append(pred_list, np.array(out))

    user2 = []
    problem2 = []

    pred_list = pred_list.reshape(num_data, -1)

    for user_index, pred_item in enumerate(pred_list):
        user2.extend([user_index] * args.infer_cnt)
        problem2.extend(pred_item)

    u2 = pd.DataFrame(user2, columns=['user_id'])
    i2 = pd.DataFrame(problem2, columns=['problem_id'])
    all2 = pd.concat([u2, i2], axis=1)

    ans2 = de_numerize(all2, id2user, id2problem)
    ans2.columns = ['user_id', 'problem_id']
    new_ans2 = ans2.sort_values('user_id')

    new_ans2.to_csv('models/new_output.csv', index=False)

    # print('Output Saved!!')
    return new_ans2