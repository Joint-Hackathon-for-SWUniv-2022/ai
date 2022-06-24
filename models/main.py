from preprocessing import preprocessing_all
from inference import infer_all
from train_multivae import train_vae
import pandas as pd
import numpy as np
import os
from config import args
import re

def make_train_data(df_problems_solved):
    #df_problems_solved['problems_id'] = df_problems_solved['problem_id'].str.split(',')
    #df_problems_solved = df_problems_solved.replace(['', 'null'], [np.nan, np.nan])
    #df_problems_solved = df_problems_solved.explode('problem_id').dropna().reset_index(drop=True)
    #df_problems_solved = df_problems_solved.replace(['', 'null'], [np.nan, np.nan])
    #df_problems_solved = df_problems_solved.dropna()
    #df_problems_solved = df_problems_solved.drop(columns=['id'])
    df_problems_solved = df_problems_solved.astype({'user_id':'str', 'problem_id':'int'})
    df_problems_solved.columns = ['user_id', 'problem_id']
    return df_problems_solved


def general_problem_preprocessing():
    df_problems_solved = pd.read_csv('data/solved_data.csv')
    raw_data = make_train_data(df_problems_solved)

    print("Preprocessing Start!!")
    preprocessing_all(raw_data)


def vae_train():
    print("Train Start!!")
    train_vae()


#def dae_train():
#    print("Train Start!!")
#    train_dae()


def general_pb_infer(user_id):
    df_problems_solved = pd.read_csv('data/solved_data.csv')
    raw_data = make_train_data(df_problems_solved)

    print("Inference Start!!")
    output = infer_all(raw_data)

    output = output.reset_index()
    output = output.drop(['index'], axis=1)

    temp =[]
    tag_list = []
    problem_list = pd.read_csv("data/problem_data.csv")

    for i in range(len(output)):
        t= int(output['problem_id'][i])
        temp.append(t)

    for i in range(len(temp)):
        for j in range(len(problem_list)):
            if(problem_list['problem_id'][j] == temp[i]):
                tag = problem_list['problem_tag'][j]
                tag= re.sub("\"|\'|\[|\]","", tag)
                tag_ll = tag.split(", ")
                tag_list.append(tag_ll)

    output['problem_tag'] = tag_list

    temp_output = []

    for i in range(len(output)):
        if(user_id == output['user_id'][i]):
            problem_id = output['problem_id'][i]
            problem_tag= output['problem_tag'][i]
            temp_output.append([user_id, problem_id, problem_tag])
    
    final_output = pd.DataFrame(
        temp_output, columns= ['user_id', 'problem_id', 'problem_tag']
    )

    final_output.to_csv(os.path.join(args.dataset, 'final_output.csv'), index=False)

    return temp_output