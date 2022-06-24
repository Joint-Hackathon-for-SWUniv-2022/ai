from requests import Response
from main import *

def get_tag(output):
    problem_cat= [
        ["greedy", "binary_search", "parametric_search", "ternary_search", "pbs", "bidirectional_search"], 
        ["bfs", "dfs"], 
        ["dp", "dp_tree", "dp_bitfield", "dp_connection_profile", "dp_deque"],
        ["string"],
        ["implementation"]
        ]
    
    tag_result= [0,0,0,0,0,0]

    for i in range(len(output)):
        for j in range(len(output['problem_tag'][i])):
            tf = 0
            for k in range(len(problem_cat)):
                if(output['problem_tag'][i][j] in problem_cat[k]):
                    tag_result[k] += i
                    tf = 1
            if(tf == 0):
                tag_result[5] += i
    
    max_score = max(tag_result)
    
    for i in range(len(tag_result)):
        tag_result[i] = 100 - tag_result[i] / max_score * 100

    return tag_result

def get_level(user_id):
    problem_cat= [
        ["greedy", "binary_search", "parametric_search", "ternary_search", "pbs", "bidirectional_search"], 
        ["bfs", "dfs"], 
        ["dp", "dp_tree", "dp_bitfield", "dp_connection_profile", "dp_deque"],
        ["string"],
        ["implementation"]
        ]

    user_data = pd.read_csv('data/user_data.csv')
    problem_data= pd.read_csv('data/problem_data.csv')

    for i in range(len(user_data)):
        if(user_data['user_id'][i] == user_id):
            s = user_data['problem_id'][i]
            s= re.sub("\"|\'|\[|\]","", s)
            s = s.split(', ')
            user_solved_problem = list(map(int, s))

    tag = [0,0,0,0,0,0]
    tag_result = []

    for i in range(len(user_solved_problem)):
        for j in range(len(problem_data)):
            if(user_solved_problem[i] == problem_data['problem_id'][j]):
                problem_tag = problem_data['problem_tag'][j]
                problem_tag = re.sub("\"|\'|\[|\]","", problem_tag)
                problem_tag = problem_tag.split(', ')
                tf = 0
                for k in range(len(problem_tag)):
                    for l in range(len(problem_cat)):
                        if(problem_tag[k] in problem_cat[l]):
                            tag[l] += problem_data['problem_level'][j] * 4 + 16
                            if(problem_data['problem_level'][j] >= 20):
                                tag[l] += 4
                            tf = 1
                    if(tf == 0):
                        tag[5] += problem_data['problem_level'][j] * 4 + 16
                        if(problem_data['problem_level'][j] >= 20):
                            tag[l] += 4
        
    tag_result.append(tag)

    return tag_result

from flask import Flask, jsonify
import json

app = Flask(__name__)

@app.route('/')
def infer():
    user_id= 'kny26023'

    output = general_pb_infer(user_id)
    # result = get_level(user_id)
    print(output)
    output = json.dumps(output)

    
    return jsonify(output)

app.run()