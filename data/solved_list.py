import json
from time import sleep
import requests
import sqlite3
import pandas as pd

def get_user_id(one_user_id):
    """
    정보 조회 - 백준 사이트에서 user의 프로필 정보 반환
    :return: 백준 프로필 정보
    :rtype: dict
    """
    profile=[]
    for page_id in range(140, 141):
        sleep(10)
        url = f"https://solved.ac/api/v3/ranking/tier?page={page_id}"
        users = requests.get(url)

        if(users.status_code == requests.codes.ok):
            users_list = json.loads(users.content.decode('utf-8'))
            users_profile = users_list["items"]
            for user_profile in users_profile:
                users_tier = user_profile["tier"]
                users_handle = user_profile["handle"]
                profile.append([users_tier, users_handle])
            print(f"{page_id} 성공")
        else:
            print("유저 정보 요청 실패")
            print(users.status_code)

    return profile


def get_profile(user_id):
    """
    정보 조회 - user_id를 입력하면 백준 사이트에서 해당 user의 프로필 정보 반환
    :param str user_id: 사용자 id
    :return: 백준 프로필 정보
    :rtype: dict
    """
    url = f"https://solved.ac/api/v3/user/show?handle={user_id}"
    user_profile = requests.get(url)
    profile= {}
    if(user_profile.status_code == requests.codes.ok):
        profile = json.loads(user_profile.content.decode('utf-8'))
        profile = \
            {
                "tier": profile.get("tier"),
                "class": profile.get("class"),
                "solvedCount": profile.get("solvedCount")
            }
    else:
        print("프로필 요청 실패")
    return profile


def check_user(user_id):
    url = f"https://solved.ac/api/v3/search/problem?query=solved_by%3A{user_id}&sort=level&direction=desc"
    user_solved = requests.get(url)
    if(user_solved.status_code == requests.codes.ok):
        solved = json.loads(user_solved.content.decode('utf-8'))
        count = solved.get("count")
        items = solved.get("items")
        pages = (count - 1) // 100 + 1
    else:
        print("푼 문제들 요청 실패")
        print(user_solved.status_code)
    return pages, items


def get_solved(user_id, pages, items):
    """
    정보 조회 - user_id를 입력하면 백준 사이트에서 해당 user가 푼 문제들 번호(level 높은 순)를 list로 반환
    :param items:
    :param pages:
    :param str user_id: 사용자id
    :return: 해당 user가 푼 문제들 번호
    :rtype: list
    """
    url = f"https://solved.ac/api/v3/search/problem?query=s%40{user_id}&sort=level&direction=desc"
    solved_problems = []
    for item in items:
        solved_problems.append(item.get("problemId"))
    for page in range(0, pages):
        sleep(5)
        page_url = f"{url}&page={page + 1}"
        user_solved = requests.get(page_url)
        if(user_solved.status_code == requests.codes.ok):
            sleep(5)
            solved = json.loads(user_solved.content.decode('utf-8'))
            items = solved.get("items")
            for item in items:
                solved_problems.append(item.get("problemId"))
            print(f"{page+1}번째 페이지 성공")
        else:
            print("푼 문제들 요청 실패")
            print(user_solved.status_code)
            print(url)
    return len(solved_problems), solved_problems


def get_problem_by_level(level):
    """
    정보 조회 - level을 입력하면 백준 사이트에서 해당 level에 해당하는 문제들을 반환
    :param level:
    :return: 해당 level의 문제들
    """
    url = f"https://solved.ac/api/v3/search/problem?query=tier%3A{level}"
    user_level_problem = requests.get(url)
    if(user_level_problem.status_code == requests.codes.ok):
        level_problem = json.loads(user_level_problem.content.decode('utf-8'))
        pages = (level_problem.get("count") - 1) // 100 + 1
    else:
        print("난이도별 문제 요청 실패")

    problems = set()
    for page in range(pages):
        page_url = f"{url}&page={page + 1}"
        print(page_url)
        user_level_problem = requests.get(page_url)
        if(user_level_problem.status_code == requests.codes.ok):
            level_problem = json.loads(user_level_problem.content.decode('utf-8'))
            items = level_problem.get("items")
            for item in items:
                problems.add(item.get("problemId"))
        else:
            print("난이도별 문제 요청 실패")
    return problems


def find_problem(problem_id):
    """
    정보 조회 - problem_id를 입력하면 백준 사이트에서 해당 문제의 tag를 반환
    :param level:
    :return: 해당 problem의 tag
    """
    url= f"https://solved.ac/api/v3/problem/show?problemId={problem_id}"
    user_problem = requests.get(url)
    level = -1
    tag_list = []
    average_tries= -1
    if(user_problem.status_code == requests.codes.ok):
        problem = json.loads(user_problem.content.decode('utf-8'))
        level = problem.get("level")
        tags = problem.get("tags")
        average_tries= problem.get("averageTries")
        is_solvable= problem.get("isSolvable")
        if(len(tags) == 0):
            print("태그 없음")
        else:
            for tag in tags:
                tag_list.append(tag['key'])
    else:
        print("ID별 문제 요청 실패")
        print(f"{user_problem.status_code}")

    return level, is_solvable, average_tries, tag_list


def get_user(one_user_id):
    users_profile= get_user_id(one_user_id)
    users_solvedList= []
    solvedList= []
    for user_profile in users_profile:
        sleep(5)
        user_tier= user_profile[0]
        user_id = user_profile[1]
        print(f"========{user_id}님의 프로필========")
        print("check_user start")
        pages, items = check_user(user_id)
        sleep(5)
        print("get_solved start")
        count, solved_list = get_solved(user_id, pages, items)
        users_solvedList.append([user_id, user_tier, solved_list])
        for i in solved_list:
            if(i not in solvedList):
                solvedList.append(i)

    user_data= pd.DataFrame(
        users_solvedList, columns=["user_id", "tier", "problem_id"]
    )

    user_data.to_csv('data/user_data.csv', index=False)
    solvedList.sort()

    return users_solvedList, solvedList


def get_problem(solvedList):
    problem_cat= [
        ["binary_search", "parametric_search", "ternary_search", "pbs", "bidirectional_search"], 
        ["bfs", "dfs"], 
        ["dp", "dp_tree", "dp_bitfield", "dp_connection_profile", "dp_deque"],
        ["string"],
        ["implementation"]
        ]
    problem= []
    for problem_id in solvedList:
        level, is_solvable, average_tries, tags = find_problem(problem_id)
        sleep(10)
        if(level != -1):
            tf=0
            for i in range(5):
                for j in range(len(problem_cat[i])):
                    if(problem_cat[i][j] in tags):
                        problem.append([problem_id, level, is_solvable, average_tries, tags])
                        tf = 1
                        break             
                if(tf == 1):
                    break  
            print(f"{problem_id}번 문제 태깅 완료!")

    problem_data= pd.DataFrame(
        problem, columns= ["problem_id", "problem_level", "problem_is_solvable", "problem_average_tries", "problem_tag"]
    )

    problem_data.to_csv('data/problem_data.csv', index=False)

    return problem


def get_solved_problem(user_solvedList, problem):
    problems= []
    for user_ids, tier, problem_ids in user_solvedList:
        for user_id in user_ids:
            for id in problem_ids:
                for problem_id, level, is_solvable, average_tries, tags in problem:
                    if(id ==  problem_id):
                        problems.append([user_id, id])
                        break
    
    problem_data= pd.DataFrame(
        problems, columns=["user_id", "problem_id"]
    )

    problem_data.to_csv('data/solved_data.csv', index=False)

def get_solved_list():
    user_list= pd.read_csv("data/user_data.csv")
    users_solvedList= []
    solvedList= []
    for i in range(len(user_list)):
        id= user_list['user_id']
        tier= user_list['tier']
        problem_id= list(map(int, user_list['problem_id'][i].strip('[').strip(']').split(", ")))
        users_solvedList.append([id, tier, problem_id])
        for problem in problem_id:
            if(problem not in solvedList):
                solvedList.append(problem)
        print(f"{i}번째 유저 통과")
    
    solvedList.sort()

    return users_solvedList, solvedList

def get_problem_list():
    problem_list = pd.read_csv("data/problem_data.csv")
    problem = []
    for i in range(len(problem_list)):
        problem_id = int(problem_list['problem_id'][i])
        problem_level = int(problem_list['problem_level'][i])
        problem_is_solvable = problem_list['problem_is_solvable'][i]
        problem_average_tries = int(problem_list['problem_average_tries'][i])
        problem_tag = problem_list['problem_tag'][i].strip('[').strip(']').split(", ")
        problem.append([problem_id, problem_level, problem_is_solvable, problem_average_tries, problem_tag])
    
    problem_data= pd.DataFrame(
        problem, columns=["problem_id", "problem_level", "problem_is_solvable", "problem_average_tries", "problem_tag"]
    )

    problem_data.to_csv('data/solved_data.csv', index=False)

    return problem


#user_solvedList = get_user()
users_solvedList, solvedList = get_solved_list()
print(len(solvedList))
#problem = get_problem(solvedList)
problem = get_problem_list()
get_solved_problem(users_solvedList, problem)