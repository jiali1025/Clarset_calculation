import numpy as np
import xlrd
from itertools import combinations
from itertools import product
import random
import easygraphics.dialog as dlg
import tkinter as tk
import sys
from 封面图_png import img as surface_graph
from 开放结构_png import img as open_shell
from 封闭结构_png import img as close_shell
import base64
import os
from tabulate import tabulate


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# 读excel数据转为矩阵函数
def CarbonMatrix(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]  # 获取excel中第一个sheet表
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows, ncols))
    for x in range(ncols):
        cols = table.col_values(x)
        cols1 = np.matrix(cols)  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x] = cols1  # 把数据进行存储
    return datamatrix, nrows, ncols

def nested_list_to_tuple_recur(nested_list):  # 嵌套列表转化为嵌套元组便于集合筛选出重复序列
    return tuple(
        nested_list_to_tuple_recur(l) if isinstance(l, list)
        else l for l in nested_list
    )

def radical_carbon_accurate(path,carbon_loop,carbon_reloop):
    long_carbon_loop = int(len(carbon_loop) / 2)
    m1, row_1, col_1 = CarbonMatrix(path)
    print(m1, row_1, col_1)  # 读入碳位点矩阵
    m1 = m1.astype(np.uint8)  # 将其格式转化为整数
    #将所有结果均记录在result中，位置在桌面
    sys.stdout = Logger('C:\\tool.box-result\\open-shell-exact-result.txt')
    # 将有可能出现自由基的C位点全部标出
    two_carbon = []
    for i in range(row_1):
        if sum(m1[i, :]) == 2:
            two_carbon.append(i)
    print(f"Possible free radical locations:")
    print(two_carbon)
    print("\n")

    # 读入矩阵为1处的坐标位置
    m2 = np.triu(m1, 1)  # 由于此矩阵为对称矩阵，所以我们只用截取上半部分的进行后续操作
    a_list = []  # 创建空列表储存碳键位为1
    if row_1 == col_1:
        for i in range(row_1):
            for j in range(col_1):
                if m2[i, j] == 1:
                    a_list.append([i, j])
                    sorted(a_list)
    else:
        dlg.show_message(f"The number of rows and columns of the input matrix is not equal. Please re-enter")

    # print(a_list),a_list输出没问题

    # 自由基选点处，从16个碳位点中选取2个作为自由基，并在矩阵读入时不考虑已被选为自由点的坐标

    changdu = set()#记录不同自由基位点时，list的长度
    for p1 in combinations(two_carbon, 2):  # 生成两个自由基的所有组合
        c_list = []
        for i in range(row_1):
            if i == p1[0] or i == p1[1]:
                continue
            else:
                for j in range(col_1):
                    if j == p1[0] or j == p1[1]:
                        continue
                    else:
                        if m2[i, j] == 1:
                            c_list.append([i, j])
                            sorted(c_list)
        # print(c_list) # 除去自由基点外的键位

        row_tiqu2 = [[] for x in range(row_1)]
        for i in range(row_1):
            for j in range(len(c_list)):
                if c_list[j][0] == i:
                    row_tiqu2[i].append(c_list[j])

        list2 = row_tiqu2  # 将嵌套列表中的空列表删去
        while list2.__contains__([]):
            list2.remove([])  # 例如通过len(list(2)):32-35，确定下面的循环的范围
        cd = len(list2)
        changdu.add(cd)

    print(changdu)
    c_max = max(changdu)
    c_min = min(changdu)
    count_list = []
    count_zong = 0
    benhuan_count = []
    for p1 in combinations(two_carbon, 2):  # 生成两个自由基的所有组合
        count_1 = 0
        c_list = []
        for i in range(row_1):
            if i == p1[0] or i == p1[1]:
                continue
            else:
                for j in range(col_1):
                    if j == p1[0] or j == p1[1]:
                        continue
                    else:
                        if m2[i, j] == 1:
                            c_list.append([i, j])
                            sorted(c_list)

        # print(c_list) # 除去自由基点外的键位

        # 按行封装碳点
        row_tiqu2 = [[] for x in range(row_1)]
        for i in range(row_1):
            for j in range(len(c_list)):
                if c_list[j][0] == i:
                    row_tiqu2[i].append(c_list[j])

        list2 = row_tiqu2  # 将嵌套列表中的空列表删去
        print(row_tiqu2)
        while list2.__contains__([]):
            list2.remove([])  # 通过len(list(2)):32-35，确定if的范围

        for i in range(c_min, c_max + 1):
            if i == len(list2):
                zz_list2 = []
                for p in product(*list2):  # 需要变动的地方
                    b_list2 = []
                    cc_list2 = []
                    for i in range(len(p)):
                        if p[i][0] in cc_list2 or p[i][1] in cc_list2:
                            continue
                        else:
                            b_list2.append(p[i])
                        cc_list2.append(p[i][0])
                        cc_list2.append(p[i][1])
                    if len(b_list2) == (row_1 - 2) / 2:
                        zz_list2.append(b_list2)

                zj = nested_list_to_tuple_recur(zz_list2)
                jw = set(zj)

                for i2 in jw:
                    s = 0
                    i2 = set(i2)
                    c_list2 = []
                    for n in range(long_carbon_loop):
                        if carbon_loop[2 * n].issubset(i2) or carbon_loop[2 * n + 1].issubset(i2):
                            c_list2.append(n + 1)

                    print(f"Location of free radical combination:{p1}")
                    print(i2)
                    print(f"The {c_list2} ring is a benzene ring structure")

                    c_list2 = sorted(c_list2)

                    s2 = set()
                    for h in range(7):
                        random.shuffle(c_list2)
                        for i in c_list2:  # 这里非常奇怪，为什么输出不全，所以我们使用random打乱后输出
                            zh_list = []
                            zh_list.append(i)
                            for j in c_list2:
                                flag = 0
                                for x in zh_list:
                                    if j in carbon_reloop[x - 1]:
                                        flag = flag + 1
                                if flag == 0:
                                    zh_list.append(j)
                            zh_list = sorted(zh_list)
                            zh_list = tuple(zh_list)
                            s2.add(zh_list)

                    print("Combinations of benzene ring structures that conform to the rules may include:")
                    print(s2)

                    long = 0
                    for l in s2:
                        if len(l) > long:
                            long = len(l)
                        else:
                            continue
                    print(f"The maximum Clar number for this structure is {long}")
                    benhuan_count.append(long)
                    count_1 = count_1 + len(s2)
                    count_zong = count_zong + len(s2)
                count_list.append(count_1)

                print(f"Radical group {p1} has a total number of combinations {count_1}")
                print("\n")

    count_percent = []
    for i in count_list:
        percent = i / count_zong
        count_percent.append(percent)

    print(f"The total number of combinations is {count_zong}")
    result = list(combinations(two_carbon, 2))
    print(f"All of the free radical groups are:")
    print(result)
    print(f"The combination number of the corresponding free radical group is:")
    print(count_list)
    print("The percentage of the combination of the corresponding free radical group is:")
    print(count_percent)

    total_atom = row_1
    atom = [*range(total_atom)]
    atom_count = []

    for element in atom:
        single_count = []
        for i in range(len(result)):
            if element in result[i]:
                single_count.append(count_list[i])
        atom_count.append(sum(single_count))
    '''
    radical probability at atom position
    '''
    total_radical = sum(atom_count)
    prob = [n / total_radical for n in atom_count]
    prob = [i * 100 for i in prob]
    b = []
    for j in prob:
        bb = "%.2f%%" % j
        b.append(bb)
    print(f"The percentage of free radicals at each carbon site is the probability of 0~{row_1-1}")
    print(b)
    print(f"The maximum Clar number in all combinations is {max(benhuan_count)}")
    info = {'Carbon site': ["carbon" + str(i) for i in range(row_1)],
            'probability of free radical': b}
    print(tabulate(info, headers='keys', tablefmt='fancy_grid'))
    with open('C:\\tool.box-result\\open-shell-carbon-exact-site-probability.txt', 'w+') as file0:
        print(tabulate(info, headers='keys', tablefmt='fancy_grid'), file=file0)
    dlg.show_message(f"The total number of combinations is:{count_zong}\n"
                     f"The percentage of free radicals at each carbon site is the probability of 0~{row_1-1}:\n{b}\n"
                     f"The maximum Clar number in all combinations is {max(benhuan_count)}")

def full_carbon_accurate(path,carbon_loop,carbon_reloop):
    sys.stdout = Logger('C:\\tool.box-result\\close-shell-exact-result.txt')
    long_carbon_loop = int(len(carbon_loop) / 2)
    m1, row_1, col_1 = CarbonMatrix(path)  # 读入碳位点数据
    print(m1, row_1, col_1)
    m1 = m1.astype(np.uint8)  # 将其格式转化为整数
    m2 = np.triu(m1, 1)  # 由于此矩阵为对称矩阵，所以我们只用截取上半部分的进行后续操作
    a_list = []  # 创建空列表储存碳键位为1
    count_benhuan = []
    # 读入矩阵为1处的坐标位置
    for i in range(row_1):
        for j in range(col_1):
            if m2[i, j] == 1:
                a_list.append([i, j])
                sorted(a_list)
    print(a_list)
    count_zong = 0
    # 按行封装碳键数
    row_tiqu2 = [[] for x in range(row_1)]
    for i in range(row_1):
        for j in range(len(a_list)):
            if a_list[j][0] == i:
                row_tiqu2[i].append(a_list[j])

    list2 = row_tiqu2  # 将嵌套列表中的空列表删去
    while list2.__contains__([]):
        list2.remove([])  # 通过len(list(2)):32-35，确定if的范围
    print(list2)

    zz_list1 = []
    for p in product(*list2):
        b_list2 = []
        cc_list2 = []
        for i in range(len(p)):
            if p[i][0] in cc_list2 or p[i][1] in cc_list2:
                continue
            else:
                b_list2.append(p[i])
            cc_list2.append(p[i][0])
            cc_list2.append(p[i][1])
        if len(b_list2) == row_1 / 2:
            zz_list1.append(b_list2)

    zj = nested_list_to_tuple_recur(zz_list1)
    jw = set(zj)
    print('Close-shell structure benzene ring location:\n')

    for i2 in jw:
        i2 = set(i2)
        c_list2 = []
        for n in range(long_carbon_loop):
            if carbon_loop[2 * n].issubset(i2) or carbon_loop[2 * n + 1].issubset(i2):
                c_list2.append(n + 1)
        i2 = list(i2)
        i2 = sorted(i2)
        print(i2)
        print(f"The {c_list2} ring is a benzene ring structure")
        c_list2 = sorted(c_list2)

        s2 = set()
        for h in range(7):
            random.shuffle(c_list2)
            for i in c_list2:  # 这里非常奇怪，为什么输出不全，所以我们使用random打乱后输出
                zh_list = []
                zh_list.append(i)
                for j in c_list2:
                    flag = 0
                    for x in zh_list:
                        if j in carbon_reloop[x - 1]:
                            flag = flag + 1
                    if flag == 0:
                        zh_list.append(j)
                zh_list = sorted(zh_list)
                zh_list = tuple(zh_list)
                s2.add(zh_list)

        print(s2)
        s2 = list(s2)
        count_zong+=len(s2)
        long = 0
        for l in s2:
            if len(l) > long:
                long = len(l)
            else:
                continue
        print(f"The maximum Clar number for this structure is {long}")
        count_benhuan.append(long)
    print(count_benhuan)
    print(f"The structure is close-shell structure and the maximum Clar number for this structure is {max(count_benhuan)}")
    dlg.show_message(f"The total number of combinations is:{count_zong}\n"
                     f"The maximum Clar number for this structure is {max(count_benhuan)}")

def radical_carbon_random_sample(path,carbon_loop,carbon_reloop):
    sys.stdout = Logger('C:\\tool.box-result\\open-shell-approximate-result.txt')
    long_carbon_loop = int(len(carbon_loop) / 2)
    m1, row_1, col_1 = CarbonMatrix(path)
    print(m1, row_1, col_1)  # 读入碳位点矩阵
    m1 = m1.astype(np.uint8)  # 将其格式转化为整数
    # 将有可能出现自由基的C位点全部标出
    two_carbon = []
    for i in range(row_1):
        if sum(m1[i, :]) == 2:
            two_carbon.append(i)
    print(f"Possible free radical locations:")
    print(two_carbon)
    print("\n")

    # 读入矩阵为1处的坐标位置
    m2 = np.triu(m1, 1)  # 由于此矩阵为对称矩阵，所以我们只用截取上半部分的进行后续操作
    a_list = []  # 创建空列表储存碳键位为1
    if row_1 == col_1:
        for i in range(row_1):
            for j in range(col_1):
                if m2[i, j] == 1:
                    a_list.append([i, j])
                    sorted(a_list)
    else:
        dlg.show_message(f"The number of rows and columns of the input matrix is not equal. Please re-enter")

    # print(a_list),a_list输出没问题

    # 自由基选点处，从16个碳位点中选取2个作为自由基，并在矩阵读入时不考虑已被选为自由点的坐标

    changdu = set()  # 记录不同自由基位点时，list的长度
    for p1 in combinations(two_carbon, 2):  # 生成两个自由基的所有组合
        c_list = []
        for i in range(row_1):
            if i == p1[0] or i == p1[1]:
                continue
            else:
                for j in range(col_1):
                    if j == p1[0] or j == p1[1]:
                        continue
                    else:
                        if m2[i, j] == 1:
                            c_list.append([i, j])
                            sorted(c_list)
        # print(c_list) # 除去自由基点外的键位

        row_tiqu2 = [[] for x in range(row_1)]
        for i in range(row_1):
            for j in range(len(c_list)):
                if c_list[j][0] == i:
                    row_tiqu2[i].append(c_list[j])

        list2 = row_tiqu2  # 将嵌套列表中的空列表删去
        while list2.__contains__([]):
            list2.remove([])  # 例如通过len(list(2)):32-35，确定下面的循环的范围
        cd = len(list2)
        changdu.add(cd)

    print(changdu)
    c_max = max(changdu)
    c_min = min(changdu)
    count_list = []
    count_zong = 0
    benhuan_count = []
    for p1 in combinations(two_carbon, 2):  # 生成两个自由基的所有组合
        count_1 = 0
        c_list = []
        for i in range(row_1):
            if i == p1[0] or i == p1[1]:
                continue
            else:
                for j in range(col_1):
                    if j == p1[0] or j == p1[1]:
                        continue
                    else:
                        if m2[i, j] == 1:
                            c_list.append([i, j])
                            sorted(c_list)

        # print(c_list) # 除去自由基点外的键位

        # 按行封装碳点
        row_tiqu2 = [[] for x in range(row_1)]
        for i in range(row_1):
            for j in range(len(c_list)):
                if c_list[j][0] == i:
                    row_tiqu2[i].append(c_list[j])

        list2 = row_tiqu2  # 将嵌套列表中的空列表删去
        print(row_tiqu2)
        while list2.__contains__([]):
            list2.remove([])  # 通过len(list(2)):32-35，确定if的范围

        for i in range(c_min, c_max + 1):
            if i == len(list2):
                zz_list2 = []
                for x in range(24000):
                    p = []
                    for k in range(len(list2)):
                        z = random.sample(list2[k], 1)
                        for y in z:
                            p.append(y)
                    b_list2 = []
                    cc_list2 = []
                    for i in range(len(p)):
                        if p[i][0] in cc_list2 or p[i][1] in cc_list2:
                            continue
                        else:
                            b_list2.append(p[i])
                        cc_list2.append(p[i][0])
                        cc_list2.append(p[i][1])
                    if len(b_list2) == (row_1 - 2) / 2:
                        zz_list2.append(b_list2)

                zj = nested_list_to_tuple_recur(zz_list2)
                jw = set(zj)

                for i2 in jw:
                    i2 = set(i2)
                    c_list2 = []
                    for n in range(long_carbon_loop):
                        if carbon_loop[2 * n].issubset(i2) or carbon_loop[2 * n + 1].issubset(i2):
                            c_list2.append(n + 1)

                    print(f"Location of free radical combination:{p1}")
                    print(i2)
                    print(f"The {c_list2} ring is a benzene ring structure")

                    c_list2 = sorted(c_list2)

                    s2 = set()
                    for h in range(7):
                        random.shuffle(c_list2)
                        for i in c_list2:  # 这里非常奇怪，为什么输出不全，所以我们使用random打乱后输出
                            zh_list = []
                            zh_list.append(i)
                            for j in c_list2:
                                flag = 0
                                for x in zh_list:
                                    if j in carbon_reloop[x - 1]:
                                        flag = flag + 1
                                if flag == 0:
                                    zh_list.append(j)
                            zh_list = sorted(zh_list)
                            zh_list = tuple(zh_list)
                            s2.add(zh_list)

                    print("Combinations of benzene ring structures that conform to the rules may include:")
                    print(s2)

                    long = 0
                    for l in s2:
                        if len(l) > long:
                            long = len(l)
                        else:
                            continue
                    print(f"The maximum Clar number for this structure is {long}")
                    benhuan_count.append(long)
                    count_1 = count_1 + len(s2)
                    count_zong = count_zong + len(s2)
                count_list.append(count_1)

                print(f"Radical group {p1} has a total number of combinations {count_1}")
                print("\n")

    count_percent = []
    for i in count_list:
        percent = i / count_zong
        count_percent.append(percent)

    print(f"The total number of combinations is {count_zong}")
    result = list(combinations(two_carbon, 2))

    print(f"All of the free radical groups are:")
    print(result)
    print(f"The combination number of the corresponding free radical group is:")
    print(count_list)
    print("The percentage of the combination of the corresponding free radical group is:")
    print(count_percent)

    total_atom = row_1
    atom = [*range(total_atom)]
    atom_count = []

    for element in atom:
        single_count = []
        for i in range(len(result)):
            if element in result[i]:
                single_count.append(count_list[i])
        atom_count.append(sum(single_count))

    '''
    radical probability at atom position
    '''
    total_radical = sum(atom_count)
    prob = [n / total_radical for n in atom_count]
    prob = [n / total_radical for n in atom_count]
    prob = [i * 100 for i in prob]
    b = []
    for j in prob:
        bb = "%.2f%%" % j
        b.append(bb)
    print(f"The percentage of free radicals at each carbon site is the probability of 0~{row_1-1}")
    print(b)
    print(f"The maximum Clar number in all combinations is {max(benhuan_count)}")
    info = {'Carbon site': ["carbon" + str(i) for i in range(row_1)],
            'probability of free radical': b}
    print(tabulate(info, headers='keys', tablefmt='fancy_grid'))
    with open('C:\\tool.box-result\\open-shell-carbon-approximate-site-probability.txt', 'w+') as file0:
        print(tabulate(info, headers='keys', tablefmt='fancy_grid'), file=file0)
    dlg.show_message(f"The total number of combinations is:{count_zong}\n"
                     f"The percentage of free radicals at each carbon site is the probability of 0~{row_1 - 1}:\n{b}\n"
                     f"The maximum Clar number in all combinations is {max(benhuan_count)}")

def full_carbon_random_sample(path,carbon_loop,carbon_reloop):
    sys.stdout = Logger('C:\\tool.box-result\\close-shell-approximate-result.txt')
    long_carbon_loop = int(len(carbon_loop) / 2)
    m1, row_1, col_1 = CarbonMatrix(path)  # 读入碳位点数据
    print(m1, row_1, col_1)
    m1 = m1.astype(np.uint8)  # 将其格式转化为整数
    m2 = np.triu(m1, 1)  # 由于此矩阵为对称矩阵，所以我们只用截取上半部分的进行后续操作
    a_list = []  # 创建空列表储存碳键位为1
    count_benhuan = []
    # 读入矩阵为1处的坐标位置
    for i in range(row_1):
        for j in range(col_1):
            if m2[i, j] == 1:
                a_list.append([i, j])
                sorted(a_list)
    print(a_list)
    count_zong=0
    # 按行封装碳键数
    row_tiqu2 = [[] for x in range(row_1)]
    for i in range(row_1):
        for j in range(len(a_list)):
            if a_list[j][0] == i:
                row_tiqu2[i].append(a_list[j])

    list2 = row_tiqu2  # 将嵌套列表中的空列表删去
    while list2.__contains__([]):
        list2.remove([])  # 通过len(list(2)):32-35，确定if的范围
    print(list2)

    zz_list1 = []
    for x in range(5000000):
        p = []
        for k in range(len(list2)):
            z = random.sample(list2[k], 1)
            for y in z:
                p.append(y)
        b_list2 = []
        cc_list2 = []
        for i in range(len(p)):
            if p[i][0] in cc_list2 or p[i][1] in cc_list2:
                continue
            else:
                b_list2.append(p[i])
            cc_list2.append(p[i][0])
            cc_list2.append(p[i][1])
        if len(b_list2) == row_1 / 2:
            zz_list1.append(b_list2)

    zj = nested_list_to_tuple_recur(zz_list1)
    jw = set(zj)
    print('Close-shell structure benzene ring location:\n')

    for i2 in jw:
        i2 = set(i2)
        c_list2 = []
        for n in range(long_carbon_loop):
            if carbon_loop[2 * n].issubset(i2) or carbon_loop[2 * n + 1].issubset(i2):
                c_list2.append(n + 1)
        i2 = list(i2)
        i2 = sorted(i2)
        print(i2)
        print(f"The {c_list2} ring is a benzene ring structure")
        c_list2 = sorted(c_list2)

        s2 = set()
        for h in range(7):
            random.shuffle(c_list2)
            for i in c_list2:  # 这里非常奇怪，为什么输出不全，所以我们使用random打乱后输出
                zh_list = []
                zh_list.append(i)
                for j in c_list2:
                    flag = 0
                    for x in zh_list:
                        if j in carbon_reloop[x - 1]:
                            flag = flag + 1
                    if flag == 0:
                        zh_list.append(j)
                zh_list = sorted(zh_list)
                zh_list = tuple(zh_list)
                s2.add(zh_list)

        print(s2)
        s2 = list(s2)
        count_zong=count_zong+len(s2)
        long = 0
        for l in s2:
            if len(l) > long:
                long = len(l)
            else:
                continue
        print(f"The maximum Clar number for this structure is {long}")
        count_benhuan.append(long)
    print(count_benhuan)
    print(f"The structure is close-shell structure and the maximum Clar number for this structure is {max(count_benhuan)}")
    dlg.show_message(f"The total number of combinations is:{count_zong}\n"
                     f"The maximum Clar number for this structure is {max(count_benhuan)}")

def Carbon_reloop(path):
    data = xlrd.open_workbook(path)  # 打开excel表所在路径
    sheet = data.sheet_by_name('Sheet1')  # 读取数据，以excel表名来打开
    d = []
    for r in range(sheet.nrows):  # 将表中数据按行逐步添加到列表中，最后转换为list结构
        data1 = []
        for c in range(sheet.ncols):
            data1.append(sheet.cell_value(r, c))
        d.append(list(data1))
    for i in range(len(d)):
        while "" in d[i]:  # 判断是否有空值在列表中
            d[i].remove("")  # 如果有就直接通过remove删除
        d[i] = list(map(int, d[i]))
    return d

def Carbon_loop(path):
    data = xlrd.open_workbook(path)  # 打开excel表所在路径
    sheet = data.sheet_by_name('Sheet1')  # 读取数据，以excel表名来打开
    d = []
    for r in range(sheet.nrows):  # 将表中数据按行逐步添加到列表中，最后转换为list结构
        data1 = []
        for c in range(sheet.ncols):
            data1.append(sheet.cell_value(r, c))
        d.append(list(data1))
    for i in range(len(d)):
        d[i] = list(map(int, d[i]))
    carbon_loop=[]
    for i in range(sheet.nrows):
        sr1=set()
        sr2=set()
        for j in range(3):
            sr1.add((d[i][2*j],d[i][2*j+1]))
            sr2.add((d[i][2*j+6],d[i][2*j+7]))
        carbon_loop.append(sr1)
        carbon_loop.append(sr2)
    return carbon_loop


#文件名读取
filename1 = dlg.get_open_file_name("Please select the path where the carbon matrix is located",dlg.FileFilter.AllFiles)
if filename1 == '':
    dlg.show_message(f"Unselected file")
    exit(-1)

filename2 = dlg.get_open_file_name("Please select the carbon_loop file",dlg.FileFilter.AllFiles)
if filename1 == '':
    dlg.show_message(f"Unselected file")
    exit(-1)

filename3 = dlg.get_open_file_name("Please select the carbon_reloop file",dlg.FileFilter.AllFiles)
if filename1 == '':
    dlg.show_message(f"Unselected file")
    exit(-1)

#图片读取
tmp1 = open('one.png', 'wb')  # 创建临时的文件
tmp1.write(base64.b64decode(surface_graph))  ##把这个one图片解码出来，写入文件中去。
tmp1.close()

tmp2 = open('open-shell.png', 'wb')  # 创建临时的文件
tmp2.write(base64.b64decode(open_shell))  ##把这个one图片解码出来，写入文件中去。
tmp2.close()

tmp3 = open('close-shell.png', 'wb')  # 创建临时的文件
tmp3.write(base64.b64decode(close_shell))  ##把这个one图片解码出来，写入文件中去。
tmp3.close()

#文件读取
carbon_loop=Carbon_loop(filename2)
carbon_reloop=Carbon_reloop(filename3)
print(carbon_reloop)
print(carbon_loop)
#gui界面设计
class Page_1:  # 这是第一个页面,用于选择模式
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Please Select Your Mode")
        global photo
        photo = tk.PhotoImage(file="one.png")
        tk.Label(self.root, image=photo, relief='raised').grid(row=1, column=0, rowspan=2,columnspan=4, padx=5, pady=5)
        tk.Label(self.root, text='Welcome to Clar Combination', font=('微软雅黑', 22)).grid(row=0,column=0,columnspan=4)
        self.button1 = tk.Button(self.root, text='Close Shell', command=self.change_page2, width=12)# 字体，字号，按钮框还得调一下
        self.button2 = tk.Button(self.root, text='Open Shell', command=self.change_page3, width=12)# 字体，字号，按钮框还得调一下
        self.button1.grid(row=3, column=0)
        self.button2.grid(row=3, column=3)
        self.root.mainloop()

    def change_page2(self):
        # pass  # 不知道怎么写，先占位
        self.root.destroy()
        Page_2()

    def change_page3(self):
        # pass  # 不知道怎么写，先占位
        self.root.destroy()
        Page_3()

class Page_2:  #这是第二个页面，用于选择精度
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Close Shell")
        #self.window.config(bg="#0F375A")
        global photo
        photo = tk.PhotoImage(file="close-shell.png")
        tk.Label(self.root, image=photo, relief='raised').grid(row=1, column=1, rowspan=3, columnspan=3, padx=5, pady=5)
        tk.Label(self.root, text='Please select mode', font=('微软雅黑', 22)).grid(row=0, column=0, columnspan=5)
        self.button1 = tk.Button(self.root, text='exact', command=lambda: full_carbon_accurate(filename1,carbon_loop,carbon_reloop), width=12,height=2)  # 字体，字号，按钮框还得调一下
        self.button2 = tk.Button(self.root, text='approximate', command=lambda: full_carbon_random_sample(filename1,carbon_loop,carbon_reloop), width=12,height=2)  # 字体，字号，按钮框还得调一下
        self.button3 = tk.Button(self.root, text='back', command=self.back, width=12,height=2)  # 字体，字号，按钮框还得调一下
        self.button1.grid(row=4,column=0)  # 位置，放置按钮
        self.button2.grid(row=4,column=2)  # 位置，放置按钮
        self.button3.grid(row=4,column=4)  # 位置，放置按钮
        self.root.mainloop()

    def back(self):
        # pass  # 不知道怎么写，先占位
        self.root.destroy()
        Page_1()


class Page_3:  # 这是第三个页面，用于选择精度
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Open Shell")
        #self.window.config(bg="#0F375A")
        global photo
        photo = tk.PhotoImage(file="open-shell.png")
        tk.Label(self.root, image=photo, relief='raised').grid(row=1, column=1, rowspan=3, columnspan=3, padx=5, pady=5)
        tk.Label(self.root, text='Please select mode', font=('微软雅黑', 22)).grid(row=0, column=0, columnspan=5)
        self.button1 = tk.Button(self.root, text='exact', command=lambda: radical_carbon_accurate(filename1,carbon_loop,carbon_reloop), width=12,height=2)  # 字体，字号，按钮框还得调一下
        self.button2 = tk.Button(self.root, text='approximate', command=lambda: radical_carbon_random_sample(filename1,carbon_loop,carbon_reloop), width=12,height=2)  # 字体，字号，按钮框还得调一下
        self.button3 = tk.Button(self.root, text='back', command=self.back, width=12,height=2)  # 字体，字号，按钮框还得调一下
        self.button1.grid(row=4, column=0)  # 位置，放置按钮
        self.button2.grid(row=4, column=2)  # 位置，放置按钮
        self.button3.grid(row=4, column=4)  # 位置，放置按钮
        self.root.mainloop()

    def back(self):
        # pass  # 不知道怎么写，先占位
        self.root.destroy()
        Page_1()

file = "C:\\tool.box-result"
mkdir(file)  # 调用函数，创建文件夹
p1 = Page_1()  # 这两个页单，可单独运行
