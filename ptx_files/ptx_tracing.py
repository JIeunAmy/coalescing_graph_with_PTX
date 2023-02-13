from email import parser
import os, sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns
import pandas as pd
from tqdm import tqdm

import argparse

param_dict = dict()

'''
tidx_list = list() #list(range(0,512))
tidy_list = list()
ctaidy_list = list()
ctaidx_list = list()
'''
global ctaid_map
ctaid_map = list()

def ADD(a,b):
    return a+ b
def SUB(a,b):
    return a-b
def MADLO(a,b):
    return (( a )&0x0000000011111111) + b #(( a )&0x0000000011111111) + b
def MUL(a,b):
    return a * b
def SHL(a,b):
    return int(a) << int(b)
def OR(a,b):
    return a | b
def AND(a,b):
    return a & b
def DIV(a,b):
    return a / b
def SELP(a,b,c):
    if c==1:
        return a
    else:
        return b

def OPERATE(a,b,func):
    if type(a) == str:
        a_list = param_dict[a]
    else:
        a_list = a

    if type(b) == str:
        b_list = param_dict[b]
    else:
        b_list = b
        
    tmp_list = list()
    ctaidy_tmp_dict = list()
    for i in range(ctaidy*ctaidx):
        ctaidy_tmp_dict.append(list())
        
    if type(a) == str and a.startswith("%ctaid"):
        a_len = len(a_list)
        for a_ in a_list: #found the error!!!!!!!!!!!!!!!!!
            for b_ in b_list:
                tmp_list.append(func(a_,b_))
            if a == "%ctaid.x":
                for b_ in param_dict["%ctaid.y"]:
                    ctaidy_tmp_dict[a_+(b_*a_len)]= tmp_list
            else:
                for b_ in param_dict["%ctaid.x"]:
                    ctaidy_tmp_dict[a_+(b_*a_len)] = tmp_list
            tmp_list = list()
        return ctaidy_tmp_dict
    elif type(b) == str and  b.startswith("%ctaid"):
        b_len = len(b_list)
        for b_ in b_list: 
            for a_ in a_list:
                tmp_list.append(func(a_, b_))
            if b == "%ctaid.x":
                for a_ in param_dict["%ctaid.y"]:
                    ctaidy_tmp_dict[(a_*b_len)+b_]= tmp_list
            else:
                for a_ in param_dict["%ctaid.x"]:
                    ctaidy_tmp_dict[(a_*b_len)+b_] = tmp_list
            tmp_list = list()
        return ctaidy_tmp_dict
    else:
        if len(a_list)!=0  and type(a_list[0]) == list:
            for idx_, a_ in enumerate(a_list):
                for a_int in a_:
                    for b_ in b_list:
                        if type(b_) == list:
                            for b_int in b_:
                                tmp_list.append(func(a_int, b_int))
                        else:
                            tmp_list.append(func(a_int, b_))
                ctaidy_tmp_dict[idx_] = tmp_list
                tmp_list = list()
            return ctaidy_tmp_dict
        elif len(b_list)!=0 and type(b_list[0]) == list:
            for idx_, b_ in enumerate(b_list):
                #print(b_)
                for b_int in b_:
                    for a_ in a_list:
                        tmp_list.append(func(a_, b_int))
                ctaidy_tmp_dict[idx_] = tmp_list
                tmp_list = list()
            #print("DDD")
            #print(ctaidy_tmp_dict)
            return ctaidy_tmp_dict
        else:
            for a_ in a_list:
                for b_ in b_list:
                    tmp_list.append(func(a_,b_))
            return tmp_list
    return tmp_list

def tracing(tree,idx):
    opcode = tree[idx]["opcode"]
    child_num = tree[idx]["child_num"]
    child_list = list()
    if child_num == 0:
        reg_ = tree[idx]["reg"]
        return reg_ # str return
    while(child_num>0):
        for id_, node_ in enumerate(tree):
            if(node_["parent"]==idx):
                child_list.append(id_)
                child_num -= 1
    child_num = tree[idx]["child_num"]
    if child_num==2:
        if opcode.startswith("add"):
            res_ = OPERATE(tracing(tree,child_list[0]),tracing(tree, child_list[1]),ADD)
            return res_
        elif opcode.startswith("sub"):
            res_ = OPERATE(tracing(tree,child_list[0]),tracing(tree, child_list[1]),SUB)
            return res_
        elif opcode.startswith("shl"):
            res_ = OPERATE(tracing(tree, child_list[0]), tracing(tree, child_list[1]),SHL)
            return res_
        elif opcode.startswith("mul"):
            res_ = OPERATE(tracing(tree, child_list[0]),tracing(tree, child_list[1]),MUL)
            return res_
        elif opcode.startswith("or"):
            res_ = OPERATE(tracing(tree, child_list[0]),tracing(tree, child_list[1]),OR)
            return res_
    elif child_num==3:
        if opcode.startswith("mad"):
            res_ = OPERATE(OPERATE(tracing(tree, child_list[0]), tracing(tree, child_list[1]), MUL), tracing(tree, child_list[2]), MADLO)
            return res_
        elif opcode.startswith("selp"):
            res_ = OPERATE(tracing(tree, child_list[0]),tracing(tree, child_list[1]),tracing(tree, child_list[2]),SELP)
            return res_
    else:
        #print(tree[child_list[0]]["reg"])
        return tracing(tree, child_list[0])
        
def make_ctaid_map(formular):
    for i in range(ctaidy*ctaidx):
        for j in range(ctaidy*ctaidx):
            ctaid_map[i][j]=0

    if type(formular[0]) == list:
        print(f"len_0: {len(formular[0])}, len_1: {len(formular[1])}")
        for i in tqdm(range(ctaidy*ctaidx)):
            for j in range(i+1, ctaidy*ctaidx):
                cnt = 0
                for f_i in formular[i]:
                    for f_j in formular[j]:
                        if f_i==f_j:
                            cnt+=1
                            break
                ctaid_map[j][i] = cnt
    else:
        print("*****************no ctaid*****************************")



def file_open(file_name):
    with open(file_name, "r") as json_file:
        syntax_tree = json.load(json_file)
        for kernel_name, line in syntax_tree.items():
            print(f"kernel name: {kernel_name}")

            kernel_map=list()
            for i in range(ctaidy*ctaidx):
                kernel_map.append(list())
                for j in range(ctaidy*ctaidx):
                    kernel_map[i].append(0)

            for id_,( key, tree) in tqdm(enumerate(line.items())):
                #print(f"------------------{id_}----------------")
                formular = tracing(tree,0)
                make_ctaid_map(formular)
                
                for i in range(ctaidy*ctaidx):
                    for j in range(i+1, ctaidy*ctaidx):
                        kernel_map[j][i] += ctaid_map[j][i]
                        #print(ctaid_map[i][j], end=" ")
                    #print()
                #print("--------------------------------------------------------------------------")
                #for f_ in formular:
                #    print(f_)
                #exit(1)
                #print(f"{id_}: {formular}")
            np_kernel_map = np.array(kernel_map)
            dp_kernel_map = pd.DataFrame(np_kernel_map)
            sns.heatmap(dp_kernel_map,cmap="OrRd")
            if os.path.isdir(f"img/{app_name}") == False:
                os.makedirs(f"img/{app_name}")
            
            plt.savefig(f"img/{app_name}/{kernel_name}_{ctaidx}-{ctaidy}.png")
            plt.clf()
            #exit(1)
            

def backprop_init():
    global tidx
    global tidy
    global ctaidx
    global ctaidy
    
    tidx = 16
    tidy = 16
    ctaidx = 1
    ctaidy = 64 #512

    tidx_list = list(range(0,tidx)) #list(range(0,512))
    tidy_list = list(range(0,tidy))
    ctaidy_list = list(range(0,ctaidy))
    ctaidx_list = list(range(0,ctaidx))


    param_dict["%ctaid.y"] = ctaidy_list #1024/16
    param_dict["%ctaid.x"] = ctaidx_list #1
    param_dict["%tid.y"] = tidy_list #list(range(0,16))
    param_dict["%tid.x"] = tidx_list #list(range(0,16))
    param_dict["_Z22bpnn_layerforward_CUDAPfS_S_S_ii_param_0"] = [0]
    param_dict["_Z22bpnn_layerforward_CUDAPfS_S_S_ii_param_2"] = [64*16+64]
    param_dict["_Z22bpnn_layerforward_CUDAPfS_S_S_ii_param_5"] = [64*16*2]
    param_dict["_Z24bpnn_adjust_weights_cudaPfiS_iS_S__param_0"] = [64*16*3]
    param_dict["_Z24bpnn_adjust_weights_cudaPfiS_iS_S__param_2"] = [64*16*4]
    param_dict["_Z24bpnn_adjust_weights_cudaPfiS_iS_S__param_5"] = [64*16*5]
    param_dict["_Z24bpnn_adjust_weights_cudaPfiS_iS_S__param_4"] = [64*16*6]
    param_dict["_Z24bpnn_adjust_weights_cudaPfiS_iS_S__param_1"] = [64*16*7]
    param_dict["4"] = [4]
    param_dict["16"] = [16]
    param_dict["1"] = [1]
    for i in range(ctaidy*ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy*ctaidx):
            ctaid_map[i].append(0)
    file_open("syntax_tree/rodinia/rodinia_backprop.json")
    #res = OPERATE("%ctaid.y",OPERATE("_Z22bpnn_layerforward_CUDAPfS_S_S_ii_param_5","%tid.y",ADD),SHL)


def MM2_init():
    global tidx
    global tidy
    global ctaidx
    global ctaidy
    
    tidx = 32 #32
    tidy = 8 #8
    ctaidx = 32 #8 #32
    ctaidy = 128 #32 #128
    tidx_list = list(range(0,tidx)) #list(range(0,512))
    tidy_list = list(range(0,tidy))
    ctaidx_list = list(range(0,ctaidx))
    ctaidy_list = list(range(0,ctaidy))
    param_dict["%ctaid.x"] = ctaidx_list #1024/16
    param_dict["%ctaid.y"] = ctaidy_list #1024/16
    param_dict["%tid.x"] = tidx_list #list(range(0,16))
    param_dict["%tid.y"] = tidy_list #list(range(0,16))
    param_dict["%ntid.x"] = [tidx]
    param_dict["%ntid.y"] = [tidy]

    param_dict["_Z11mm2_kernel1iiiiffPfS_S__param_7"] = [0]
    param_dict["_Z11mm2_kernel1iiiiffPfS_S__param_8"] = [4194304]
    param_dict["_Z11mm2_kernel2iiiiffPfS_S__param_6"] =[4194304*2]
    param_dict["_Z11mm2_kernel2iiiiffPfS_S__param_7"] = [4194304*3]
    param_dict["_Z11mm2_kernel2iiiiffPfS_S__param_8"] = [4194304*4]
    param_dict['10'] = [10]
    param_dict['4'] = [4]
    param_dict['2048'] = [2048]
    param_dict['2'] = [2]
    param_dict['0'] = [0]
    for i in range(ctaidy*ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy*ctaidx):
            ctaid_map[i].append(0)
    file_open("syntax_tree/polybench/polybench_2MM.json")

def gemm_init():
    global tidx
    global tidy
    global ctaidx
    global ctaidy
    
    tidx = 32 #32
    tidy = 8 #8
    ctaidx = 2 #2
    ctaidy = 8 #8
    tidx_list = list(range(0,tidx)) #list(range(0,512))
    tidy_list = list(range(0,tidy))
    ctaidx_list = list(range(0,ctaidx))
    ctaidy_list = list(range(0,ctaidy))
    param_dict["%ctaid.x"] = ctaidx_list #1024/16
    param_dict["%ctaid.y"] = ctaidy_list #1024/16
    param_dict["%tid.x"] = tidx_list #list(range(0,16))
    param_dict["%tid.y"] = tidy_list #list(range(0,16))
    param_dict["%ntid.x"] = [tidx]
    param_dict["%ntid.y"] = [tidy]

    param_dict["_Z11gemm_kerneliiiffPfS_S__param_7"] = [4194304*4]
    param_dict["_Z11gemm_kerneliiiffPfS_S__param_5"] = [4194304]
    param_dict["_Z11gemm_kerneliiiffPfS_S__param_6"] =[4194304*2]

    param_dict['0'] = [0]
    param_dict['4'] = [4]
    param_dict['1024'] = [1024]
    param_dict['9'] = [9]
    param_dict['2'] = [2]
    for i in range(ctaidy*ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy*ctaidx):
            ctaid_map[i].append(0)
    file_open("syntax_tree/polybench/polybench_GEMM.json")


    

def bfs_init():

    global tidx
    global tidy
    global ctaidx
    global ctaidy


    tidx = 256
    tidy = 1
    ctaidx = 256
    ctaidy = 1

    tidx_list = list(range(0,tidx)) #list(range(0,512))
    tidy_list = list(range(0,tidy))
    ctaidx_list = list(range(0,ctaidx))
    ctaidy_list = list(range(0,ctaidy))


    param_dict["%ctaid.x"] = ctaidx_list #1024/16
    param_dict["%ctaid.y"] = ctaidy_list
    param_dict["%tid.y"] = tidy_list #list(range(0,16))
    param_dict["%tid.x"] = tidx_list #list(range(0,16))
    param_dict["%ntid.x"] = [ctaidx]
    param_dict["%ntid.y"] = [ctaidy]
 
    param_dict["_Z6KernelP4NodePiPbS2_S2_S1_i_param_2"] = [0]
    param_dict["_Z6KernelP4NodePiPbS2_S2_S1_i_param_0"] = [64*16]
    param_dict["_Z6KernelP4NodePiPbS2_S2_S1_i_param_1"] = [64*16*2]
    param_dict["_Z6KernelP4NodePiPbS2_S2_S1_i_param_4"] = [64*16*3]
    param_dict["_Z6KernelP4NodePiPbS2_S2_S1_i_param_5"] = [64*16*4]
    param_dict["_Z7Kernel2PbS_S_S_i_param_1"] = [64*16*5]
    param_dict["4"] = [4]
    param_dict["16"] = [16]
    param_dict["1"] = [1]
    param_dict["9"] = [9]
    param_dict["8"] = [8]
    for i in range(ctaidy*ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy*ctaidx):
            ctaid_map[i].append(0)
    #file_open("syntax_tree/rodinia/rodinia_bfs.json")
    file_open("syntax_tree/rodinia/rodinia_bfs_256.json")

def hotspot_init():
    
    global tidx
    global tidy
    global ctaidx
    global ctaidy


    tidx = 16
    tidy = 16
    ctaidx = 4
    ctaidy = 8

    tidx_list = list(range(0,tidx)) #list(range(0,512))
    tidy_list = list(range(0,tidy))
    ctaidx_list = list(range(0,ctaidx))
    ctaidy_list = list(range(0,ctaidy))


    param_dict["%ctaid.x"] = ctaidx_list #1024/16
    param_dict["%ctaid.y"] = ctaidy_list
    param_dict["%tid.y"] = tidy_list #list(range(0,16))
    param_dict["%tid.x"] = tidx_list #list(range(0,16))
    param_dict["%ntid.x"] = [ctaidx]
    param_dict["%ntid.y"] = [ctaidy]
 
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_2"] = [64*16*5]
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_0"] = [64*16]
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_7"] = [64*16*2]
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_6"] = [64*16*3]
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_4"] = [64*16*4]
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_1"] = [64*16*6]

    param_dict["4"] = [4]
    param_dict["16"] = [16]
    param_dict["1"] = [1]
    param_dict["9"] = [9]
    param_dict["8"] = [8]
    for i in range(ctaidy*ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy*ctaidx):
            ctaid_map[i].append(0)
    file_open("syntax_tree/rodinia/rodinia_hotspot.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="select application to trace")
    parser.add_argument('-ap')
    args = parser.parse_args()
    switcher = { 
        "backprop": backprop_init, 
        "MM2": MM2_init, 
        "bfs": bfs_init,
        "hotspot": hotspot_init,
        "gemm": gemm_init
        }
    global app_name
    app_name = args.ap
    switcher.get(app_name)()
    '''
    if os.path.isdir(f"{app_name}") == False:
        os.makedirs(f"{app_name}")
        print("DDDDDDDDDDD")
    '''
    #switcher(app_name)


