import os, sys
import json
import math
param_dict = dict()
tidx = 1
tidy = 1
ctaidy = 16

tidx_list = list(range(0,16))
tidy_list = list(range(0,16))
ctaidy_list = list(range(0,ctaidy))
ctaid_map = list()

def ADD(a,b):
    return f"({a} + {b})"
def ABS(a):
    if a<0:
        a*=-1
    return a
def SUB(a,b):
    return f"({a} - {b})"
def MADLO(a,b,c):
    return f"(( {a} *  {b} )&0x0000000011111111) + {c}"
def MAD(a,b,c):
    return f"(( {a} *  {b} )&0x0000000011111111) + {c}"
def MUL(a,b):
    return f"({a} * {b})"
def LD(a):
    return f"{a}"
def ST(a):
    return f"{a}"
def CVTA(a):
    return f"{a}"
def CVT(a):
    return f"{a}"

def MOV(a):
    return f"{a}"
def NOT(a):
    return f"~{a}"
def SHR(a,b):
    return f"({a} >> {b})"

def SHL(a,b):
    #print(f"{a} {b}")
    return f"({a} << {b})"
def OR(a,b):
    return f"({a} | {b})"

def AND(a,b):
    return f"({a} & {b})"
def DIV(a,b):
    return f"({a} / {b})"
def SELP(a,b,c):
    if c==1:
        return f"{a}"
    else:
        return f"{b}"
    #return f"({a} if {c==1} else {b})"
def SETP(a,b,c):
    return SELP(a,b,c)
def SETP_GE(a,b,c):
    if c>=1:
        return f"{a}"
    else:
        return f"{b}"
def SETP_LT(a,b,c):
    if c<1:
        return f"{a}"
    else:
        return f"{b}"
def SETP_NE(a,b,c):
    if c!=1:
        return f"{a}"
    else:
        return f"{b}"
def SETP_GT(a,b,c):
    if c >1:
        return f"{a}"
    return f"{b}"
def SETP_EQ(a,b,c):
    if c==1:
        return f"{a}"
    return f"{b}"
def BFE(a,b,c):
    return f"{a}"
def PRMT(a,b,c):
    return f"{a}"
def SQRT(a):
    return f"{pow(a)}"
def MIN(a,b):
    if a>b:
        return f"{b}"
    return f"{a}"
def MAX(a,b):
    if a<b:
        return f"{b}"
    return f"{a}"
def NEG(a):
    return f"{a*-1}"
def CLZ(a):
    cnt =0
    while(True):
        if a%10:
            cnt+=1
        else:
            break
    return f"{cnt}"
def RCP(a):
    return f"(1/{a})"
def REM(a):
    return f"{a}"

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
    for i in range(ctaidy):
        ctaidy_tmp_dict.append(list())
        
    if type(a) == str and a == "%ctaid.y":
        for a_ in a_list:
            for b_ in b_list:
                tmp_list.append(func(a_,b_))
            ctaidy_tmp_dict[a_]= tmp_list
            tmp_list = list()
        return ctaidy_tmp_dict
    elif type(b) == str and b == "%ctaid.y":
        for b_ in b_list: 
            for a_ in a_list:
                tmp_list.append(func(a_, b_))
            ctaidy_tmp_dict[b_] = tmp_list
            tmp_list = list()
        return ctaidy_tmp_dict
    else:
        if type(a_list[0]) == list:
            for idx_, a_ in enumerate(a_list):
                for a_int in a_:
                    for b_ in b_list:
                        tmp_list.append(func(a_int, b_))
                ctaidy_tmp_dict[idx_] = tmp_list
                tmp_list = list()
            return ctaidy_tmp_dict
        elif type(b_list[0]) == list:
            for idx_, b_ in enumerate(b_list):
                for b_int in b_:
                    for a_ in a_list:
                        tmp_list.append(func(a_, b_int))
                ctaidy_tmp_dict[idx_] = tmp_list
                tmp_list = list()
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
            str_ = ADD(str(tracing(tree,child_list[0])),str(tracing(tree,child_list[1])))
            return str_
        elif opcode.startswith("sub"):
            str_ = SUB(str(tracing(tree,child_list[0])),str(tracing(tree,child_list[1])))
            return str_
        elif opcode.startswith("shl"):
            str_ = SHL(str(tracing(tree,child_list[0])),str(tracing(tree,child_list[1])))
            return str_
        elif opcode.startswith("mul"):
            str_ = MUL(str(tracing(tree,child_list[0])),str(tracing(tree,child_list[1])))
            return str_
        elif opcode.startswith("or"):
            str_ = OR(str(tracing(tree,child_list[0])),str(tracing(tree,child_list[1])))
            return str_
        elif opcode.startswith("div"):
            str_ = DIV(str(tracing(tree,child_list[0])),str(tracing(tree,child_list[1])))
            return str_
    elif child_num==3:
        if opcode.startswith("mad"):
            str_ = MADLO(str(tracing(tree,child_list[0])),str(tracing(tree,child_list[1])),str(tracing(tree,child_list[2])))
            return str_
        if opcode.startswith("selp"):
            str_ = SELP(str(tracing(tree,child_list[0])),str(tracing(tree,child_list[1])),str(tracing(tree,child_list[2])))
            return str_
    else:
        return tracing(tree, child_list[0])

def make_ctaid_map(formular):
    
    for i in range(ctaidy):
        ctaid_map.append(list())
        for j in range(ctaidy):
            ctaid_map[i][j]=0
            
    if type(formular[0]) == list:
        for i in range(ctaidy):
            for j in range(i+1, ctaidy):
                cnt = 0
                for f_i in formular[i]:
                    if f_i in formular[j]:
                        cnt+=1
                ctaid_map[i][j] = cnt
                ctaid_map[j][i] = cnt
    else:
        print("*****************no ctaid*****************************")



def file_open(file_name):
    with open(file_name, "r") as json_file:
        syntax_tree = json.load(json_file)
        for kernel_name, line in syntax_tree.items():
            print(f"kernel name: {kernel_name}")
            
            for id_,( key, tree) in enumerate(line.items()):
                formular = tracing(tree,0)
                print(formular)
                
def backprop_init():
    file_open("syntax_tree/rodinia/rodinia_backprop.json")

def srad_v1_init():
    file_open("syntax_tree/rodinia/rodinia_srad_v1.json")

def bfs_init():
    file_open("syntax_tree/rodinia/rodinia_bfs.json")

def hotspot_init():
    file_open("syntax_tree/rodinia/rodinia_hotspot.json")
    
def MM2_init():
    file_open("syntax_tree/polybench/polybench_2MM.json")

def GEMM_init():
    file_open("syntax_tree/polybench/polybench_GEMM.json")
    
def Convolution2D_init():
    file_open("syntax_tree/polybench/polybench_Convolution2D.json")

if __name__ == "__main__":
    #backprop_init()
    #MM2_init()
    #GEMM_init()
    #Convolution2D_init()
    #srad_v1_init()
    #bfs_init()
    hotspot_init()


