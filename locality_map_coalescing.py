import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math
import argparse
import json
from tqdm import tqdm
from math import sqrt
import csv
import re

Block = dict()
thread_dict = dict()
thread_dict = {    
    "%ctaid.y": "ctaid_y",
    "%nctaid.y": "nctaid_y",
    "%ntid.y": "ntid_y",
    "%tid.y": "tid_y",
    "%nctaid.x": "nctaid_x",
    "%ctaid.x": "ctaid_x",
    "%ntid.x": "ntid_x",
    "%tid.x": "tid_x",
}
thread_list = {
    "nctaid_y",    
    "ctaid_y",
    "ntid_y",
    "tid_y",
    "nctaid_x",
    "ctaid_x",
    "ntid_x",
    "tid_x"
}

def selp(a,b,c):
    if(c==1):
        return a
    return b
def clz(a,options):
    len = int(options)
    tmp = ["0"]*len
    a_bin = format(a,'b')
    a_bin = list(a_bin)
    idx_ = -1
    for i in reversed(a_bin):
        tmp[idx_]=i
        idx_ -= 1
    return (tmp.index('1'))

def get_threadID(nctaid_x,nctaid_y,ntid_x,ntid_y,file_name,param_file_name, formular_file_name, app_name):
    param_f = open(param_file_name,"r")
    parameter_dict = json.load(param_f)
    formul_f = open(formular_file_name)
    formul_dict = json.load(formul_f)
    
    #print(parameter_dict)
    #print(f"{nctaid_x}, {nctaid_y}, {ntid_x}, {ntid_y}")
    
    parameter_dict["ntaid_x"] = nctaid_x
    parameter_dict["ntaid_y"] = nctaid_y
    parameter_dict["nctaid_x"] = nctaid_x
    parameter_dict["nctaid_y"] = nctaid_y
    parameter_dict["ntid_x"] = ntid_x
    parameter_dict["ntid_y"] = ntid_y
    parameter_dict["0d3FF0000000000000"] = 0x3FF0000000000000
    parameter_dict["0f00000000"] = 0.0
    parameter_dict["0x42A00000"] = 0x42A00000 
    parameter_dict["0xFFFFFFFFFFFFFFFF"] = 0xFFFFFFFFFFFFFFFF 
    
    for kernel_name, ld_global in formul_dict.items():
        
        print(kernel_name)
        if len(ld_global) == 0:
            continue
        thread_map = list()
        kernel_map=list()
        for i in range(nctaid_y*nctaid_x):
            kernel_map.append(list())
            for j in range(nctaid_y*nctaid_x):
                kernel_map[i].append(0)
        for i in range(ntid_x*ntid_y):
            thread_map.append(list())
            for j in range(ntid_x*ntid_y):
                thread_map[i].append(0)

        if kernel_name not in Block:
            Block[kernel_name] = list()
            thread_dict[kernel_name] = list()
        ## coalescing에 맞게 결과 반영
        
        
        ## coalescing 결과를 이 전에 반영
        coalescing_avg = 0.0
        coalescing_map = [0]*32
        line_cnt = 0
        for ctaid_y in range(nctaid_y):
            parameter_dict["ctaid_y"] = ctaid_y
            for ctaid_x in range(nctaid_x):
                parameter_dict["ctaid_x"] = ctaid_x
                blockId = ctaid_y*nctaid_x + ctaid_x
                Block[kernel_name].append(list())
                thread_dict[kernel_name].append(list())
                start_point = [0]*len(ld_global)
                warp_start_point = [0]*len(ld_global)
                coalescing_cnt = 1
                coalescing_max = 0.0
                for tid_y in range(ntid_y):
                    parameter_dict["tid_y"] = tid_y
                    for tid_x in range(ntid_x):
                        parameter_dict["tid_x"] = tid_x
                        
                        tId = ntid_x*tid_y + tid_x
                        thread_dict[kernel_name][blockId].append(list())
                        global_cnt = 0
                        for ld_reg, formul_list in ld_global.items():
                            formul = ld_global[ld_reg]["final_formular"]
                            #print(formul)
                            res = eval(formul.format(**parameter_dict))
                            if(tId%32 == 0):
                                start_point[global_cnt] = int(res/32)*32#res

                                coalescing_cnt = 1
                                coalescing_map[0] += 1
                                coalescing_avg +=1
                            else:
                                if(res>=start_point[global_cnt] and res < start_point[global_cnt]+128):
                                    coalescing_cnt += 1
                                    offset = int((res-start_point[global_cnt])/4)
                                    # if coalescing_map[offset] == None: coalescing_map[offset] = 1
                                    # else:
                                    coalescing_map[offset] +=1
                                    
                                    coalescing_avg +=1
                            if(tId%32 == 31):
                                line_cnt += 1

                                if coalescing_max <coalescing_cnt: 
                                    coalescing_max = coalescing_cnt
                            thread_dict[kernel_name][blockId][tId].append(res)
                            Block[kernel_name][blockId].append(res)
                            global_cnt += 1
                    
        graph_name = app_name
        if app_name == "backprop":
            if(kernel_name.split("_")[2].startswith("layer") ):
                graph_name+= "_forward"
            else:
                graph_name+="_adjust_weights"
        if app_name == "bfs":
            if(kernel_name.startswith("_Z7Kernel2")):
                graph_name+="_kernel2"
            else:
                graph_name+="_kernel1"
        if app_name == "b+tree":
            if(kernel_name.startswith("findK")):
                graph_name+="_findK"
            else:
                graph_name+="_findRageK"
        # print("DDDDDDDDD")
               
        # no annotation
        np_coalescing_map = np.array(coalescing_map)/line_cnt
        dp_coalescing_map = pd.DataFrame(np_coalescing_map,dtype="float")
        #y = np.array(coalescing_map)/line_cnt
        plt.figure(figsize=(80,5))
        plt.rc('font', size=20) 
        # plt.rcParams["figure.figsize"] = (50,2)
        # plt.xlabel('thread id', fontsize=20)
        # plt.title(f"{graph_name}", fontsize=20)
        # plt.yticks([None])
        # sns.heatmap(dp_coalescing_map.T,cmap="Reds",vmin=0,annot=True, annot_kws={"size": 20})
        hplot = sns.heatmap(dp_coalescing_map.T,cmap="Reds",vmin=0,annot_kws={"size": 20},square=True)
        #hplot = sns.heatmap([y],vmin=0,cmap="Reds",annot=True,annot_kws={"size": 20},square=True)
        sector_expectation = 0
        cnt_tmp = 0
        tmp_ = 0
        for idx, i in enumerate(np_coalescing_map):
            #print("ddddd")
            i = float(int(i*100)/100.0)
            #continue
            if cnt_tmp%8 == 0:
                cnt_tmp = 0
                tmp_ = 0
            if tmp_ == 0 and i!=0:
                sector_expectation+=1
                tmp_ = 1
            
            #print(f"{idx}: {i}")
            #if idx%8 == 7:
            #    print(f"------{sector_expectation}------")
            cnt_tmp += 1
        #exit(1)
        hplot.set(yticklabels=[])
        hplot.set(ylabel="coalescing")
        hplot.set(xlabel="data location")
        hplot.set(title=f"{graph_name}") 
        if(not os.path.isdir(f"result_img/{app_name}")):
            os.makedirs(f"result_img/{app_name}")
        plt.savefig(f"result_img/{app_name}/{graph_name}_{nctaid_x}_{nctaid_y}_{ntid_x}_{ntid_y}coalescing.png")
        plt.clf()
        with open("1216_locality_result.csv","a") as f:
            wr = csv.writer(f)
            #wr.writerow([graph_name,f"{coalescing_avg/line_cnt/32*100}%",sector_expectation, line_cnt, len(ld_global), line_cnt/len(ld_global)])
            wr.writerow([graph_name,f"{coalescing_avg/line_cnt/32*100}%",len(ld_global), sector_expectation, sector_expectation/(coalescing_avg/line_cnt/32*100)])
        

        '''
        #with annotation
        y = np.array(coalescing_map)/line_cnt
        fig, ax = plt.subplots()
        plt.figure(figsize=(45,2.5))
        plt.rc('font', size=20) 
        hplot = sns.heatmap([y],vmin=0,cmap="Reds",annot=True,annot_kws={"size": 20},square=True)
        hplot.set(yticklabels=[])
        hplot.set(ylabel="coalescing")
        hplot.set(xlabel="data location")
        hplot.set(title=f"{graph_name}") 
        plt.savefig(f"new_img/{app_name}/{graph_name}_{nctaid_x}_{nctaid_y}_{ntid_x}_{ntid_y}_annote_coalescing.png")
        plt.clf()
        llll = 0
        for mp_ in coalescing_map:
            llll += mp_
        print(f"{llll}, {coalescing_avg},{line_cnt}, {len(ld_global)}")
        with open("final_locality_result.csv","a") as f:
            wr = csv.writer(f)
            wr.writerow([graph_name,coalescing_avg/line_cnt,line_cnt])
        '''

        continue
        intra_locality = 0
        access_cnt = 0
        cnt_total = 0
        for idx, cnt in enumerate(coalescing_map):
            print(f"{idx}: {cnt}")
            if(cnt != 0 ):
                access_cnt += 1
                intra_locality += line_cnt/cnt
                print(line_cnt/cnt)
            cnt_total += cnt
        print("DDDDDDDDDDDDDDDDDDDDDDDD")
        print(f"{graph_name}, {(32/access_cnt)*intra_locality}")
        print(f"{coalescing_avg/line_cnt}, {(coalescing_avg/line_cnt)/32*100}%",end=", ")
        
        
        tr_sparse = 0
        check_cnt = 0
        for bid_ in tqdm(range(nctaid_x*nctaid_y)):
            for tid_ in range(ntid_x*ntid_y):
                for tjd_ in range(tid_+1, ntid_x*ntid_y):
                    check_cnt += 1
                    co = len([_ for _ in thread_dict[kernel_name][bid_][tid_] if _ in thread_dict[kernel_name][bid_][tjd_] ])
                    thread_map[tjd_][tid_] += co
                    thread_map[tid_][tjd_] += co
                    # for i in range(len(ld_global)):
                    #     if(thread_dict[kernel_name][0][tid_][i] == thread_dict[kernel_name][0][tjd_][i]):
                    if thread_map[tjd_][tid_] != 0:
                        tr_sparse+=1
        tr_sparse = tr_sparse/check_cnt
        
        for tid_ in range(ntid_x*ntid_y):
            for tjd_ in range(tid_+1, ntid_x*ntid_y):
                thread_map[tjd_][tid_] = thread_map[tjd_][tid_]/(nctaid_x*nctaid_y)
                thread_map[tid_][tjd_] = thread_map[tid_][tjd_]/(nctaid_x*nctaid_y)
        # tr_sparse = tr_sparse*2/(ntid_x*ntid_y)
        
        '''
        th_sparse = 0
        max_ = 0
        # print(f"{len(Block[kernel_name][0])}, {len(ld_global)*ntid_x*ntid_y} ")

        for id_ in tqdm(range(nctaid_x*nctaid_y)):
            for jd_ in range(id_+1, nctaid_x*nctaid_y):
                # for i in range(len(ld_global)*ntid_x*ntid_y):
                co = len([_ for _ in Block[kernel_name][id_] if _ in Block[kernel_name][jd_] ])
                    
                kernel_map[id_][jd_] = co
                kernel_map[jd_][id_] = co
                if max_ <kernel_map[jd_][id_]:
                    max_ = kernel_map[jd_][id_]
                if kernel_map[jd_][id_] != 0:
                    th_sparse += 1
        th_sparse = th_sparse*2/(nctaid_x*nctaid_y)
        '''
        intra_dos = 0
        th_cnt =0
        th_access_cnt=0
        for tx in range(ntid_x*ntid_y):
            for ty in range(tx+1,ntid_x*ntid_y):
                th_access_cnt+=1
                th_cnt += thread_map[tx][ty]
        th_inst_cnt = len(formul_dict[kernel_name])
        if th_access_cnt == 0 :
            intra_dos = 0
        else:
            intra_dos = th_cnt/(len(ld_global)*th_access_cnt)
        '''
        inter_dos = 0
        tb_cnt = 0
        tb_access_cnt=0
        for tbx in range(nctaid_x*nctaid_y):
            for tby in range(tbx+1,nctaid_x*nctaid_y):
                if kernel_map[tbx][tby]!=0:
                    tb_cnt += kernel_map[tbx][tby]
                tb_access_cnt+=1
        tb_inst_cnt = len(formul_dict[kernel_name])
        inter_dos = tb_cnt/(tb_access_cnt*len(ld_global)*ntid_x*ntid_y)

        for id_ in range(nctaid_x*nctaid_y):
            kernel_map[id_][id_] = None
            for jd_ in range(id_+1, nctaid_x*nctaid_y):
                kernel_map[id_][jd_] = None
        np_kernel_map = np.array(kernel_map)
        #np_kernel_map = np.array(kernel_map)
        dp_kernel_map = pd.DataFrame(np_kernel_map,dtype="float")
        sns.heatmap(dp_kernel_map,cmap="Reds",vmin=0)
        if os.path.isdir(f"new_img/{app_name}") == False:
            os.makedirs(f"new_img/{app_name}")

        plt.savefig(f"new_img/{app_name}/{graph_name}_{nctaid_x}_{nctaid_y}_{ntid_x}_{ntid_y}.png")
        plt.clf()
        '''
        print(f"name: {graph_name}")
        for id_ in range(ntid_x*ntid_y):
            thread_map[id_][id_] =None
            for jd_ in range(id_+1, ntid_x*ntid_y):
                thread_map[id_][jd_] = None
        np_thread_map = np.array(thread_map)
        dp_thread_map = pd.DataFrame(np_thread_map,dtype="float")
        tb_plot = sns.heatmap(dp_thread_map,cmap="Reds",vmin=0)
        tb_plot.set(title=f"{graph_name}") 
        plt.savefig(f"new_img/{app_name}/{graph_name}_threadmap_{nctaid_x}_{nctaid_y}_{ntid_x}_{ntid_y}.png")
        plt.clf()
        # print("***************************")
        # print(f"ctaid x: {nctaid_x}, ctaid y: {nctaid_y}, tid x: {ntid_x}, tid y:{ntid_y}")
        # print(f"max element: {max_}")
        # print("inter degree of sharing: ",inter_dos)
        # print("intra degree of sharing locality: ",intra_dos)
        # print("inter sparsity: ",th_sparse)
        # print("intra sparsity: ",tr_sparse)
        # print(f"{intra_dos},{inter_dos}")#,th_sparse,tr_sparse)
        # print("***************************")
        with open("final_locality_result.csv","a") as f:
            wr = csv.writer(f)
            wr.writerow([file_name,graph_name,intra_locality,tr_sparse,coalescing_avg/line_cnt/len(ld_global),nctaid_x,nctaid_y,ntid_x,ntid_y, line_cnt])

        '''
        for id_ in range(nctaid_x*nctaid_y):
            for jd_ in range(id_+1, nctaid_x*nctaid_y):
                print(kernel_map[id_][jd_], end=" ")
            print()'''
        #intra thread block
    param_f.close()
    formul_f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",help="file_name",default="GEMM_2_8_32_8")
    parser.add_argument("-d",help="directory",default="syntax_tree/")
    parser.add_argument("-t",help="thread change",default="")
    parser.add_argument("-s",help="shared mode",default=0,type=int)
    args = parser.parse_args()
    file_name = args.f
    dir_name = args.d
    change_th = args.t
   
    
    

    param_file_name = f"{file_name}_param.json"
    formular_file_name = f"{file_name}_formular.json"
    param_file_name = os.path.join(dir_name,file_name,param_file_name)
    formular_file_name = os.path.join(dir_name,file_name,formular_file_name)
    file_name = file_name.split(".")[0]
    file_info = file_name.split("_")
    tid_y =  int(file_info[-1])
    tid_x =  int(file_info[-2])
    ctaid_y = int(file_info[-3])
    ctaid_x = int(file_info[-4])
    if change_th != "":
        ctaid_x,ctaid_y,tid_x,tid_y = change_th.split(",")
        ctaid_x = int(ctaid_x)
        ctaid_y = int(ctaid_y)
        tid_x = int(tid_x)
        tid_y = int(tid_y)
    app_name = file_info[0]
    ntid_y = tid_y
    ntid_x = tid_x
    nctaid_y = ctaid_y
    nctaid_x = ctaid_x
    print(file_name, end=", ")
    get_threadID(nctaid_x,nctaid_y,ntid_x,ntid_y,file_name,param_file_name,formular_file_name, app_name)
