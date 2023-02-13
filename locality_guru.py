import json
import os, sys
import argparse
import re
from ptx_files import ptx_tracing_string
from numpy import sort
from tqdm import tqdm

formular_tree = dict()
syntax_tree = dict()
bb_graph = dict()
kernel_info = dict()
parameter_dict = dict()
inf_ = 987654321
address_ = 1024
search_type = ["ld.global"]
shared_mode = 0
op_dict = {
    "abs":ptx_tracing_string.ABS,
    "add":ptx_tracing_string.ADD,
    "sub":ptx_tracing_string.SUB,
    "mad": ptx_tracing_string.MADLO,
    "fma": ptx_tracing_string.MADLO,
    "mul":ptx_tracing_string.MUL,
    "ld": ptx_tracing_string.LD,
    "st": ptx_tracing_string.ST,
    "not": ptx_tracing_string.NOT,
    "cvt": ptx_tracing_string.CVTA,
    "cvta": ptx_tracing_string.CVTA,
    "mov": ptx_tracing_string.MOV,
    "shl": ptx_tracing_string.SHL,
    "shr": ptx_tracing_string.SHR,
    "or": ptx_tracing_string.OR,
    "bfe": ptx_tracing_string.BFE,
    "prmt": ptx_tracing_string.PRMT,
    "sqrt": ptx_tracing_string.SQRT,
    "min": ptx_tracing_string.MIN,
    "max": ptx_tracing_string.MAX,
    "neg": ptx_tracing_string.NEG,
    "and": ptx_tracing_string.AND,
    "div": ptx_tracing_string.DIV,
    "rem": ptx_tracing_string.REM,
    "rcp": ptx_tracing_string.RCP,
    "selp": ptx_tracing_string.SELP,
    "setp":ptx_tracing_string.SETP,
    "clz": ptx_tracing_string.CLZ,
    "setp.ge": ptx_tracing_string.SETP_GE,
    "setp.lt":ptx_tracing_string.SETP_LT,
    "setp.ne":ptx_tracing_string.SETP_NE,
    "setp.gt":ptx_tracing_string.SETP_GT,
    "setp.eq":ptx_tracing_string.SETP_EQ
}

def get_opcode(inst,search_flag="st"):
    initial = inst
    inst = re.sub("\n","",inst) #inst.split("\n")[0]
    inst = re.sub("[,;\[\]\{\}]","",inst)
    inst = inst.split(" ")
    try:
        inst.remove("")
    except:
        print("",end="")
    if len(inst)<3:
        return "-999999999", "-999999999", ["-999999999"]
    
    '''
    if tmp[0]=="setp":
        inst[0] = tmp[0]+"."+tmp[1]
    else:
        inst[0] = tmp[0]
    '''
    search_flag = search_flag.split('.')[0]
    if inst[0].startswith(search_flag):
        tmp = inst[0].split(".")
        if(search_flag=="st"):
            # print("DDD")
            t = inst[-1]
            inst[-1] = inst[-2]
            inst[-2] = t
        # print(tmp)
        try:
            if tmp[2]=='v2':
                dst = dict()
                offset = int(tmp[3][1:])
                src_offset = inst[-1].split("+")
                if len(inst[-1].split("+"))>1:
                    dst[inst[1]]=inst[-1]
                    dst[inst[2]]=src_offset[0]+"+"+str(offset+int(src_offset[1]))
                else:
                    dst[inst[1]]=inst[-1]
                    dst[inst[2]]=inst[-1]+"+"+str(offset)
                return inst[0], dst,"-999999999"
        except:
            print("Error")
    # print("DDDDDDDDDDs")
    return inst[0], inst[1], inst[2:]

def initialize_trees(ptx_file_name):
    global formular_tree
    global syntax_tree
    global bb_graph
    global kernel_info
    global parameter_dict
    global address_
    global inf_
    with open(ptx_file_name,"r") as f:
        instructions = f.readlines()
        kernel_name = ""
        inst_len = len(instructions)
        bb_n = -1
        large_bb_n = -1
        for idx_, inst in enumerate(instructions):    
            if inst.startswith(".visible"): #find kernel name
                kernel_name = re.sub("\(\n","",inst.split(" ")[-1])
                syntax_tree[kernel_name] = dict()
                bb_graph[kernel_name] = dict()
                kernel_info[kernel_name] = {"start_id":idx_, "end_id":-1}
                #kernel_info[kernel_name] ddd
                bb_n = -1
            elif inst.startswith("}"):
                kernel_info[kernel_name]["end_id"] = idx_
            elif inst.startswith("$"): #find basic block(where $ is not in)
                #bb_n = int(re.sub("[:\n]","",inst.split("_")[-1]))
                try:
                    bb_n = int(re.sub("[:\n]","",inst.split("_")[-1]))
                except:
                    continue
                bb_graph[kernel_name][bb_n] = {"from_bb":inst,"visited":False,"next":-1,"start_line":idx_,"end_line":0,"has_ld_global":False}
            elif inst.startswith(search_type[shared_mode]):#("ld.global"): #find global instruction
                opcode, dst, src = get_opcode(inst,search_type[shared_mode])
                # print(f"inst:{inst}, {src}")
                offset = "0"
                tmp = src[-1].split("+")
                if len(tmp) == 2:
                    offset = "+"+tmp[-1]
                src_ = tmp[0]+"_"+str(idx_)
                src = tmp[0]
                syntax_tree[kernel_name][src_] = list()
                ld_dict = {"reg_name":src, "score_from_up":inf_, "score_from_down":0, "child":0, "my_idx":-1,"parent_loc":-1, "parent_reg":"", "BB_N":bb_n, "opcode":"",  "child0":0,"child1":0,"child2":0, "offset":offset,"line":idx_-kernel_info[kernel_name]["start_id"]}
                if bb_n != -1:
                    bb_graph[kernel_name][bb_n]["has_ld_global"]=True
                syntax_tree[kernel_name][src_].append(ld_dict)          
            if inst == "\n" and bb_n != -1:
                if bb_n in bb_graph[kernel_name]:
                    bb_graph[kernel_name][bb_n]["end_line"] = idx_
                    '''
                    loop_ = instructions[idx_-1]
                    if loop_.startswith("@"):
                        loop_ = re.sub(r"[\n@;]","",loop_)
                        loop_ = loop_.split( )
                        print(loop_)
                        bb_graph[kernel_name][bb_n]["next"] = int(loop_[-1].split("_")[-1])
                        
                        ld_dict = {"reg_name":loop_[0], "score_from_up":inf_, "score_from_down":0, "child":0, "my_idx":-1,"parent_loc":-1, "parent_reg":"", "BB_N":bb_n, "opcode":"",  "child0":0,"child1":0,"child2":0, "offset":offset,"line":idx_-kernel_info[kernel_name]["start_id"]}
                        syntax_tree[kernel_name][loop_[0]+"_"+loop_[1]+"_"+str(idx_)] = list()
                        syntax_tree[kernel_name][loop_[0]+"_"+loop_[1]+"_"+str(idx_)].append(ld_dict)
                    '''
                    bb_n = -1

                    
def trace_syntax_tree(total_ld_global_info, kernel_lines,start_idx):
    global formular_tree
    global syntax_tree
    global bb_graph
    global kernel_info
    global parameter_dict
    global address_
    global inf_
    ld_global_info = total_ld_global_info[start_idx]
    ld_global = ld_global_info["reg_name"]
    #print(ld_global)
    #print(kernel_lines[ld_global_info["line"]])
    start_line = ld_global_info["line"]
    line_length = len(kernel_lines[start_line:])
    syntax_tree_len = len(total_ld_global_info)
    for idx_, line_ in enumerate(reversed(kernel_lines[:start_line])):
        opcode, dst, src = get_opcode(line_,search_type[shared_mode])

        if src[0] == "-999999999":
            continue
        if ld_global in dst and type(dst)==dict:
            # print("dddddddd")
            src = [dst[ld_global]]
            dst = ld_global

        if dst == ld_global:     
            ld_global_info["child"] = len(src)
            ld_global_info["opcode"] = opcode
            for id, s in enumerate(src):
                offset = "0"
                
                tmp = s.split("+")
                if len(tmp) == 2:
                    offset = "+"+tmp[-1]
                s = tmp[0]
                ld_global_info[f"child{id}"] = syntax_tree_len+id
                ld_dict = {"reg_name":s, "score_from_up":inf_, "score_from_down":0, "child":0, "my_idx":id,"parent_loc":start_idx, "parent_reg":ld_global, "BB_N":-1, "opcode":"",  "child0":0,"child1":0,"child2":0, "offset":offset,"line":start_line-(idx_+1)} #-(idx_)
                total_ld_global_info.append(ld_dict)
            
            for id, _ in enumerate(src):
                trace_syntax_tree(total_ld_global_info, kernel_lines,syntax_tree_len+id)
    return total_ld_global_info

def print_syntax_tree(syntax_tree_global, my_idx, depth):
    global formular_tree
    global syntax_tree
    global bb_graph
    global kernel_info
    global parameter_dict
    global address_
    global inf_
    my_dict = syntax_tree_global[my_idx]
    child_len = my_dict["child"]
    full_opcode_ = my_dict["opcode"]
    opcode_ = full_opcode_.split(".")[0]
    options_ = []
    if len(full_opcode_.split("."))>1:
        options_ = full_opcode_.split(".")[1:]
    indent = "    "
    reg_name = my_dict["reg_name"]
    offset_ = my_dict["offset"]
    '''
    print(f"{indent*depth}{reg_name}")
    print(f"{indent*(depth+1)}{opcode_}")
    '''
    if child_len == 0 :
        try:
            param = int(my_dict["reg_name"])
            #parameter_dict[my_dict["reg_name"]] = param
            if offset_ != "0":
                return my_dict["reg_name"]+offset_
            return my_dict["reg_name"]
        except:
            if my_dict["reg_name"].startswith("%"):
                param = re.sub("%","",my_dict["reg_name"])
                param = re.sub("\.","_",param)
                #if my_dict["reg_name"] not in parameter_dict:
                #    parameter_dict[my_dict["reg_name"]] = param
                if offset_ != "0":
                    param += offset_
                return "{"+param+"}"
                #ddddd
            if my_dict["reg_name"] not in parameter_dict:
                parameter_dict[my_dict["reg_name"]] = address_
                address_ +=address_
                if offset_ != "0":
                    return "{"+my_dict["reg_name"]+"}"+offset_
            return "{"+my_dict["reg_name"]+"}"
    if child_len == 1:
        tmp1 = print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child0"],depth+1)
        #print("one child")
        try:
            if offset_ !="0":
                if opcode_=="clz":
                    return op_dict.get(opcode_)(tmp1,full_opcode_) + offset_
                return op_dict.get(opcode_)(tmp1) + offset_
            if opcode_=="clz":
                return op_dict.get(opcode_)(tmp1,full_opcode_)
            return op_dict.get(opcode_)(tmp1) 
        except Exception as e:
            print(e)
            print(opcode_)
            print("************************************")
            exit(1)
    elif child_len == 2:
        tmp1 =print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child0"],depth+1)
        tmp2 = print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child1"],depth+1)
        #print("two child")
        try:
            if offset_ != "0":
                return op_dict.get(opcode_)(tmp1,tmp2) + offset_
            return op_dict.get(opcode_)(tmp1,tmp2)
        except Exception as e:
            print(e)
            print(tmp1)
            print(tmp2)
            print(options_)
            print(opcode_)
            print("*****************************************")
            exit(1)
    elif child_len == 3:
        #print("three child")
        tmp1 = print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child0"],depth+1)
        tmp2 = print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child1"],depth+1)
        tmp3 = print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child2"],depth+1)
        try:
            if offset_ != "0":
                return op_dict.get(opcode_)(tmp1,tmp2,tmp3) + offset_
            return op_dict.get(opcode_)(tmp1,tmp2, tmp3)
        except:
            print(e)
            print(opcode_)

def trace(ptx_file_name):
    global formular_tree
    global syntax_tree
    global bb_graph
    global kernel_info
    global parameter_dict
    global address_
    global inf_
    # address_ = 10240
    f = open(ptx_file_name,"r")
    file_lines = f.readlines()
    for kernel_name in kernel_info:
        formular_tree[kernel_name] = dict()
        kernel_lines = file_lines[kernel_info[kernel_name]["start_id"]:kernel_info[kernel_name]["end_id"]]
        print("       "+kernel_name)
        for ld_global in syntax_tree[kernel_name]:
            formular_tree[kernel_name][ld_global] = dict()
            #print(ld_global)
            #print("_____________________________")
            trace_syntax_tree(syntax_tree[kernel_name][ld_global],kernel_lines,0)   
            #print(print_syntax_tree(syntax_tree[kernel_name][ld_global], 0,0))
            final_formular = print_syntax_tree(syntax_tree[kernel_name][ld_global], 0,0)
            formular_tree[kernel_name][ld_global]["final_formular"] = final_formular
            #exit(1)


def main(ptx_file_name):
    global formular_tree
    global syntax_tree
    global bb_graph
    global kernel_info
    global parameter_dict
    global address_
    global inf_

    initialize_trees(ptx_file_name)
    trace(ptx_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="dir name",default="./original/")
    parser.add_argument("-f", help="file name",default="")
    parser.add_argument("-s", help="shared mode",default=0, type=int)

    args = parser.parse_args()
    dir_name = args.d
    file_input = args.f
    shared_mode = args.s
    shared_flag=["."]
    print(file_input)

    if dir_name!="" and file_input!="":

        main(os.path.join(dir_name,file_input))
        # exit(1)
        file_dir = file_input.split("/")[-1]
        file_dir = file_dir.split(".")[0]
        print(file_dir)
        save_dir = f"syntax_tree/{file_dir}"
        print(save_dir)
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)
        with open(os.path.join(save_dir,file_dir+f"_param.json"),"w") as f:
            json.dump(parameter_dict, f, indent=4)
        with open(os.path.join(save_dir,file_dir+f"_st.json"),"w") as f:
            json.dump(syntax_tree, f, indent=4)
        with open(os.path.join(save_dir,file_dir+f"_formular.json"),"w") as f:
                json.dump(formular_tree, f, indent=4)
    else:
        file_list = os.listdir(dir_name)
        for file_name in file_list:
            
            if file_name == "particlefilter.ptx" :
                continue
            # if file_name.startswith("b+") or file_name.startswith("heart") or file_name.startswith("hotspot") or file_name.startswith("3D") or file_name.startswith("convSep") or file_name.startswith("heart"):
            #    continue
            formular_tree = dict()
            syntax_tree = dict()
            bb_graph = dict()
            kernel_info = dict()
            parameter_dict = dict()
            #address_ = 10240

            #main(os.path.join(dir_name,file_name))
            try:
                print(file_name)
                main(os.path.join(dir_name,file_name))
            except Exception as e:
                print(e)
                print(f"error in {file_name}")
                continue
            file_dir = file_name.split("/")[-1]
            file_dir = file_dir.split(".")[0]
            print(file_dir)
            #exit(1)
            save_dir = f"ptx_files/syntax_tree/{file_dir}"
            print(save_dir)
            if os.path.isdir(save_dir) == False:
                os.makedirs(save_dir)

            with open(os.path.join(save_dir,file_dir+f"_param{shared_flag[shared_mode]}json"),"w") as f:
                json.dump(parameter_dict, f, indent=4)
            with open(os.path.join(save_dir,file_dir+f"_st{shared_flag[shared_mode]}json"),"w") as f:
                json.dump(syntax_tree, f, indent=4)
            with open(os.path.join(save_dir,file_dir+f"_formular{shared_flag[shared_mode]}json"),"w") as f:
                json.dump(formular_tree, f, indent=4)

