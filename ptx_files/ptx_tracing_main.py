import ptx_tracing
from ptx_tracing import file_open,make_ctaid_map,tracing, OPERATE, ADD, MADLO, MUL, SHL, backprop_init


tidx_list = list(range(0,16))
tidy_list = list(range(0,16))
ctaidy_list = list(range(0,ctaidy))
ctaid_map = list()


def backprop_init():
    global tidx
    global tidy
    global ctaidx
    global ctaidy
    
    tidx = 16
    tidy = 16
    ctaidx = 1
    ctaidy = 1024

    tidx_list = list(range(0,tidx)) #list(range(0,512))
    tidy_list = list(range(0,tidy))
    ctaidy_list = list(range(0,ctaidy))
    ctaidx_list = list(range(0,ctaidx))


    param_dict["%ctaid.y"] = ctaidy_list #1024/16
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




def main():
    
    backprop_init()

if __name__== "__main__":
    main()