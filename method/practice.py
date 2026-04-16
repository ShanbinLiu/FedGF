import torch
import numpy as np
def local_dissimilarity(clients_model,server_model,dw):
    total = 0
    dw_client = {}
    for key in server_model.state_dict().keys():
        dw_client[key]=torch.zeros_like(server_model.state_dict()[key]).float()
    for key in server_model.state_dict().keys():
        tem_square=torch.zeros_like(server_model.state_dict()[key]).float()
        for i in range(len(clients_model)):
            temporary=server_model.state_dict()[key]-clients_model[i][key]-dw[key]
            tem_square=tem_square+torch.pow(temporary,2)
        dw_client[key]=tem_square/len(clients_model)
    for key in server_model.state_dict().keys():
        total=total+torch.sum(dw_client[key])
    params_total = sum([p.numel() for p in dw.values()])
    Vc=float(total/params_total)
    return Vc
def global_update_scales(Dw):
    total=0
    params = [param for param in Dw.values()]
    total_params = sum([p.numel() for p in Dw.values()])
    for param in params:
        square=torch.pow(param,2)
        total=total+torch.sum(square)
    G_s=float(total/total_params)
    return G_s
def delta_Dw(server_model,clients_model):
    delta={}
    for key in server_model.state_dict().keys():
        delta[key]=torch.zeros_like(server_model.state_dict()[key]).float()
    for key in server_model.state_dict().keys():
        tmp = torch.zeros_like(server_model.state_dict()[key]).float()
        for client_idx in range(len(clients_model)):
            tmp = tmp + server_model.state_dict()[key]-clients_model[client_idx][key]
        tmp = tmp / len(clients_model)
        delta[key] = tmp
    return delta
def deltaa(ws_list,wc_list):
    for i in range(30):
        a=ws_list[i]['layer.weight'].equal(wc_list[i]['layer.weight'])
        b=ws_list[i]['layer.bias'].equal(wc_list[i]['layer.bias'])
        print([a,b])
def deltaaa(model,models):
    a=torch.equal(model['layer.weight'],models['layer.weight'])
    b=torch.equal(model['layer.bias'], models['layer.bias'])
    print([a,b])