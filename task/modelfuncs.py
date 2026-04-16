import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import numpy as np

device=None
optim = None
lossfunc=None

def test(model, dataset, loader, batchsize):
    model.eval()
    test_loss = 0
    #data_loader = DataLoader(dataset, batch_size=batchsize)
    l = len(loader)
    output_all = torch.tensor([], dtype=torch.float32, device=device)
    label_all = torch.tensor([], dtype=torch.float32, device=device)
    with torch.no_grad():
        for idx, (features, labels) in enumerate(loader):
            features, labels = features.to(device), labels.to(device)
            log_probs = model(features)
            test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
            output_all = torch.cat([output_all, log_probs.detach()], dim=0)
            label_all = torch.cat([label_all, labels.detach()], dim=0)
    test_loss /= len(dataset)
    accuracy = top_acc(output_all,label_all)*100
    return accuracy, test_loss

def top_acc(output, label):
    total = label.size(0)
    pred = output.data.max(1)[1]
    correct = pred.eq(label.view(-1)).sum().item()

    return correct * 1.0 / total

def modeldict_weighted_average(ws, weights=[]):
    w_avg = {}
    for layer in ws[0].keys():
        w_avg[layer] = torch.zeros_like(ws[0][layer])
    if weights == []: weights = [1.0/len(ws) for i in range(len(ws))]
    for wid in range(len(ws)):
        for layer in w_avg.keys():
            w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights[wid]
    return w_avg

def modeldict_zeroslike(w):
    res = {}
    for layer in w.keys():
        res[layer] = w[layer] - w[layer]
    return res

def modeldict_scale(w, c):
    res = {}
    for layer in w.keys():
        res[layer] = w[layer] * c
    return res

def modeldict_sub(w1, w2):
    res = {}
    for layer in w1.keys():
        res[layer] = w1[layer] - w2[layer]
    return res

def modeldict_norm(w, p=2):
    return torch.norm(modeldict_to_tensor1D(w), p)

def modeldict_to_tensor1D(w):
    res = torch.Tensor().to(device)
    for layer in w.keys():
        res = torch.cat((res, w[layer].view(-1)))
    return res

def modeldict_to_param(w):
    res = torch.Tensor().to(device)
    for param in w.parameters():
        res = torch.cat((res, param.view(-1)))
    return res

def modeldict_add(w1, w2):
    res = {}
    for layer in w1.keys():
        res[layer] = w1[layer] + w2[layer]
    return res

def modeldict_dot(w1, w2):
    res = 0
    for layer in w1.keys():
        w1[layer] = w1[layer].to(torch.float32)
        w2[layer] = w2[layer].to(torch.float32)
        s = 1
        for l in w1[layer].shape:
            s *= l
        res += (w1[layer].view(1, s).mm(w2[layer].view(1, s).T))
    return res.item()

def modeldict_dot_layer(w1, w2):
    res = {}
    for layer in w1.keys():
        w1[layer] = w1[layer].to(torch.float32)
        w2[layer] = w2[layer].to(torch.float32)
        s = 1
        for l in w1[layer].shape:
            s *= l
        res[layer]=(w1[layer].view(1, s).mm(w2[layer].view(1, s).T))
    return res

def modeldict_print(w):
    for layer in w.keys():
        print("{}:{}".format(layer, w[layer]))

def modeldict_propsub(w1, w2, prop):
    res = {}
    for layer in w1.keys():
        res[layer] = w1[layer] - w2[layer] * prop
    return res

def invert_grad(m1, m2):
    #  reconstruct the input from the network's gradients
    m1 = np.array([[i.data.item()] for i in m1])
    return torch.matmul(torch.tensor(np.linalg.pinv(m1)), m2.double().cpu()).to(device)

def modeldict_weighted_average_va(ws, weights=[]):
    w_avg = {}
    for layer in ws[0].keys():
        w_avg[layer] = torch.zeros_like(ws[0][layer])
    weights_normal =[1.0/len(ws) for i in range(len(ws))]
    for wid in range(len(ws)):
        for layer in w_avg.keys():
            if 'layer3' in layer or 'layer4' in layer or 'fc' in layer:
                w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights[wid]
            else:
                w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights_normal[wid]
    return w_avg

def modeldict_weighted_average_non_liner(ws, weights_layer1=[],weights_layer2=[],weights_layer3=[],weights_layer4=[],weights_layer5=[],weights_layer6=[]):
    w_avg = {}
    for layer in ws[0].keys():
        w_avg[layer] = torch.zeros_like(ws[0][layer])
    for wid in range(len(ws)):
        for layer in w_avg.keys():
            if 'layer1' in layer:
                w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights_layer2[wid]
            elif 'layer2' in layer:
                w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights_layer3[wid]
            elif 'layer3' in layer:
                w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights_layer4[wid]
            elif 'layer4' in layer:
                w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights_layer5[wid]
            elif 'fc' in layer:
                w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights_layer6[wid]
            else:
                w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights_layer1[wid]
    return w_avg

if __name__ == '__main__':
    #w= {'a': torch.Tensor([[1, 4], [3, 4]]),  'c':torch.Tensor([1])}
    a = {'a': torch.Tensor([[1, 4], [3, 4]])}
    w=torch.Tensor([1, 4, 3, 4])
    res=modeldict_norm(a)
    print(res)
    norm=torch.norm(w)
    print(norm)


