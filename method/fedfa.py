from task import modelfuncs
import copy
from .fedbase import BaseServer, BaseClient
import numpy as np
from log import *

class Server(BaseServer):
    def __init__(self, option, model, clients):
        super(Server, self).__init__(option, model, clients)
        self.m = modelfuncs.modeldict_zeroslike(self.model.state_dict())
        self.beta = option['beta']
        self.alpha = 1.0 - self.beta
        self.gamma = option['gamma']
        self.learning_rate = option['learning_rate']
        self.paras_name=['beta','gamma','momentum']

    def communicate(self, cid):
        # setting client(cid)'s model with latest parameters
        self.trans_model.load_state_dict(self.model.state_dict())
        self.clients[cid].setModel(self.trans_model)
        # wait for replying of the update and loss
        w = self.clients[cid].reply()[0]
        freq = self.clients[cid].frequency
        acc,loss = self.clients[cid].test('train')
        return w, loss, acc, freq

    def iterate(self, t):
        ws, losses, ACC, F = [], [], [], []
        # sample clients
        selected_clients = self.sample()
        # training
        for cid in selected_clients:
            w, loss_train, acc_train, freq = self.communicate(cid)
            print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(self.clients[cid].name, loss_train, acc_train))
            ws.append(w)
            losses.append(loss_train)
            ACC.append(acc_train)
            F.append(freq)
        # aggregate
        # calculate ACCi_inf, fi_inf
        sum_acc = np.sum(ACC)
        sum_f = np.sum(F)
        ACCinf = [-np.log2(1.0*acc/sum_acc+0.000001) for acc in ACC]
        Finf = [-np.log2(1-1.0*f/sum_f+0.00001) for f in F]
        sum_acc = np.sum(ACCinf)
        sum_f = np.sum(Finf)
        ACCinf = [acc/sum_acc for acc in ACCinf]
        Finf = [f/sum_f for f in Finf]
        # calculate weight = αACCi_inf+βfi_inf
        p = [self.alpha*accinf+self.beta*finf for accinf,finf in zip(ACCinf,Finf)]
        w_new = self.aggregate(ws, p)
        dw = modelfuncs.modeldict_sub(w_new, self.model.state_dict())
        # calculate m = γm+(1-γ)dw
        self.m = modelfuncs.modeldict_add(modelfuncs.modeldict_scale(self.m, self.gamma), modelfuncs.modeldict_scale(dw, 1 - self.gamma))
        if t%5==0:
            w_new = modelfuncs.modeldict_sub(w_new, modelfuncs.modeldict_scale(self.m, self.learning_rate))
        self.model.load_state_dict(w_new)
        # output info
        loss_avg = sum(losses) / len(losses)
        accs_avg = sum(ACC) / len(ACC)
        return loss_avg, accs_avg

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_vaild_dict = {'x':[],'y':[]},data_test_dict={'x':[],'y':[]}, partition = True):
        super(Client, self).__init__(option=option, name=name, data_train_dict =data_train_dict, data_vaild_dict=data_vaild_dict,data_test_dict=data_test_dict, partition =partition)
        self.frequency = 0

    def reply(self):
        self.frequency += 1
        loss, acc = self.train()
        return copy.deepcopy(self.model.state_dict()), loss, acc
