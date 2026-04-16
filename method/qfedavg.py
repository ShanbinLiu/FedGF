from task import modelfuncs
from .fedbase import BaseServer, BaseClient
import numpy as np
import torch
from log import *
import math

class Server(BaseServer):
    def __init__(self, option, model, clients):
        super(Server, self).__init__(option, model, clients)
        self.q = option['q']
        self.learning_rate = option['learning_rate']
        self.L = 1.0/option['learning_rate']
        self.paras_name = ['q']
        print('q:',self.q)
        print('learning_rate:',self.learning_rate)
        print('L:',self.L)

    def communicate(self, cid):
        # setting client(cid)'s model with latest parameters
        self.trans_model.load_state_dict(self.model.state_dict())
        self.clients[cid].setModel(self.trans_model)
        # wait for replying of the update and loss
        # calculate the loss  Fk(w_(t-1)) of the global model on client[k]'s training dataset
        loss = self.clients[cid].test('train')[1]
        train_w, _, train_acc = self.clients[cid].reply()
        return train_w, loss, train_acc

    def iterate(self, t):
        ws, losses, Deltas, hs, accs = [], [], [], [], []
        # sample clients
        if self.clients_per_round == len(self.clients):
            selected_clients = range(len(self.clients))
        else:
            selected_clients = self.sample()
        # training
        for cid in selected_clients:
            w, loss_train, acc_train = self.communicate(cid)
            print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(self.clients[cid].name, loss_train, acc_train))
            ws.append(w)
            losses.append(loss_train)
            accs.append(acc_train)
            # plug in the weight updates into the gradient
            grad = modelfuncs.modeldict_scale(modelfuncs.modeldict_sub(self.model.state_dict(), w), 1.0 / self.learning_rate)
            delta = modelfuncs.modeldict_scale(grad, np.float_power(loss_train + 1e-10, self.q))
            Deltas.append(delta)
            norm_dwk = 0.0
            for key, _ in self.model.named_parameters():
                norm_dwk+=(grad[key]**2).sum()
            hs.append(self.q * np.float_power(loss_train + 1e-10, (self.q - 1)) * norm_dwk + self.L * np.float_power(loss_train + 1e-10, self.q))
        # aggregate
        w_new = self.aggregate(Deltas, hs)
        self.model.load_state_dict(w_new)
        # output info
        loss_avg = sum(losses) / len(losses)
        accs_avg = sum(accs) / len(accs)
        return loss_avg, accs_avg

    def aggregate(self, Deltas, hs):
        demominator=sum(hs)
        scaled_deltas = []
        for delta in Deltas:
            scaled_deltas.append(modelfuncs.modeldict_scale(delta, 1.0 / demominator))
        updates = {}
        for layer in scaled_deltas[0].keys():
            updates[layer] = torch.zeros_like(scaled_deltas[0][layer])
            for sdelta in scaled_deltas:
                updates[layer] += sdelta[layer]
        w_new = modelfuncs.modeldict_sub(self.model.state_dict(), updates)
        return w_new

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_vaild_dict = {'x':[],'y':[]},data_test_dict={'x':[],'y':[]}, partition = True):
        super(Client, self).__init__(option=option, name=name, data_train_dict =data_train_dict, data_vaild_dict=data_vaild_dict,data_test_dict=data_test_dict, partition =partition)
