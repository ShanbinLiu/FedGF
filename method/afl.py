from task import modelfuncs
from .fedbase import BaseServer, BaseClient
import numpy as np
import copy
from log import *

class Server(BaseServer):
    def __init__(self, option, model, clients):
        super(Server, self).__init__(option, model, clients)
        # algorithm hyper-parameters
        self.dynamic_lambdas = np.ones(self.num_clients) * 1.0 / self.num_clients
        self.learning_rate = option['learning_rate']
        self.learning_rate_lambda = option['learning_rate_lambda']
        self.result_modeldict = copy.deepcopy(self.model.state_dict())
        self.paras_name=['learning_rate_lambda']

    def communicate(self, cid):
        # setting client(cid)'s model with latest parameters
        self.trans_model.load_state_dict(self.model.state_dict())
        self.clients[cid].setModel(self.trans_model)
        # calculate the loss  Fk(w_(t-1)) of the global model on client[k]'s training dataset
        _, loss = self.clients[cid].test('train')
        # wait for replying of the update and loss
        train_w, _, train_acc = self.clients[cid].reply()
        return train_w, loss, train_acc

    def iterate(self, t):
        ws, losses, grads, accs = [], [], [], []
        # training
        for cid in range(self.num_clients):
            w, loss_train ,acc_train= self.communicate(cid)
            print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(self.clients[cid].name, loss_train, acc_train))
            ws.append(w)
            losses.append(loss_train)
            accs.append(acc_train)
            grads.append(modelfuncs.modeldict_scale(modelfuncs.modeldict_sub(self.model.state_dict(), w), 1.0 / self.learning_rate))

        # aggregate grads
        grad = self.aggregate(grads, self.dynamic_lambdas)
        w_new = modelfuncs.modeldict_sub(self.model.state_dict(), modelfuncs.modeldict_scale(grad, self.learning_rate))
        self.model.load_state_dict(w_new)
        # update lambdas
        for lid in range(len(self.dynamic_lambdas)):
            self.dynamic_lambdas[lid] += self.learning_rate_lambda * losses[lid]
        self.dynamic_lambdas = self.project(self.dynamic_lambdas)
        # record resulting model
        self.result_modeldict = modelfuncs.modeldict_add(modelfuncs.modeldict_scale(self.result_modeldict, t), w_new)
        self.result_modeldict = modelfuncs.modeldict_scale(self.result_modeldict, 1.0 / (t + 1))
        # output info
        loss_avg = sum(losses) / len(losses)
        accs_avg = sum(accs) / len(accs)
        return loss_avg, accs_avg

    def aggregate(self, ws, p):
        return modelfuncs.modeldict_weighted_average(ws, p)

    def project(self, p):
        u = sorted(p, reverse=True)
        res = []
        rho = 0
        for i in range(len(p)):
            if (u[i] + (1.0/(i + 1)) * (1 - np.sum(np.asarray(u)[:i+1]))) > 0:
                rho = i + 1
        lmbd = (1.0/rho) * (1 - np.sum(np.asarray(u)[:rho]))
        for i in range(len(p)):
            res.append(max(p[i]+lmbd, 0))
        return res

    def test_on_clients(self, round):
        accs, losses = [], []
        for c in self.clients:
            self.trans_model.load_state_dict(self.result_modeldict)
            acc_test, loss_test = modelfuncs.test(self.trans_model, c.test_data, c.ldr_test, self.batchsize)
            print_log('site-{:<10s}| test loss: {:.4f} | test acc: {:.4f}'.format(c.name, loss_test, acc_test))
            accs.append(acc_test)
            losses.append(loss_test)
        return accs, losses

    def vaild_on_clients(self, round):
        accs, losses = [], []
        for c in self.clients:
            self.trans_model.load_state_dict(self.result_modeldict)
            acc_test, loss_test = modelfuncs.test(self.trans_model, c.val_data, c.ldr_vaild, self.batchsize)
            print_log('site-{:<10s}| test loss: {:.4f} | test acc: {:.4f}'.format(c.name, loss_test, acc_test))
            accs.append(acc_test)
            losses.append(loss_test)
        return accs, losses

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_vaild_dict = {'x':[],'y':[]},data_test_dict={'x':[],'y':[]}, partition = True):
        super(Client, self).__init__(option=option, name=name, data_train_dict =data_train_dict, data_vaild_dict=data_vaild_dict,data_test_dict=data_test_dict, partition =partition)