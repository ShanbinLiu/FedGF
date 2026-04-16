from .fedbase import BaseServer, BaseClient
from log import *
import numpy as np
from method.practice import *
from task import datafuncs, modelfuncs
import time


class Server(BaseServer):
    def __init__(self, option, model, clients):
        super(Server, self).__init__(option, model, clients)
        self.lamda = option['lamda']
        self.client_weights = [self.client_vols[i] / self.data_vol for i in range(self.num_clients)]
        self.flag = False
        self.times_list = []
        self.threshold = option['threshold']
        self.threshold_half = int(self.threshold / 2)
        self.eta = option['eta']
        self.paras_name = ['lamda', 'threshold', 'eta']

    def run(self):
        client_accuracy = []
        std_accuracy = []
        ave_accuracy = []
        ave_train_loss = []
        ave_vaild_accuracy = []
        ave_train_accuracy = []
        gini_test_list = []
        gini_vaild_list = []
        best_server_val_acc, best_server_val_round = 0.0, 0
        best_server_val_gini, best_server_val_gini_round = 1.0, 0
        for client in self.clients:
            client.setModel(self.model)
        for round in range(self.num_rounds):
            print_log("============ Train epoch: {}/{} ============".format(round, self.num_rounds))
            train_losses, train_accs = self.iterate(round)
            ave_train_accuracy.append(train_accs)
            ave_train_loss.append(train_losses)
            if self.eval_interval > 0 and (round == 0 or round % self.eval_interval == 0):
                print_log('============== {} =============='.format('Global Validation'))
                accs_vaild, _ = self.vaild_on_clients(round)
                start_time = time.time()
                gini_vaild = self.gini(accs_vaild)
                gini_vaild_list.append(gini_vaild)
                if round >= self.threshold and round % self.threshold_half == 0 and np.mean(
                        gini_vaild_list[round - self.threshold:round - self.threshold_half]) - np.mean(
                        gini_vaild_list[round - self.threshold_half:round]) < self.eta:
                    self.flag = True
                end_time = time.time()
                execution_time = end_time - start_time
                self.times_list.append(execution_time)
                if self.flag:
                    falseweight = []
                    for client_idx in range(len(self.clients)):
                        falseweight.append(1 - accs_vaild[client_idx])
                    print(falseweight)
                    a = np.sum(falseweight)
                    for i in range(len(self.clients)):
                        falseweight[i] = falseweight[i] / a * self.lamda
                    f_exp = np.exp(falseweight)
                    total = f_exp.sum()
                    for client_idxx in range(len(self.clients)):
                        self.client_weights[client_idxx] = f_exp[client_idxx] / total
                    print(self.client_weights)
                ave_vaild_accuracy.append(np.mean(accs_vaild))
                accs, _ = self.test_on_clients(round)
                client_accuracy.append(accs)
                gini_test = self.gini(accs)
                gini_test_list.append(gini_test)
                ave_accuracy.append(np.mean(accs))
                print_log("Mean of vaild accuracy: {}".format(
                    ave_vaild_accuracy[-1]) + "|" + "Mean of test accuracy: {}".format(ave_accuracy[-1]))
                print_log("The gini coefficient: {}".format(gini_test))
                std_accuracy.append(np.std(accs))
            if ave_vaild_accuracy[-1] > best_server_val_acc:
                best_server_val_acc = ave_vaild_accuracy[-1]
                best_server_val_round = round
            if gini_vaild_list[-1] < best_server_val_gini:
                best_server_val_gini = gini_vaild_list[-1]
                best_server_val_gini_round = round
        print_log('============== {} =============='.format('Summary'))
        print_log('best server val round: {} | best server val acc: {:.4f}'.format(best_server_val_round,
                                                                                   best_server_val_acc))
        print_log('best test acc: {}'.format(ave_accuracy[best_server_val_round]))
        print_log('best std: {}'.format(std_accuracy[best_server_val_round]))
        print_log('best gini coefficient: {}'.format(gini_test_list[best_server_val_round]))
        print_log('best val gini : {}'.format(best_server_val_gini))
        print_log('best val gini round : {}'.format(best_server_val_gini_round))
        outdict = {
            "train_acc": ave_train_accuracy,
            "train_loss": ave_train_loss,
            "test_acc": ave_accuracy,
            "test_std": std_accuracy,
            "test_gini": gini_test_list,
            "client_accs": {},
            "best_test_gini": gini_test_list[best_server_val_round]
        }
        for cid in range(self.num_clients):
            outdict['client_accs'][self.clients[cid].name] = client_accuracy[best_server_val_round][cid]
        self.Gs_list = str(self.Gs_list)
        self.Vc_list = str(self.Vc_list)
        ave_train_accuracy = str(ave_train_accuracy)
        ave_train_loss = str(ave_train_loss)
        ave_accuracy = str(ave_accuracy)
        std_accuracy = str(std_accuracy)
        gini_test_list = str(gini_test_list)
        gini_vaild_list = str(gini_vaild_list)
        mean_time = sum(self.times_list) / len(self.times_list)
        mean_time = str(mean_time)
        self.times_list = str(self.times_list)
        print_log("Gs_list:" + self.Gs_list)
        print_log("Vc_list:" + self.Vc_list)
        print_log("train_acc" + ave_train_accuracy)
        print_log("train_loss" + ave_train_loss)
        print_log("test_acc" + ave_accuracy)
        print_log("test_std" + std_accuracy)
        print_log("test_gini" + gini_test_list)
        print_log("vaild_gini:" + gini_vaild_list)
        print_log("mean_time:" + mean_time)
        print_log("times_list" + self.times_list)
        return outdict

    def iterate(self, t):
        ws, losses, accs = [], [], []
        # sample clients
        if self.clients_per_round == len(self.clients):
            selected_clients = range(len(self.clients))
        else:
            selected_clients = self.sample()
        # training
        for cid in selected_clients:
            w, loss_train, acc_train = self.communicate(cid)
            print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(self.clients[cid].name, loss_train,
                                                                                    acc_train))
            ws.append(w)
            accs.append(acc_train)
            losses.append(loss_train)
        w_new = self.aggregate(ws, p=self.client_weights)
        self.model.load_state_dict(w_new)
        # output info
        loss_avg = sum(losses) / len(losses)
        accs_avg = sum(accs) / len(accs)
        return loss_avg, accs_avg

    def vaild_on_clients(self, round):
        accs, losses = [], []
        for c in self.clients:
            acc_vaild, loss_vaild = modelfuncs.test(self.model, c.val_data, c.ldr_vaild, self.batchsize)
            print_log('site-{:<10s}| vail loss: {:.4f} | vail acc: {:.4f}'.format(c.name, loss_vaild, acc_vaild))
            accs.append(acc_vaild / 100 + 1e-10)
            losses.append(loss_vaild)
        return accs, losses


class Client(BaseClient):
    def __init__(self, option, name='', data_train_dict={'x': [], 'y': []}, data_vaild_dict={'x': [], 'y': []},
                 data_test_dict={'x': [], 'y': []}, partition=True):
        super(Client, self).__init__(option=option, name=name, data_train_dict=data_train_dict,
                                     data_vaild_dict=data_vaild_dict, data_test_dict=data_test_dict,
                                     partition=partition)


