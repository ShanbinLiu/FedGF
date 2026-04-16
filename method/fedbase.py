from curses.ascii import FS
import numpy as np
from task import datafuncs, modelfuncs
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
from task.modelfuncs import device,lossfunc,optim
from method.practice import *
from log import *
import torchvision.transforms as transforms
class BaseServer():
    def __init__(self, option, model, clients):
        # basic setting
        self.model = model
        self.trans_model = copy.deepcopy(self.model).to(modelfuncs.device)
        self.eval_interval = option['eval_interval']
        # clients settings
        self.clients = clients
        self.num_clients = len(self.clients)
        self.client_vols = [ck.datasize for ck in self.clients]
        self.data_vol = sum(self.client_vols)
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        self.clients_per_round = max(int(self.num_clients * option['proportion']), 1)
        # sampling and aggregating methods
        self.sample_option = option['sample']
        self.agg_option = option['aggregate']
        self.lr=option['learning_rate']
        # names of additional parameters
        self.paras_name=[]
        self.Gs_list = []
        self.Vc_list = []
        self.batchsize=option['batch_size']
        self.dataset = option['dataset']

    def gini(self,L):
        s1 = 0  # 分子
        s2 = 0  # 分母
        for i in L:
            s2 += 2 * (len(L)-1) * i
            for j in L:
                s1 += abs(i - j)
        return s1 / s2

    def run(self):
        client_accuracy=[]
        std_accuracy=[]
        ave_accuracy=[]
        ave_train_loss=[]
        client_vaild_accuracy=[]
        ave_vaild_accuracy=[]
        ave_train_accuracy=[]
        gini_test_list=[]
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
            if self.eval_interval>0 and (round==0 or round%self.eval_interval==0):
                print_log('============== {} =============='.format('Global Validation'))
                accs_vaild , _ = self.vaild_on_clients(round)
                gini_vaild = self.gini(accs_vaild)
                gini_vaild_list.append(gini_vaild)
                client_vaild_accuracy.append(accs_vaild)
                ave_vaild_accuracy.append(np.mean(accs_vaild))
                accs , _ = self.test_on_clients(round)
                client_accuracy.append(accs)
                gini_test = self.gini(accs)
                gini_test_list.append(gini_test)
                ave_accuracy.append(np.mean(accs))
                print_log("Mean of vaild accuracy: {}".format(ave_vaild_accuracy[-1])+"|"+"Mean of test accuracy: {}".format(ave_accuracy[-1]))
                print_log("The gini coefficient: {}".format(gini_test))
                std_accuracy.append(np.std(accs))
            if ave_vaild_accuracy[-1] > best_server_val_acc:
                best_server_val_acc = ave_vaild_accuracy[-1]
                best_server_val_round = round
                torch.save(self.model.state_dict(), 'task/{}/record/best_model.pth'.format(self.dataset))
            if gini_vaild_list[-1] < best_server_val_gini:
                best_server_val_gini = gini_vaild_list[-1]
                best_server_val_gini_round = round
        print_log('============== {} =============='.format('Summary'))
        print_log('best server val round: {} | best server val acc: {:.4f}'.format(best_server_val_round,best_server_val_acc))
        print_log('best test acc: {}'.format(ave_accuracy[best_server_val_round]))
        print_log('best std: {}'.format(std_accuracy[best_server_val_round]))
        print_log('best gini coefficient: {}'.format(gini_test_list[best_server_val_round]))
        print_log('best val gini : {}'.format(best_server_val_gini))
        print_log('best val gini round : {}'.format(best_server_val_gini_round))
        print_log('best val gini round test gini : {}'.format(gini_test_list[best_server_val_gini_round]))
        outdict={
            "train_acc":ave_train_accuracy,
            "train_loss":ave_train_loss,
            "test_acc":ave_accuracy,
            "test_std":std_accuracy,
            "test_gini": gini_test_list,
            "client_accs":{},
            "best_test_gini": gini_test_list[best_server_val_round]
            }
        for cid in range(self.num_clients):
            outdict['client_accs'][self.clients[cid].name]=client_accuracy[best_server_val_round][cid]
        self.Gs_list=str(self.Gs_list)
        self.Vc_list=str(self.Vc_list)
        ave_train_accuracy=str(ave_train_accuracy)
        ave_train_loss=str(ave_train_loss)
        ave_accuracy=str(ave_accuracy)
        std_accuracy=str(std_accuracy)
        gini_test_list=str(gini_test_list)
        gini_vaild_list = str(gini_vaild_list)
        print_log("Gs_list:"+self.Gs_list)
        print_log("Vc_list:"+self.Vc_list)
        print_log("train_acc:"+ave_train_accuracy)
        print_log("train_loss:" + ave_train_loss)
        print_log("test_acc:" + ave_accuracy)
        print_log("test_std:" + std_accuracy)
        print_log("test_gini:" + gini_test_list)
        print_log("vaild_gini:" + gini_vaild_list)
        return outdict

    def communicate(self, cid):
        # setting client(cid)'s model with latest parameters
        self.trans_model.load_state_dict(self.model.state_dict())
        self.clients[cid].setModel(self.trans_model)
        # wait for replying of the update and loss
        return self.clients[cid].reply()

    def iterate(self, t):
        ws, losses, accs = [], [], []
        w_c=[]
        # sample clients
        if self.clients_per_round==len(self.clients):
            selected_clients=range(len(self.clients))
        else:
            selected_clients = self.sample()
        # training
        for cid in selected_clients:
            w, loss_train ,acc_train= self.communicate(cid)
            print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(self.clients[cid].name, loss_train, acc_train))
            ws.append(w)
            accs.append(acc_train)
            losses.append(loss_train)
        for i in selected_clients:
            w_c.append(self.clients[i].model.state_dict())
        delta = delta_Dw(self.model, w_c)
        Gs = global_update_scales(delta)
        Vc = local_dissimilarity(w_c, self.model, delta)
        self.Gs_list.append(Gs)
        self.Vc_list.append(Vc)
        # aggregate
        print(selected_clients)
        list_select=[self.client_vols[i] for i in selected_clients]
        print(list_select)
        num=sum(list_select)
        print(num)
        w_new = self.aggregate(ws, p=[1.0*self.client_vols[id]/num for id in selected_clients])
        self.model.load_state_dict(w_new)
        # output info
        loss_avg = sum(losses) / len(losses)
        accs_avg = sum(accs)/len(accs)
        return loss_avg, accs_avg

    def sample(self, replacement=False):
        cids = [i for i in range(self.num_clients)]
        selected_cids = []
        if self.sample_option == 'uniform':
            selected_cids = np.random.choice(cids, self.clients_per_round, replace=replacement)
        elif self.sample_option =='prob':
            selected_cids = np.random.choice(cids, self.clients_per_round, replace=replacement, p=[nk/self.data_vol for nk in self.client_vols])
        return list(selected_cids)

    def aggregate(self, ws, p=[]):
        """
         weighted_scale                 |uniform                    |weighted_com
        ============================================================================================
        N/K * Σpk * wk                 |1/K * Σwi                  |(1-Σpk) * w_old + Σpk * wk
        """
        if self.agg_option == 'weighted_scale':
            K = len(ws)
            N = self.num_clients
            q = [1.0*pk*N/K for pk in p]
            return modelfuncs.modeldict_weighted_average(ws, q)
        elif self.agg_option == 'uniform':
            return modelfuncs.modeldict_weighted_average(ws, p)
        elif self.agg_option == 'weighted_com':
            return modelfuncs.modeldict_add(modelfuncs.modeldict_scale(self.model.state_dict(), 1 - sum(p)), modelfuncs.modeldict_weighted_average(ws, p))

    def test_on_clients(self, round):
        accs, losses = [], []
        for c in self.clients:
            acc_test, loss_test = modelfuncs.test(self.model, c.test_data, c.ldr_test, self.batchsize)
            print_log('site-{:<10s}| test loss: {:.4f} | test acc: {:.4f}'.format(c.name, loss_test, acc_test))
            accs.append(acc_test)
            losses.append(loss_test)
        return accs, losses

    def vaild_on_clients(self, round):
        accs, losses = [], []
        for c in self.clients:
            acc_vaild, loss_vaild = modelfuncs.test(self.model, c.val_data, c.ldr_vaild, self.batchsize)
            print_log('site-{:<10s}| vail loss: {:.4f} | vail acc: {:.4f}'.format(c.name, loss_vaild, acc_vaild))
            accs.append(acc_vaild)
            losses.append(loss_vaild)
        return accs, losses


class BaseClient():
    def __init__(self,  option, name = '', data_train_dict = {'x':[],'y':[]},data_vaild_dict = {'x':[],'y':[]},data_test_dict={'x':[],'y':[]}, partition = True):
        self.name = name
        self.frequency = 0
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
        ])
        # client's dataset
        if option['dataset'] == 'domainnet' and not partition:
            print('domainnet')
            self.train_data = datafuncs.DomainnetDataset(data_train_dict['x'], data_train_dict['y'], self.train_transform)
            self.test_data = datafuncs.DomainnetDataset(data_test_dict['x'], data_test_dict['y'])
            self.val_data = datafuncs.DomainnetDataset(data_vaild_dict['x'], data_vaild_dict['y'])
        elif option['dataset'] == 'office10' and not partition:
            print('office10')
            self.train_data = datafuncs.XYDataset(data_train_dict['x'], data_train_dict['y'])
            self.test_data = datafuncs.XYDataset(data_test_dict['x'], data_test_dict['y'])
            self.val_data = datafuncs.XYDataset(data_vaild_dict['x'], data_vaild_dict['y'])
        else:
            print('other')
            data_x = data_train_dict['x'] + data_test_dict['x']
            data_y = data_train_dict['y'] + data_test_dict['y']
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)
            self.train_data = datafuncs.XYDataset(data_x[:int(len(data_x)*0.8)], data_y[:int(len(data_y)*0.8)])
            self.test_data = datafuncs.XYDataset(data_x[int(len(data_x)*0.8):int(len(data_x)*0.9)], data_y[int(len(data_x)*0.8):int(len(data_x)*0.9)])
            self.val_data = datafuncs.XYDataset(data_x[int(len(data_x)*0.9):], data_y[int(len(data_x)*0.9):])
        self.datasize = len(self.train_data)
        # hyper-parameters for training
        self.epochs = option['num_epochs']
        self.learning_rate = option['learning_rate']
        self.batch_size = min(option['batch_size'], self.datasize)
        self.momentum = option['momentum']
        self.weight_decay=option['weight_decay']
        self.model = None
        self.ldr_train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.ldr_vaild = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.ldr_test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def setModel(self, model):
        self.model = copy.deepcopy(model)

    def train(self):
        self.model.train()
        if self.batch_size == -1:
            # full gradient descent
            self.batch_size = len(self.train_data)
        optimizer = optim(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,weight_decay=self.weight_decay)
        epoch_loss = []
        epoch_accs = []
        for iter in range(self.epochs):
            batch_loss = []
            output_all = torch.tensor([], dtype=torch.float32, device=device)
            label_all = torch.tensor([], dtype=torch.float32, device=device)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(device), labels.to(device)
                self.model.zero_grad()
                outputs = self.model(images)
                loss = lossfunc(outputs, labels)
                output_all = torch.cat([output_all, outputs.detach()], dim=0)
                label_all = torch.cat([label_all, labels.detach()], dim=0)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item() / len(labels))
            epoch_accs.append(self.top1_acc(output_all,label_all))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss), sum(epoch_accs)/len(epoch_accs)

    def test(self, dataflag='test'):
        if dataflag == 'test':
            return modelfuncs.test(self.model, self.test_data, self.ldr_test, self.batch_size)
        elif dataflag == 'train':
            return modelfuncs.test(self.model, self.train_data, self.ldr_train,self.batch_size)
        elif dataflag =='validate':
            return modelfuncs.test(self.model, self.val_data, self.ldr_vaild, self.batch_size)

    def reply(self):
        loss,acc = self.train()
        return copy.deepcopy(self.model.state_dict()), loss, acc

    def top1_acc(self,output, label):
        total = label.size(0)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.view(-1)).sum().item()

        return correct * 1.0 / total
