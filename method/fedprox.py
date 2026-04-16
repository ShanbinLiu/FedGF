from .fedbase import BaseServer, BaseClient
import torch.nn.functional as F
from torch.utils.data import DataLoader
from task.modelfuncs import device,lossfunc,optim, modeldict_to_tensor1D
import copy
import torch

class Server(BaseServer):
    def __init__(self, option, model, clients):
        super(Server, self).__init__(option, model, clients)

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_test_dict={'x':[],'y':[]}, partition = True):
        super(Client, self).__init__(option, name, data_train_dict, data_test_dict, partition)
        self.mu=option['mu']
        self.paras_name = ['mu']

    def train(self):
        # global parameters
        src_model = copy.deepcopy(self.model.state_dict())
        src_vec = modeldict_to_tensor1D(src_model)
        self.model.train()
        if self.batch_size == -1:
            self.batch_size = len(self.train_data)
        optimizer = optim(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
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
                tmp_vec = torch.Tensor().to(device)
                for param in self.model.state_dict().values():
                    tmp_vec = torch.cat((tmp_vec, param.view(-1)))
                reg_loss = torch.norm(tmp_vec - src_vec, 2)**2
                loss+=self.mu/2*reg_loss
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item() / len(labels))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_accs.append(self.top1_acc(output_all, label_all))
        return sum(epoch_loss) / len(epoch_loss), sum(epoch_accs)/len(epoch_accs)

