import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class dataset(object):
    def __init__(self, batch_size, data_path):
        
        #createBatchData
        session_key = 'SessionID'
        item_key = 'ItemID'
        time_key = 'Time'
        self.batch_size = batch_size 

        self.trainDF = pd.read_csv(data_path, sep=',', dtype={session_key: int, item_key: int, time_key: float})
        print('data reading done')
 
        #set index for item
        
        item_ = self.trainDF[item_key].unique()
        item2idx = pd.Series(data=np.arange(len(item_)), index=item_)
        self.itemmap = pd.DataFrame({item_key: item_, 'item2idx':item2idx[item_].values})

        self.trainDF = pd.merge(self.trainDF, self.itemmap, on=item_key, how='inner')
        self.trainDF.sort_values([session_key, time_key], inplace=True)

        #get the cumulative size of each session
        self.offsets = np.zeros(self.trainDF[session_key].nunique()+1, dtype=np.int32)  
        self.offsets[1:] = self.trainDF.groupby(session_key).size().cumsum()

        #sort sessions by start time
        sessions_start_time = self.trainDF.groupby(session_key)[time_key].min().values
        self.session_sorted = np.argsort(sessions_start_time)

    def next_batch(self):
        iterator = np.arange(self.batch_size)
        next_available = self.batch_size - 1 
        start_idx = self.offsets[self.session_sorted[iterator]]
        end_idx = self.offsets[self.session_sorted[iterator]+1]

        terminated = []
        done = False 
        while not done:
            interval = (end_idx - start_idx).min()
            target = self.trainDF.item2idx.values[start_idx]
            for idx in range(interval - 1):
                input = target
                target = self.trainDF.item2idx.values[start_idx+idx+1]
                inputTensor = torch.LongTensor(input)
                targetTensor = torch.LongTensor(target)
                yield inputTensor, targetTensor, terminated

            start_idx = start_idx + (interval-1)
            terminated = np.arange(self.batch_size)[(end_idx - start_idx) <= 1]
            for s in terminated:
                next_available += 1
                if next_available >= len(self.offsets) - 1:
                    done = True
                    break
                iterator[s] =  next_available
                start_idx[s] = self.offsets[self.session_sorted[next_available]]
                end_idx[s] = self.offsets[self.session_sorted[next_available]+1]   


class SessionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, dropout, batch_size):
        super(SessionModel, self).__init__()
        self.device = torch.device('cuda')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.onehot_input = torch.FloatTensor(self.batch_size, self.input_size)
        self.onehot_input = self.onehot_input.to(self.device)
        self.activation = nn.ReLU()
        self.hidden2output = nn.Linear(self.hidden_size, self.output_size)
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.layers, dropout=self.dropout) 
        self = self.to(self.device)

    def encode(self, input):
        self.onehot_input.zero_()
        index = input.view(-1, 1)
        return self.onehot_input.scatter_(1, index, 1) 

    def forward(self, input, hidden):
        tensor_input  = self.encode(input)
        tensor_input = tensor_input.unsqueeze(0)
        output, hidden = self.gru(tensor_input, hidden)
        output = output.view(-1, output.size(-1))
        logit = self.activation(self.hidden2output(output))  
        return logit, hidden

    def init_hidden(self):
        h = torch.zeros(self.layers, self.batch_size, self.hidden_size).to(self.device)
        return h
   
class TOP1Loss(nn.Module):

    def __init__(self):
        super(TOP1Loss, self).__init__()

    def forward(self, logit):
        diff = -(logit.diag().view(-1,1).expand_as(logit) - logit)
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(logit ** 2).mean()
        return loss

def resetHidden(hidden, terminated):
    if len(terminated) != 0:
        hidden[:, terminated, :] = 0
    return hidden

def Eval(model, eval_generator, losser):
    print("start evaluation")
    model.eval()
    losses = []
    recalls = []
    mrrs = []
    topk = 20
    def cal_recall(indices, targets):
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0:
            return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall

    def cal_mrr(indices, targets):
        tmp = targets.view(-1,1)
        targets = tmp.expand_as(indices)
        hits = (targets == indices).nonzero()
        ranks = hits[:, -1] + 1
        ranks = ranks.float()
        rranks = torch.reciprocal(ranks)
        mrr = torch.sum(rranks).data.item() / targets.size(0)
        return mrr

    with torch.no_grad():
        hidden = model.init_hidden()
        for input_tensor, target_tensor, terminated in eval_generator:
            input_tensor = input_tensor.to(model.device) 
            target_tensor = target_tensor.to(model.device)
            logit, hidden = model(input_tensor, hidden)
            logit_sampled = logit[:, target_tensor.view(-1)]
            loss = losser(logit_sampled) 
            _, indices = torch.topk(logit, topk, -1)
            recall = cal_recall(indices, target_tensor)
            mrr = cal_mrr(indices, target_tensor)  
            losses.append(loss.item())
            recalls.append(recall)
            mrrs.append(mrr)
    mean_losses = np.mean(losses)
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    print('mean_losses', mean_losses)            
    print('mean_recall', mean_recall)            
    print('mean_mrr', mean_mrr) 
           
if __name__ == '__main__':

    batch_size = 5
    hidden_size = 100
    layers = 3 
    dropout = 0.5
    lr = 0.01
    weight_decay = 0
    momentum = 0
    eps = 1e-6
    epochs = 10 
    device = torch.device('cuda')
    load_model = True 

    ds = dataset(batch_size, "../../data/session-model/train.txt")
    eval_ds = dataset(batch_size, "../../data/session-model/validation.txt")
    input_size = len(ds.itemmap['ItemID'].unique())
    if load_model:
        print("loading model")
        model = torch.load("model.pt")
    else:
        model = SessionModel(input_size,hidden_size,input_size, layers, dropout, batch_size)
    losser = TOP1Loss()
  
    #initial model
    for p in model.parameters():
        p.data.uniform_(-1, 1)
    
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
   
    #training
    for ep in range(epochs):
        print('Start Epoch #', ep)
        model.train()
        losses = []
        hidden = model.init_hidden()
        generator = ds.next_batch()
        for input_tensor, target_tensor, terminated in generator:
            input_tensor = input_tensor.to(device) 
            target_tensor = target_tensor.to(device) 
            optimizer.zero_grad()
            hidden = resetHidden(hidden, terminated).detach()
            logit, hidden = model(input_tensor, hidden)
            logit_sampled = logit[:, target_tensor.view(-1)]
            loss = losser(logit_sampled)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        mean = np.mean(losses)
        print('training data mean: ', mean) 
        eval_generator = eval_ds.next_batch()
        Eval(model, eval_generator, losser)
        if ep == epochs -1:   
            print("saving model") 
            torch.save(model, "model.pt")
