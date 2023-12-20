import torch
import datetime
import numpy as np

device = "cuda"
#device = "cpu"

class LSTM(torch.nn.Module):

    def __init__(self, tensor_data, hidden_1, hidden_2, hidden_3=0):
        #data format: time, day, user_id, problem_id, {network_value}, correct_and_continued
        super(LSTM, self).__init__()
        self.user_labels = tensor_data['users']
        self.users = self.user_labels.shape[0]
        self.problem_labels = tensor_data['problems']
        self.problems = self.problem_labels.shape[0]
        self.split_users()
        self.networks = tensor_data['networks']
        self.tensor = torch.zeros(0,self.networks+4,device=device)
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.hidden_3 = hidden_3
        self.epsilon = 0.1
        self.delta = 1
        self.day = 0
        self.days = tensor_data['days']
        self.h = torch.zeros((self.users, self.hidden_1, self.days+1)\
            ,device=device,dtype=torch.float32)
        #self.h.retain_grad = True
        self.C = torch.zeros((self.users, self.hidden_2, self.days+1)\
            ,device=device,dtype=torch.float32)
        #self.C.retain_grad = True
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.user_embeddings = torch.nn.Embedding(self.users, self.hidden_1\
            , device=device, dtype=torch.float32)
        self.problem_embeddings = torch.nn.Embedding(\
            self.problems*self.networks, self.hidden_1\
            , device=device, dtype=torch.float32)
        if self.hidden_3>0:
            self.topic_embeddings = torch.nn.Embedding(self.hidden_3\
                , self.hidden_1, device=device, dtype=torch.float32)
            self.problem_embeddings.weight.data \
                /= 10
            self.topic_embeddings.weight.data \
                = torch.zeros_like(self.topic_embeddings.weight.data)
            self.topic_embeddings.weight.data[:,0]\
                = torch.cos(2*3.1415*torch.arange(self.hidden_3)/self.hidden_3)
            self.topic_embeddings.weight.data[:,1]\
                = torch.sin(2*3.1415*torch.arange(self.hidden_3)/self.hidden_3)
        self.WiA = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , bias=False, device=device, dtype=torch.float32)
        self.UiA = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , device=device, dtype=torch.float32)
        self.WcA = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , bias=False, device=device, dtype=torch.float32)
        self.UcA = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , device=device, dtype=torch.float32)
        self.Wi = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , bias=False, device=device, dtype=torch.float32)
        self.Ui = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , device=device, dtype=torch.float32)
        self.Wf = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , bias=False, device=device, dtype=torch.float32)
        self.Uf = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , device=device, dtype=torch.float32)
        self.Wo = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , bias=False, device=device, dtype=torch.float32)
        self.Uo = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , device=device, dtype=torch.float32)
        self.Wc = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , bias=False, device=device, dtype=torch.float32)
        self.Uc = torch.nn.Linear(self.hidden_1, self.hidden_2\
            , device=device, dtype=torch.float32)
        self.Wr = torch.nn.Linear(self.hidden_2, self.problems\
            , device=device, dtype=torch.float32)
        self.loss_function = torch.nn.MSELoss()

    def reset_tensor(self):
        self.tensor = torch.zeros(0,self.networks+4,device=device)

    def split_users(self):
        torch.manual_seed(42)
        rand_users = torch.randperm(self.users)
        self.training_users = self.user_labels[rand_users[:int(self.users*8/10)]]
        self.validation_users = self.user_labels[rand_users[int(self.users*0/10)\
            :int(self.users*0/10)]]
        self.test_users = self.user_labels[rand_users[int(self.users*8/10):]]

    def get_mini_batch(self,type,size=20):
        torch.manual_seed(42)
        if type=="train":
            users = self.training_users[torch.randperm(self.training_users.shape[0])][:size]
            self.training_users = self.training_users[~torch.isin(self.training_users,users)]
        if type=="test":
            users = self.test_users[torch.randperm(self.test_users.shape[0])][:size]
            self.test_users = self.test_users[~torch.isin(self.test_users,users)]
        if type=="validation":
            users = self.validation_users[torch.randperm(self.validation_users.shape[0])][:size]
            self.validation_users = self.validation_users[~torch.isin(self.validation_users,users)]
        return users

    def get_users_indices(self,data):
        return torch.cat((data,self.user_labels))\
            .unique(return_inverse=True)[1][:data.shape[0]]

    def get_problems_indices(self,data):
        return torch.cat((data,self.problem_labels))\
            .unique(return_inverse=True)[1][:data.shape[0]]

    def get_problem_topics(self,data):
        problem_embeddings = self.problem_embeddings(\
            self.get_problems_indices(data)).repeat_interleave(\
            self.hidden_3,dim=0)
        topics_embeddings = self.topic_embeddings(torch.arange(self.hidden_3\
            ,device=device)).repeat((data.shape[0],1))
        chi_sqr = ((problem_embeddings-topics_embeddings)**2).sum(dim=1)\
            .reshape(data.shape[0],self.hidden_3)
        return chi_sqr.min(dim=1)[1]

    def alphas(self,data):
        #data format: time, day, user_id, problem_id, network, value, correct_and_continued        #returns tensor, rows are users, columns are days
        #reshape tensor
        problems, problem_indices = data[:,3].unique(return_inverse=True)
        max_day = int(torch.max(data[:,1]))
        users, user_indices = data[:,2].unique(return_inverse=True)
        #creating event tensor
        event_tensor = torch.zeros(users.shape[0],problems.shape[0],device=device)
        day_indices = data[:,1]
        event_tensor[user_indices,problem_indices] += 1
        #creating historic event tensor
        old_data = self.tensor[torch.isin(self.tensor[:,2],users)\
            *torch.isin(self.tensor[:,3],problems)]
        historical_event_tensor = torch.zeros((max_day+1)*users.shape[0]\
            ,problems.shape[0],device=device)
        users_indices = old_data[:,2].unique(return_inverse=True)[1]
        day_indices = old_data[:,1]
        problem_indices = old_data[:,3].unique(return_inverse=True)[1]
        historical_event_indices = (users_indices*(max_day+1)+day_indices).int()
        historical_event_tensor[historical_event_indices,problem_indices.int()]=1
        #doting historic with last day
        all_historical_indices \
            = torch.arange(historical_event_tensor.shape[0],device=device).int()
        all_users_indices = all_historical_indices//(max_day+1)
        historical_event_tensor[all_historical_indices,:] \
            *= event_tensor[all_users_indices,:]
        #summing
        alphas = torch.sum(historical_event_tensor,dim=1)
        alphas /= torch.sqrt(torch.sum(event_tensor[all_users_indices,:],dim=1))
        alphas/= torch.sqrt(torch.sum(\
            historical_event_tensor[all_historical_indices,:],dim=1))
        #reshape
        alphas = alphas.reshape(users.shape[0],max_day+1)
        alphas = torch.softmax(alphas,dim=1)
        return alphas

    def alphas_topics(self,data):
        #data format: time, day, user_id, problem_id, network, value, correct_and_continued
        #embed new problems
        new_problems_indices = self.get_problems_indices(data[:,3]).unique()
        unknown_problems \
            = torch.arange(self.problems,device=device)\
            [self.problem_embeddings.weight.data[:,0]== 0]
        new_problems \
            = new_problems_indices\
            [torch.isin(new_problems_indices,unknown_problems)]
        days = data[:,1].min()*torch.ones_like(new_problems,device=device)
        self.problem_embeddings.weight.data[new_problems,0:2] \
            = torch.stack((torch.cos(2*3.1415*days/self.days)\
            ,torch.sin(2*3.1415*days/self.days)),dim=1)
        #returns tensor, rows are users, columns are days ago
        topics_data = torch.cat((data[:,:3]\
            ,self.get_problem_topics(data[:,3].int()).reshape((-1,1))\
            ,data[:,4:]),dim=1)
        #problems, problem_indices = data[:,3].unique(return_inverse=True)
        topics, topics_indices = topics_data[:,3].unique(return_inverse=True)
        max_day = int(torch.max(topics_data[:,1]))
        users, users_indices = topics_data[:,2].unique(return_inverse=True)
        #creating event tensor
        event_tensor = torch.zeros(users.shape[0],topics.shape[0],device=device)
        day_indices = data[:,1]
        event_tensor[users_indices,topics_indices] += 1
        event_tensor /= self.networks
        #creating topic data
        historical_data = self.tensor[torch.isin(self.tensor[:,2],users)\
            *torch.isin(self.get_problem_topics(self.tensor[:,3]),topics)]
        topics_historical_data = torch.cat((historical_data[:,:3]\
            ,self.get_problem_topics(historical_data[:,3].int())\
            .reshape((-1,1)),historical_data[:,4:]),dim=1)
        #creating historic event tensor
        historical_event_tensor \
            = torch.zeros((max_day+1)*users.shape[0],topics.shape[0],device=device)
        users_indices \
            = topics_historical_data[:,2].unique(return_inverse=True)[1]
        day_indices = topics_historical_data[:,1]
        topics_indices \
            = (topics_historical_data[:,3].unique(return_inverse=True)[1]).int()
        historical_event_indices = (users_indices*(max_day+1)+day_indices).int()
        historical_event_tensor[historical_event_indices,topics_indices]\
            += 1
        historical_event_tensor /= self.networks
        #doting historic with last day
        all_historical_indices \
            = torch.arange(historical_event_tensor.shape[0],device=device).int()
        all_users_indices = all_historical_indices//(max_day+1)
        historical_event_tensor[all_historical_indices,:] \
            *= event_tensor[all_users_indices,:]
        #summing
        alphas = torch.sum(historical_event_tensor,dim=1)
        alphas /= torch.sqrt(torch.sum(event_tensor[all_users_indices,:],dim=1)\
            +self.epsilon)
        alphas/= torch.sqrt(torch.sum(\
            historical_event_tensor[all_historical_indices,:],dim=1)+self.epsilon)
        #reshape
        alphas = alphas.reshape(users.shape[0],max_day+1)
        alphas = torch.softmax(alphas,dim=1)
        #alphas = torch.nn.functional.normalize(alphas)
        return alphas

    def hA_tm1(self, data):
        #data format: time, day, user_id, problem_id, network, value, correct_and_continued
        users = torch.unique(data[:,2]).to(device=device,dtype=torch.int64)
        users_h_index = self.get_users_indices(users)
        days = torch.max(data[:,1]).to(device=device,dtype=torch.int64)-1
        alphas = self.alphas(data)
        #h format: = users, hidden_1, days
        h = self.h[users_h_index,:,:days+1]
        user_indices = torch.arange(users.shape[0]*days\
            ,device=device,dtype=torch.int64)//days
        day_indices = torch.arange(users.shape[0]*days\
            ,device=device,dtype=torch.int64)%users.shape[0]
        h[user_indices, :, day_indices] \
            *= alphas[user_indices,day_indices].reshape((-1,1)).repeat(1,self.hidden_1)
        #h[user_indices, :, day_indices] \
        #    *= alphas[user_indices,day_indices].reshape((-1,1)).repeat(1,self.hidden_1)
        hAtm1 = torch.sum(h,dim=2)
        return hAtm1

    def hA_tm1_topics(self, data):
        #data format: time, day, user_id, problem_id, network, value, correct_and_continued
        users = torch.unique(data[:,2]).to(device=device,dtype=torch.int64)
        users_h_index = self.get_users_indices(users)
        days = torch.max(data[:,1]).to(device=device,dtype=torch.int64)-1
        alphas = self.alphas_topics(data)
        #h format: = users, hidden_1, days
        h = self.h[users_h_index,:,:days+1]
        user_indices = torch.arange(users.shape[0]*days\
            ,device=device,dtype=torch.int64)//days
        day_indices = torch.arange(users.shape[0]*days\
            ,device=device,dtype=torch.int64)%users.shape[0]
        h[user_indices, :, day_indices] \
            *= alphas[user_indices,day_indices].reshape((-1,1)).repeat(1,self.hidden_1)
        hAtm1 = torch.sum(h,dim=2)
        return hAtm1

    def data_reshaper(self,data):
        #data coming format: time, day, user_id, problem_id, {network_value}, correct_and_continued
        #data going format: time, day, user_id, problem_id, network, value, correct_and_continued
        h, w = data.shape
        return_data = torch.zeros(h*self.networks, 7, dtype=torch.int32, device=device)
        #copy first four columns
        indices = torch.arange(0,self.networks*h,1,device=device)%h
        return_data[:,:4] = data[indices,:4]
        #copy the last column
        return_data[:,w-1] = data[indices,w-1]
        #setting network data
        networks = torch.arange(0,self.networks*h,1,device=device)//h
        return_data[:,4] = networks
        #return_data[:h,5] = torch.ones(h)
        values = torch.ones(h,self.networks, device=device)
        values[:,1:] = data[:,4:w-1]
        return_data[:,5] = values[torch.arange(h*self.networks)%h,\
            torch.arange(h*self.networks)//h]
        return return_data

    def interaction_vector(self,data):
        #getting info
        users, user_indices = torch.unique(data[:,2].int(),return_inverse=True)
        problems, problem_indices = torch.unique(data[:,3].int()\
            ,return_inverse=True)
        network_indices = data[:,4].int()
        #creating tensor of embeddings
        #
        #embedding tensor shape, tensor.shape = [users*(problems*networks+1)
        #user_1 problem_1_embedding network_1
        #user_1 problem_2_embedding network_1
        # ...
        #user_1 problem_1_embedding network_2
        # ...
        #user_1 tensor.ones(self.hidden_1)
        #user_2 problem_1_embedding network_1
        # ...
        embeddings = torch.zeros(users.shape[0]*(problems.shape[0]*self.networks+1)\
            ,self.hidden_1,device=device,dtype=torch.float32)
        #filling embeddings tensor
        embeddings[user_indices*(problems.shape[0]*self.networks+1)+problem_indices,:]\
            = self.problem_embeddings(\
            self.get_problems_indices(data[:,3]).int()\
            +network_indices*self.problems)
        ones_indices = torch.arange(users.shape[0])\
            *((problems.shape[0]+1)*self.networks)
        embeddings[ones_indices,:] = torch.ones(self.hidden_1,device=device)
        #permuting and dotting
        combination_sizes = self.networks*problems.shape[0]+1
        combinations = torch.combinations(torch.arange(combination_sizes))
        repeated_users = torch.arange(users.shape[0]*combinations.shape[0])\
            //combinations.shape[0]
        combinations = combinations.repeat((users.shape[0],1))
        interaction_vector \
            = torch.zeros(users.shape[0],self.hidden_1,device=device)
        first_indices = repeated_users*combination_sizes + combinations[:,0]
        second_indices = repeated_users*combination_sizes + combinations[:,1]
        interaction_vector[repeated_users,:] += embeddings[first_indices]\
            *embeddings[second_indices]
        return interaction_vector

    def forward_no_topics(self,data):
        self.tensor= torch.unique(torch.cat((self.tensor,data)),dim=0)
        data = self.data_reshaper(data)
        #data format: time, day, user_id, problem_id, network, value, correct_and_continued
        users, counts = torch.unique(data[:,2], return_counts=True)
        h_C_indices = self.get_users_indices(users)
        day = int(data[:,1].max())
        i = self.interaction_vector(data)
        hA_tm1 = self.hA_tm1(data)
        iA_t = self.sigmoid(self.WiA(i)+self.UiA(hA_tm1))
        cA_t = torch.tanh(self.WcA(i)+self.UcA(hA_tm1))
        cA_t = iA_t*cA_t
        h_tm1 = self.h[h_C_indices,:,day-1]
        it = (1-np.exp(-self.delta))*self.sigmoid(self.Wi(i)+self.Ui(h_tm1))
        ft = (np.exp(-self.delta))*self.sigmoid(self.Wf(i)+self.Uf(h_tm1))
        ot = self.sigmoid(self.Wo(i)+self.Uo(h_tm1))
        ct = torch.tanh(self.Wc(i)+self.Uc(h_tm1))
        C_tm1 = self.C[h_C_indices,:,day]
        Ct = ft*C_tm1+it*ct+iA_t*cA_t
        self.C[h_C_indices,:,day+1] = Ct
        ht = ot*torch.tanh(Ct)
        self.h[h_C_indices,:,day+1] = ht
        output = torch.tanh(self.Wr(ht))
        user_indices = (data[:,[2]*users.shape[0]]\
            == users.repeat(data.shape[0],1)).nonzero()[:,1]
        problem_indices = self.get_problems_indices(data[:,3])
        prediction = output[user_indices,problem_indices]
        prediction = (prediction+1)/2
        return prediction

    def forward(self,data):
        self.tensor= torch.unique(torch.cat((self.tensor,data)),dim=0)
        data = self.data_reshaper(data)
        #data format: time, day, user_id, problem_id, network, value, correct_and_continued
        users, counts = torch.unique(data[:,2], return_counts=True)
        h_C_indices = self.get_users_indices(users)
        day = int(data[:,1].max())
        i = self.interaction_vector(data)
        hA_tm1 = self.hA_tm1_topics(data)
        iA_t = self.sigmoid(self.WiA(i)+self.UiA(hA_tm1))
        cA_t = torch.tanh(self.WcA(i)+self.UcA(hA_tm1))
        cA_t = iA_t*cA_t
        h_tm1 = self.h[h_C_indices,:,day-1]
        it = (1-np.exp(-self.delta))*self.sigmoid(self.Wi(i)+self.Ui(h_tm1))
        ft = (np.exp(-self.delta))*self.sigmoid(self.Wf(i)+self.Uf(h_tm1))
        ot = self.sigmoid(self.Wo(i)+self.Uo(h_tm1))
        ct = torch.tanh(self.Wc(i)+self.Uc(h_tm1))
        C_tm1 = self.C[h_C_indices,:,day]
        Ct = ft*C_tm1+it*ct+iA_t*cA_t
        self.C[h_C_indices,:,day+1] = Ct
        ht = ot*torch.tanh(Ct)
        self.h[h_C_indices,:,day+1] = ht
        output = torch.tanh(self.Wr(ht))
        user_indices = (data[:,[2]*users.shape[0]]\
            == users.repeat(data.shape[0],1)).nonzero()[:,1]
        problem_indices = self.get_problems_indices(data[:,3])
        prediction = output[user_indices,problem_indices]
        prediction = (prediction+1)/2
        return prediction

    def loss(self,output,target):
        return self.loss_function(output,target)