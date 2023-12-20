import torch
import os

device = "cuda"
#device = "cpu"

def load_tensor(directory):
    #data format returned: 'timestamp_TW', 'uuid', 'ucid', 'upid', 'problem_number',
    #   'exercise_problem_repeat_session', 'is_correct', 'total_sec_taken',
    #   'total_attempt_cnt', 'used_hint_cnt', 'is_hint_used', 'level'
    tensors = os.listdir(directory)
    tensor = torch.load(directory+tensors[0])
    rows, columns = tensor.shape
    for tensorFileName in tensors[1:]:
        tensorAddition = torch.load(directory+tensorFileName)
        tensor = torch.cat((tensor,tensorAddition),0)
    return tensor

def get_tensor(directory,base_time=1533060000,students_min=100,problems_min=5):
    #data format returned: time, day, user_id, problem_id, {network_values}, correct_and_continued
    tensor = load_tensor(directory).to(device=device)
    #sorting the tensor by user and then time
    time = tensor[:,0].min()
    tensor[:,0] = tensor[:,0]-time
    max_time = tensor[:,0].max()
    sort_column = torch.zeros_like(tensor[:,0],dtype=torch.float64)
    sort_column += tensor[:,1]
    sort_column *= max_time 
    sort_column += tensor[:,0] 
    tensor = tensor[sort_column.sort(dim=0).indices]
    del sort_column
    #making column for correct answer and continuation
    still_using = (tensor[:,0][1:]-tensor[:,0][:-1]<1800)
    same_user = (tensor[:,1][1:] == tensor[:,1][:-1])
    continued = still_using*same_user
    continued = continued.unsqueeze(0)
    continued = torch.cat((continued.mT,torch.tensor([[0]]\
        ,device=device).mT)).mT
    correct_and_continued = continued*tensor[:,6:7].mT
    #getting the days column
    days = tensor[:,:1]//86400
    #combining columns
    tensor = torch.cat((tensor[:,:1].mT,days.mT,tensor[:,[1,3]].mT\
        , correct_and_continued)).mT
    #removing data points
    tensor = tensor_filterer(tensor,students_min,problems_min)
    tensor.requires_grad=False
    return tensor

def tensor_filterer(tensor,students_min,problems_min):
    student_min = 1
    problem_min = 1
    while student_min<students_min or problem_min<problems_min:
        student_id, student_count = torch.unique(tensor[:,2],return_counts=True)
        student_indices_count = torch.zeros((int(student_id.max().item())+1)\
            ,device=device,dtype=torch.int64)
        student_indices_count[student_id.int()] = student_count
        tensor = tensor[(student_indices_count[tensor[:,2].int()]>(students_min-1))]
        problem_id, problem_count = torch.unique(tensor[:,3],return_counts=True)
        problem_indices_count = torch.zeros((int(problem_id.max().item())+1)\
            ,device=device,dtype=torch.int64)
        problem_indices_count[problem_id.int()] = problem_count
        tensor = tensor[(problem_indices_count[tensor[:,3].int()]>(problems_min-1))]
        student_min = student_count.min().item()
        problem_min = problem_count.min().item()
    return tensor

def time_network_get_tensor(directory,base_time=1533060000,students_min=100,problems_min=5):
    #data format: time, day, user_id, problem_id, correct_and_continued, hour_of_day
    tensor = get_tensor(directory,base_time,students_min,problems_min)
    hour_of_day = (((tensor[:,:1]-base_time)%(86400))//(3600))/240
    tensor = torch.cat((tensor[:,:-1].mT,hour_of_day.mT,tensor[:,-1:].mT)).mT
    return tensor

def problem_number_get_tensor(directory,base_time=1533060000,students_min=100,problems_min=5):
    #data format returned: time, day, user_id, problem_id, {network_values}, correct_and_continued
    tensor = load_tensor(directory).to(device=device)
    #sorting the tensor by user and then time
    time = tensor[:,0].min()
    tensor[:,0] = tensor[:,0]-time
    max_time = tensor[:,0].max()
    sort_column = torch.zeros_like(tensor[:,0],dtype=torch.float64)
    sort_column += tensor[:,1]
    sort_column *= max_time 
    sort_column += tensor[:,0] 
    tensor = tensor[sort_column.sort(dim=0).indices]
    del sort_column
    #making column for correct answer and continuation
    same_user = torch.ones_like(tensor[:,6:7],dtype=torch.int32,device=device)
    same_user[:-1,:] = tensor[:-1,1:2]-tensor[1:,1:2]+1
    still_using = torch.ones_like(tensor[:,6:7],dtype=torch.int32,device=device)
    still_using[1:,:] = torch.heaviside((tensor[1:,0:1]-tensor[:-1,0:1])+600\
        ,torch.ones_like(still_using[1:,:]))
    correct_and_continued = tensor[:,6:7]*same_user*still_using
    #getting the days column
    days = tensor[:,:1]//86400
    #combining columns
    tensor = torch.cat((tensor[:,:1].mT,days.mT,tensor[:,[1,3,4]].mT\
        , correct_and_continued.mT)).mT
    #removing data points
    tensor = tensor_filterer(tensor,students_min,problems_min)
    tensor.requires_grad=False
    return tensor

def problem_number_time_get_tensor(directory,base_time=1533060000,students_min=100,problems_min=5):
    #data format returned: time, day, user_id, problem_id, {network_values}, correct_and_continued
    tensor = load_tensor(directory).to(device=device)
    #sorting the tensor by user and then time
    time = tensor[:,0].min()
    tensor[:,0] = tensor[:,0]-time
    max_time = tensor[:,0].max()
    sort_column = torch.zeros_like(tensor[:,0],dtype=torch.float64)
    sort_column += tensor[:,1]
    sort_column *= max_time 
    sort_column += tensor[:,0] 
    tensor = tensor[sort_column.sort(dim=0).indices]
    del sort_column
    #making column for correct answer and continuation
    same_user = torch.ones_like(tensor[:,6:7],dtype=torch.int32,device=device)
    same_user[:-1,:] = tensor[:-1,1:2]-tensor[1:,1:2]+1
    still_using = torch.ones_like(tensor[:,6:7],dtype=torch.int32,device=device)
    still_using[1:,:] = torch.heaviside((tensor[1:,0:1]-tensor[:-1,0:1])+600\
        ,torch.ones_like(still_using[1:,:]))
    correct_and_continued = tensor[:,6:7]*same_user*still_using
    #getting the days column
    days = tensor[:,:1]//86400
    #getting the time of day
    hour_of_day = (((tensor[:,:1]-base_time)%(86400))//(3600))/240
    #combining columns
    tensor = torch.cat((tensor[:,:1].mT,days.mT,tensor[:,[1,3,4]].mT\
        ,hour_of_day.mT, correct_and_continued.mT)).mT
    #removing data points
    tensor = tensor_filterer(tensor,students_min,problems_min)
    tensor.requires_grad=False
    return tensor

def city_get_tensor(directory,base_time=1533060000,students_min=100,problems_min=5):
    #data format returned: time, day, user_id, problem_id, {network_values}, correct_and_continued
    tensor = load_tensor(directory+'data/').to(device=device)
    #sorting the tensor by user and then time
    time = tensor[:,0].min()
    tensor[:,0] = tensor[:,0]-time
    max_time = tensor[:,0].max()
    sort_column = torch.zeros_like(tensor[:,0],dtype=torch.float64)
    sort_column += tensor[:,1]
    sort_column *= max_time 
    sort_column += tensor[:,0] 
    tensor = tensor[sort_column.sort(dim=0).indices]
    del sort_column
    #making column for correct answer and continuation
    same_user = torch.ones_like(tensor[:,6:7],dtype=torch.int32,device=device)
    same_user[:-1,:] = tensor[:-1,1:2]-tensor[1:,1:2]+1
    still_using = torch.ones_like(tensor[:,6:7],dtype=torch.int32,device=device)
    still_using[1:,:] = torch.heaviside((tensor[1:,0:1]-tensor[:-1,0:1])+600\
        ,torch.ones_like(still_using[1:,:]))
    correct_and_continued = tensor[:,6:7]*same_user*still_using
    #getting the days column
    days = tensor[:,:1]//86400
    #getting the city
    users = load_tensor(directory+'users/')
    cities = users[tensor[:,1],5]
    cities = cities/cities.max()/10
    #combining columns
    tensor = torch.cat((tensor[:,:1].mT,days.mT,tensor[:,[1,3,4]].mT\
        ,cities.reshape(1,-1), correct_and_continued.mT)).mT
    #removing data points
    tensor = tensor_filterer(tensor,students_min,problems_min)
    tensor.requires_grad=False
    return tensor

def gender_get_tensor(directory,base_time=1533060000,students_min=100,problems_min=5):
    #data format returned: time, day, user_id, problem_id, {network_values}, correct_and_continued
    tensor = load_tensor(directory+'data/').to(device=device)
    #sorting the tensor by user and then time
    time = tensor[:,0].min()
    tensor[:,0] = tensor[:,0]-time
    max_time = tensor[:,0].max()
    sort_column = torch.zeros_like(tensor[:,0],dtype=torch.float64)
    sort_column += tensor[:,1]
    sort_column *= max_time 
    sort_column += tensor[:,0] 
    tensor = tensor[sort_column.sort(dim=0).indices]
    del sort_column
    #making column for correct answer and continuation
    same_user = torch.ones_like(tensor[:,6:7],dtype=torch.int32,device=device)
    same_user[:-1,:] = tensor[:-1,1:2]-tensor[1:,1:2]+1
    still_using = torch.ones_like(tensor[:,6:7],dtype=torch.int32,device=device)
    still_using[1:,:] = torch.heaviside((tensor[1:,0:1]-tensor[:-1,0:1])+600\
        ,torch.ones_like(still_using[1:,:]))
    correct_and_continued = tensor[:,6:7]*same_user*still_using
    #getting the days column
    days = tensor[:,:1]//86400
    #getting the city
    users = load_tensor(directory+'users/')
    genders = users[tensor[:,1],1]
    genders = genders/genders.max()/10
    #combining columns
    tensor = torch.cat((tensor[:,:1].mT,days.mT,tensor[:,[1,3,4]].mT\
        ,genders.reshape(1,-1), correct_and_continued.mT)).mT
    #removing data points
    tensor = tensor_filterer(tensor,students_min,problems_min)
    tensor.requires_grad=False
    return tensor