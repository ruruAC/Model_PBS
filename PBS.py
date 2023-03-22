import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib  
matplotlib.use('Agg') 
import pandas as pd
"""
    tp=0: 燃油+两驱
    tp=1: 燃油+四驱
    tp=2: 混合+两驱
    tp=3: 混合+四驱
"""

#到返回，到车，到中间
in_transfer_time_dic = {
    1:[0,9,18],  # 0->1
    2:[3,15,24], # b->1
    3:[0,6,12], # 0->2
    4:[3,12,18], # b->2
    5:[0,3,6],  # 0->3
    6:[3,9,12],  # b->3
    7:[0,0,0],  # 0->4
    8:[3,6,6],  # b->4
    9:[0,6,12],  # 0->5
    10:[3,6,12],  # b->5
    11:[0,9,18],  # 0->6
    12:[3,9,18],  # b->6
}
#到车，到返回，到中间
out_transfer_time_dic = {
    1:[9,0,18],  
    2:[9,21,24], 
    3:[6,0,12], 
    4:[6,15,18], 
    5:[3,0,6],  
    6:[3,9,12],  
    7:[0,0,0],  
    8:[0,3,6],  
    9:[6,0,12],  
    10:[6,9,12],  
    11:[9,0,18],  
    12:[9,15,18], 
}

class workshop_env():
    def __init__(self):
        # ini_setting
        self.forward_lines_num = 6
        self.car_bit = 4 # type bits
        self.time_bit = 1
        self.transfer_work_bit = 12 + 1 # 策略 
        self.transfer_time_bit = 1
        self.single_state_space = 338
        self.state_space = 338 * 1
        self.action_space = 13
        self.time_norm = 9
        # path = './attach1.csv'
        # self.read_data_from_excel(path)
        self.input = np.zeros((200, 4), dtype=np.float32)
        for i in range(200):
            k = np.random.randint(3)
            self.input[i, k+1] = 1.
            self.input[i, 0] = i+1

        # our_input = np.array(our_input)

        df1=pd.DataFrame(self.input)
        df1.to_csv("./results/our_input.csv",index=False)

    def read_data_from_excel(self,path):

        df_raw = pd.read_csv(path, parse_dates=['number', 'car_type'])
        df_raw =pd.DataFrame(df_raw,dtype=np.float32)
        car_type = np.array(df_raw['car_type'], dtype=np.int64)
        self.input = np.zeros((len(car_type), self.car_bit), dtype=np.float32)
        self.input[:, 0] = np.array(df_raw['number'])
        for i in range(len(car_type)):
            self.input[i, car_type[i]+1] = 1.0

    def get_output_score(self,):
        first_car = np.where(self.output[0][1:] !=0 )[0]
        if first_car == 1 or first_car ==3:
            first_label = 4
        else:
            first_label = 2
        out = np.array(self.output, dtype=np.int)[:,1:]
        out = np.sum(out* np.array([[0, 1, 2]]), axis=1)

        k = None
        score1 = 100
        for i in range(len(out)):
            if out[i] == 2 or out[i] == 3:
                if k == None:
                    k = i
                else:
                    if i - k - 1 != 2:
                        score1 -= 1
                    k = i

        score2 = 100
        k = 0
        for i in range(1,len(out)):
            now_label = 4 if (out[i] == 1 or out[i] == 3) else 2
            pre_label = 4 if (out[i-1] == 1 or out[i-1] == 3) else 2
            if first_label == 4:
                if now_label == 4 and pre_label != 4:
                    num_4 = ((out[k:i]== 1) + (out[k:i]== 3)).sum()
                    num_2 = ((out[k:i]== 0) + (out[k:i]== 2)).sum()
                    assert(num_2+num_4 == i-k),('!!!')
                    if num_4 != num_2:
                        score2-=1
                    k = i
            else:
                if now_label == 2 and pre_label != 2:
                    num_4 = ((out[k:i]== 1) + (out[k:i]== 3)).sum()
                    num_2 = ((out[k:i]== 0) + (out[k:i]== 2)).sum()
                    assert(num_2+num_4 == i-k),('!!!')
                    if num_4 != num_2:
                        score2-=1
                    k = i

        score3 = 100 - self.num_back_usage

        score4 = 100 - 0.01 * (self.times - len(self.output) * 9 - 72)


        score = 0.4 * score1 + 0.3 * score2 + 0.2 * score3 + 0.1 * score4

        return score, score1, score2, score3, score4

    def get_state(self, num=4, out_done=False):
        x1 = self.forward_lines.copy()
        x1[:,:,-1] /= self.time_norm
        x1 = x1[:,:,1:].flatten()

        x2 = self.backward_lines.copy()
        x2[:,:,-1] /= self.time_norm
        x2 = x2[:,:,1:].flatten()

        x3 = self.in_transfer.copy()
        x3[-1] /= self.time_norm
        x3=np.append(x3[:self.transfer_work_bit],\
            x3[self.transfer_work_bit+1:],axis=0)

        x4 = self.out_transfer.copy()
        x4[-1] /= self.time_norm
        x4=np.append(x4[:self.transfer_work_bit],\
            x4[self.transfer_work_bit+1:],axis=0)

        # self.input_index=42
        if self.input_index + num < len(self.input):
            x5 = self.input[self.input_index:self.input_index+num,1:].copy()
        else:
            x5=np.zeros((num, self.car_bit-1))
            if self.input_index != len(self.input):
                left_num = len(self.input) - self.input_index
                x5[:left_num] = self.input[self.input_index:,1:].copy()
        x5=x5.flatten()

        if len(self.output) < num:
            x6 = np.zeros((num, self.car_bit-1))
            if len(self.output)!= 0:
                left_num = len(self.output)
                x6[:left_num] = np.array(self.output[::-1])[:,1:].copy()
        else:
            x6=np.array(self.output[::-1])[:num,1:].copy()
        x6 = x6.flatten()
        state = np.concatenate([x1,x2,x3,x4,x5,x6],axis=0)
        state_without_out = np.concatenate([x1,x2,x3,x4,x5],axis=0)
        available_action=self.getmask_q1()

        mean = state.mean()
        std = state.std()
        state = (state - mean) / std
        state = np.concatenate((state, self.pre_state))
        self.pre_state = state[:-self.single_state_space]
        if out_done:
            return state,available_action, int(state_without_out.sum()==2)
        else:
            return state,available_action

    def get_reward(self,action):

        # 优化1
        reward1 = 0.0
        find_ = False
        if self.is_out:
            car_type = np.where(self.output[-1][1:] !=0)[0]
            if car_type == 2 or car_type == 3:
                for k in range(len(self.output)-2,-1,-1):
                    car_type = np.where(self.output[k][1:] !=0)[0]
                    if car_type==2 or car_type == 3:
                        s=k
                        find_ = True
                        break
                if find_ and len(self.output)-1-1-s !=2:
                    reward1 = -1
                elif find_ and len(self.output)-1-1-s == 2:
                    reward1 = 4.5


        find_ = False
        reward2 = 0.0
        if self.is_out:
            if self.label is None:
                car_type = np.where(self.output[-1][1:] !=0)[0]
                if car_type == 1 or car_type ==3:
                    self.label=4
                else:
                    self.label=2
            else:
                car_type_now = np.where(self.output[-1][1:] !=0)[0]
                car_type_pre = np.where(self.output[-2][1:] !=0)[0]
                if car_type_now == 1 or car_type_now ==3:
                    now_label = 4
                else:
                    now_label = 2
                if car_type_pre == 1 or car_type_pre ==3:
                    pre_label = 4
                else:
                    pre_label = 2
                if now_label == self.label and pre_label != self.label:
                    for k in range(len(self.output)-2,-1,-1):
                        car_type = np.where(self.output[k][1:] !=0)[0]
                        if car_type==1 or car_type == 3:
                            running_label = 4
                        else:
                            running_label = 2
                        if running_label == now_label:
                            s=k
                            find_ = True
                            break
                    assert(find_ == True),("check reward2")
                    if find_:
                        block = np.array(self.output[s:-1])[:, 1:]
                        count4 = block[:, 1].sum()
                        count2 = block[:, 0].sum()+block[:, 2].sum()
                        if count4 == count2:
                            reward2 = 3
                        else:
                            reward2 = -0.1
                

        # 优化3
        reward3 = 0
        index = np.array([2,4,6,8,10,12], dtype=np.int)
        if self.out_transfer[-1] == 1 and self.out_transfer[index].sum() >= 1:
            reward3 = - 1


        reward4 = 0
        if action[1] == 0 or action[0] == 0:
            reward4 = - 0.5
        return (reward1+reward2+reward3+reward4)

        # 优化4
        # if reward1 

    def reset(self, save=False):
        self.cars_excel = None
        self.pre_state = np.zeros((self.state_space-self.single_state_space,\
            ), dtype=np.float32)
        self.input_index = 0
        self.output = []
        self.forward_lines = np.zeros((self.forward_lines_num, 10, \
            self.car_bit + self.time_bit), dtype=np.float32)
        self.backward_lines = np.zeros((1, 10, self.car_bit + \
            self.time_bit), dtype=np.float32)
        self.in_transfer = np.zeros((self.transfer_work_bit + \
            self.car_bit + self.transfer_time_bit), dtype=np.float32)
        self.out_transfer = np.zeros((self.transfer_work_bit + \
            self.car_bit + self.transfer_time_bit), dtype=np.float32)
        self.in_transfer[0] = 1
        self.out_transfer[0] = 1
        self.times = 0
        self.num_back_usage = 0
        self.is_out = False
        self.label = None
        state,available_action=self.get_state()
        if save:
            self.save_state()

        return state,available_action
        

    def input_pull(self):
        self.input_index += 1
        return self.input[self.input_index - 1]

    def output_push(self, out):
        self.output.append(out)
        self.is_out = True

    def car_encoding(self, tp):  
        """
            tp=0: 燃油+两驱
            tp=1: 燃油+四驱
            tp=2: 混合+两驱
            tp=3: 混合+四驱
        """
        assert (tp <= 3 and tp >= 0), ("check tp in car_encoding!")
        car_type = np.zeros((self.car_bit + self.time_bit,))
        car_type[tp] = 1.
        return car_type

    def getmask_q2(self):
        # 到达的时候车道无车 或者，有车 但是等待时间+到达时间 >=9
        #  ，且有一个空位（1，9），状态成立--基础
        #forward
        #如果返回道有车，必须去接车,空闲不成立,1，3，5，7，9，11
        # 都不成立，需要基础判断剩下的是否成立
        #如果没车，1-13都要判断（返回：3s: 10号有无车 or 9号>=6）
        # 放车基础判断
        #backward
        #如果有等待的车，到达的车道是确定的，两种动作（去返回或者送出）
        # ，返回需要基础判断，返回需要判断1号当前无车且到达时2号业务车
        #如果没有等待的车，判断每一个动作，到达车道(到达时间： 
        # 1号 有无车 or 2号 t+t到>=9) /返回车道(基础）有无车
        
        mask=np.zeros([2,13], dtype=np.int32)
         #forward 
        if self.in_transfer[0] == 1: #空闲  
            mask[0][0]=1
            if self.backward_lines[0,1, self.car_bit]+3>=9 or \
                self.backward_lines[0,0].sum() != 0: #如果返回道未来可以接， 返回道接车+送进哪个车道 
                for j in [2,4,6,8,10,12]: #先去返回道接车，然后送进哪个车道
                    time_notes = in_transfer_time_dic[j]
                    line = j // 2 - 1
                    if self.forward_lines[line, 0].sum() == 0:
                        mask[0][j]=1
                    elif self.forward_lines[line, 0, self.car_bit]+\
                        time_notes[1] >= 9:
                        for k in range(0,10):
                            if self.forward_lines[line, k].sum() == 0: 
                                mask[0][j]=1
                                break
            #判断能否把车送进车道
            if self.input_index<self.input.shape[0]:
                for j in [1,3,5,7,9,11]:  
                    time_notes = in_transfer_time_dic[j]
                    line = j // 2
                    if self.forward_lines[line, 0].sum() == 0:
                        mask[0][j]=1
                    elif self.forward_lines[line, 0, self.car_bit]+\
                        time_notes[1] >= 9  :
                        for k in range(0,10):
                            if self.forward_lines[line, k].sum() == 0: 
                                mask[0][j]=1
                                break
        else:
            pass
            # print("in 不空闲")
        #backward
        if self.out_transfer[0] == 1: #空闲的
            for j in [2,4,6,8,10,12]: #车道，先去车道接车再去返回
                time_notes = out_transfer_time_dic[j]
                line = j // 2 - 1
                #判断能否接到车
                if self.forward_lines[line, 9].sum() != 0:
                    mask[1][j]=1
                else:
                    if self.forward_lines[line, 8, self.car_bit]+\
                        time_notes[0] >= 9 and self.forward_lines[line, 8].sum()!=0:#接车必须有车
                        mask[1][j]=1
                if mask[1][j]:
                    fh=0
                    #判断能否放返回
                    if self.backward_lines[0,9].sum()==0:  #返回道空闲，判断能否接到车
                        fh=1
                    elif self.backward_lines[0, 9, -1]+time_notes[1]>=9:  #返回道此时不空闲，后续空闲
                        for k in range (0,10):
                            if self.backward_lines[0, k].sum() == 0: 
                                fh=1
                                break
                    mask[1][j]=fh

            for j in [1,3,5,7,9,11]: #直接送出去，车道有没有车
                time_notes = out_transfer_time_dic[j]
                line = j // 2 
                if self.forward_lines[line, 9].sum() != 0:
                    mask[1][j]=1
                else:
                    if self.forward_lines[line, 8, self.car_bit]+\
                        time_notes[0] >= 9 and self.forward_lines[line, 8].sum() != 0:
                        mask[1][j]=1
            mask[1][0]=1
        else:
            pass
            # print("out 不空闲")

        index = np.array([2,4,6,8,10,12], dtype=np.int)
        mask[1, index] =  0

        if mask[0].sum() <= 1:
            mask[0] = np.zeros(mask[0].shape, dtype=np.int32)
        if mask[1].sum() <= 1:
            mask[1] = np.zeros(mask[1].shape, dtype=np.int32)


        return mask

#到车，到返回，到中间
#到返回，到车，到中间
    def getmask_q1(self):
        # 到达的时候车道无车 或者，有车 但是等待时间+到达时间 >=9 ，
        # 且有一个空位（1，9），状态成立--基础
        #forward
        #如果返回道有车，必须去接车,空闲不成立,1，3，5，7，9，11都不成立，
        # 需要基础判断剩下的是否成立
        #如果没车，1-13都要判断（返回：3s: 10号有无车 or 9号>=6）放车基础判断
        #backward
        #如果有等待的车，到达的车道是确定的，两种动作（去返回或者送出）
        # ，返回需要基础判断，返回需要判断1号当前无车且到达时2号业务车
        #如果没有等待的车，判断每一个动作，到达车道(到达时间： 
        # 1号 有无车 or 2号 t+t到>=9) /返回车道(基础）有无车
        
        mask=np.zeros([2,13], dtype=np.int32)
         #forward 
        if self.in_transfer[0] == 1: #空闲  
            if self.backward_lines[0,0].sum() != 0: #返回车道有车 返回道接车+送进哪个车道 
                for j in [2,4,6,8,10,12]: #送进哪个车道
                    time_notes = in_transfer_time_dic[j]
                    line = j // 2 - 1
                    if self.forward_lines[line, 0].sum() == 0: 
                        mask[0][j]=1
                    elif self.forward_lines[line, 0, self.car_bit]+\
                        time_notes[1] >= 9:
                        for k in range(0,10):
                            if self.forward_lines[line, k].sum() == 0: 
                                mask[0][j]=1
                                break
            else:
                mask[0][0]=1
                if  self.backward_lines[0,1, self.car_bit]+3>=9: #如果返回道未来可以接， 返回道接车+送进哪个车道 
                    for j in [2,4,6,8,10,12]: #先去返回道接车，然后送进哪个车道
                        time_notes = in_transfer_time_dic[j]
                        line = j // 2 - 1
                        if self.forward_lines[line, 0].sum() == 0:
                            mask[0][j]=1
                        elif self.forward_lines[line, 0, self.car_bit]\
                            +time_notes[1] >= 9:
                            for k in range(0,10):
                                if self.forward_lines[line, k].sum() == 0: 
                                    mask[0][j]=1
                                    break
                #判断能否把车送进车道
                if self.input_index<self.input.shape[0]:
                    for j in [1,3,5,7,9,11]:  
                        time_notes = in_transfer_time_dic[j]
                        line = j // 2
                        if self.forward_lines[line, 0].sum() == 0:
                            mask[0][j]=1
                        elif self.forward_lines[line, 0, self.car_bit]\
                            +time_notes[1] >= 9  :
                            for k in range(0,10):
                                if self.forward_lines[line, k].sum() == 0: 
                                    mask[0][j]=1
                                    break
        else:
            pass
            # print("in 不空闲")
        #backward
        if self.out_transfer[0] == 1: #空闲的
            idmax=self.forward_lines[:,9,self.car_bit].argmax()
            if self.forward_lines[idmax,9,self.car_bit]!=0: #有车等待，必须去接车
                policy=idmax*2+1  #直接送出去
                mask[1][policy]=1 #可以送出去
                time_notes = out_transfer_time_dic[policy+1]
                if self.backward_lines[0,9].sum()==0: #返回没车，那么到达的时候肯定没车
                    mask[1][policy+1]=1 #可以放返回
                elif self.backward_lines[0, 9,self.car_bit]+\
                    time_notes[1]>=9:  #此时有车，如果有一个空位
                    for k in range (0,10):
                         if self.backward_lines[0, k].sum() == 0: 
                            mask[1][policy+1]=1
                            break
            else:
                for j in [2,4,6,8,10,12]: #车道，先去车道接车再去返回
                    time_notes = out_transfer_time_dic[j]
                    line = j // 2 - 1
                    #判断能否接到车
                    if self.forward_lines[line, 9].sum() != 0:
                        mask[1][j]=1
                    else:
                        if self.forward_lines[line, 8, self.car_bit]+\
                            time_notes[0] >= 9 and self.forward_lines[line, 8].sum()!=0:#接车必须有车
                            mask[1][j]=1
                    if mask[1][j]:
                        fh=0
                        #判断能否放返回
                        if self.backward_lines[0,9].sum()==0:  #返回道空闲，判断能否接到车
                            fh=1
                        elif self.backward_lines[0, 9, -1]+time_notes[1]>=9 :  #返回道此时不空闲，后续空闲
                            for k in range (0,10):
                                if self.backward_lines[0, k].sum() == 0: 
                                    fh=1
                                    break
                        mask[1][j]=fh
    
                for j in [1,3,5,7,9,11]: #直接送出去，车道有没有车
                    time_notes = out_transfer_time_dic[j]
                    line = j // 2 
                    if self.forward_lines[line, 9].sum() != 0:
                        mask[1][j]=1
                    else:
                        if self.forward_lines[line, 8, self.car_bit]+\
                            time_notes[0] >= 9 and self.forward_lines[line, 8].sum()!= 0:
                            mask[1][j]=1
                mask[1][0]=1
        else:
            pass
            # print("out 不空闲")
        index = np.array([2,4,6,8,10,12], dtype=np.int)
        mask[1, index] = 0

        if mask[0].sum() <= 1:
            mask[0] = np.zeros(mask[0].shape, dtype=np.int32)
        if mask[1].sum() <= 1:
            mask[1] = np.zeros(mask[1].shape, dtype=np.int32)

        return mask

    def in_transfer_time_plus1(self,):
        in_policy_index = np.where(self.in_transfer[:self.transfer_work_bit] != 0)[0]
        if in_policy_index != 0: # 如果in_transfer不空闲
            self.in_transfer[self.transfer_work_bit+self.car_bit] += 1 # 时间减一

    def out_transfer_time_plus1(self):
        out_policy_index = np.where(self.out_transfer[:self.transfer_work_bit] != 0)[0]
        if out_policy_index != 0: # 如果out_transfer不空闲
            self.out_transfer[self.transfer_work_bit+self.car_bit] += 1 # 时间减一

    def in_transfer_step(self, in_transfer_action):
        in_policy_index = np.where(self.in_transfer[:self.transfer_work_bit] != 0)[0]
        in_transfer_action_index = np.where(in_transfer_action != 0)[0]
        if in_policy_index == 0 and in_transfer_action_index != 0:
            push_line = (in_transfer_action_index - 1) // 2
            pull_back = (in_transfer_action_index - 1) % 2
            if pull_back == 0 and in_transfer_action_index != 7: 
                self.in_transfer[:self.transfer_work_bit] = in_transfer_action.copy()
                assert(self.input_index < len(self.input)),('no input')
                self.in_transfer[self.transfer_work_bit:self.\
                    transfer_work_bit+ self.car_bit] = self.input_pull()
                self.in_transfer[self.transfer_work_bit+self.car_bit] += 1
                # time_notes = in_transfer_time_dic[in_transfer_action_index[0]]
                # self.in_transfer[self.transfer_work_bit+self.car_bit] = time_notes[2]
            if pull_back == 0 and in_transfer_action_index == 7: # 如果in_transfer空闲且动作有效，赋予transfer动作，car信息，倒计时
                push_line = (in_transfer_action_index - 1) // 2
                assert (self.forward_lines[push_line, 0].sum() == 0), ("forward_lines[0]!")
                assert(self.input_index < len(self.input)),('no input')
                self.forward_lines[push_line, 0, :self.car_bit] = self.input_pull()
            if pull_back != 0:
                self.in_transfer[:self.transfer_work_bit] = in_transfer_action.copy()
                self.in_transfer[self.transfer_work_bit+self.car_bit] += 1

        if in_policy_index != 0: # 如果in_transfer不空闲
            assert (len(in_transfer_action_index) == 0), ("in_transfer_action_index!")
            time_notes = in_transfer_time_dic[in_policy_index[0]]
            push_line = (in_policy_index - 1) // 2
            pull_back = (in_policy_index - 1) % 2

            if self.in_transfer[self.transfer_work_bit+self.car_bit] == time_notes[0]:
                assert (time_notes[0] != 0), ("time_notes[0]!")
                if pull_back != 0:
                    assert (self.backward_lines[0, 0].sum() != 0), ("backward_lines[0]!")
                    self.in_transfer[self.transfer_work_bit:self.transfer_work_bit+\
                        self.car_bit] = self.backward_lines[0, 0, :self.car_bit].copy()
                    self.backward_lines[0, 0, ...] = np.zeros(self.backward_lines[0, 0, ...].shape, dtype=np.float32)

            if self.in_transfer[self.transfer_work_bit+self.car_bit] == time_notes[1]:
                assert (time_notes[1] != 0), ("time_notes[1]!")
                # print(self.forward_lines[push_line])
                assert (self.forward_lines[push_line, 0].sum() == 0), ("forward_lines[0]!")
                self.forward_lines[push_line, 0, :self.car_bit] = self.in_transfer[\
                    self.transfer_work_bit:self.transfer_work_bit+self.car_bit].copy()
                self.in_transfer[self.transfer_work_bit:self.transfer_work_bit+self.car_bit]\
                     = np.zeros((self.car_bit,), dtype=np.float32)

            if self.in_transfer[self.transfer_work_bit+self.car_bit] == time_notes[2]: # reset
                assert (time_notes[2] != 0), ("time_notes[2]!")
                self.in_transfer = np.zeros(self.in_transfer.shape, dtype=np.float32)
                self.in_transfer[0] = 1

            if self.in_transfer[self.transfer_work_bit+self.car_bit] > time_notes[2]:
                assert (False), ("超时!")


    def out_transfer_step(self, out_transfer_action):
        out_policy_index = np.where(self.out_transfer[:self.transfer_work_bit] != 0)[0]
        if out_policy_index == 0:
            assert(self.out_transfer.sum() == 1),('self.out_transfer')  
        out_transfer_action_index = np.where(out_transfer_action != 0)[0]
        if out_policy_index == 0 and out_transfer_action_index != 0:
            # assert(self.out_transfer.sum() == 1),('self.out_transfer')
            if out_transfer_action_index == 7:
                pull_line = (out_transfer_action_index - 1) // 2 # 0：放到返回车道，1：直接输出
                self.out_transfer[self.transfer_work_bit:self.transfer_work_bit+self.car_bit]\
                     = self.forward_lines[pull_line, -1, :self.car_bit].copy()
                self.forward_lines[pull_line, -1, :] = np.zeros(self.forward_lines\
                    [pull_line, -1, :].shape, dtype=np.float32)
                assert (self.out_transfer[self.transfer_work_bit:self.transfer_work_bit\
                    +self.car_bit].sum() != 0), ("car type!")
                self.output_push(self.out_transfer[self.transfer_work_bit:self.transfer_work_bit\
                    +self.car_bit].copy())
                self.out_transfer = np.zeros(self.out_transfer.shape, dtype=np.float32)
                self.out_transfer[0] = 1

            if out_transfer_action_index == 8: # 如果out_transfer空闲且动作有效，赋予out_transfer动作，倒计时
                pull_line = (out_transfer_action_index - 1) // 2 # 0：放到返回车道，1：直接输出
                self.out_transfer[self.transfer_work_bit:self.transfer_work_bit+self.car_bit]\
                     = self.forward_lines[pull_line, -1, :self.car_bit].copy()
                self.forward_lines[pull_line, -1, :] = np.zeros(self.forward_lines\
                    [pull_line, -1, :].shape, dtype=np.float32)
                self.out_transfer[:self.transfer_work_bit] = out_transfer_action.copy()
                self.out_transfer[self.transfer_work_bit+self.car_bit] += 1
            if out_transfer_action_index!=7 and out_transfer_action_index!=8: # 如果out_transfer空闲且动作有效，赋予out_transfer动作，倒计时
                self.out_transfer[:self.transfer_work_bit] = out_transfer_action.copy()
                self.out_transfer[self.transfer_work_bit+self.car_bit] += 1

        if out_policy_index != 0: # 如果out_transfer不空闲
            assert (len(out_transfer_action_index) == 0), ("out_transfer_action_index!")
            time_notes = out_transfer_time_dic[out_policy_index[0]]
            pull_line = (out_policy_index - 1) // 2 # 0：放到返回车道，1：直接输出
            push_back = (out_policy_index - 1) % 2 # 0：放到返回车道，1：直接输出
            if self.out_transfer[self.transfer_work_bit+self.car_bit] == time_notes[0]:
                # print(self.forward_lines[pull_line, :])
                assert (time_notes[0]!=0),("time_notes[0]")
                assert (self.forward_lines[pull_line, -1].sum() != 0), ("forward_lines[0]!") #接车，车道有车
                # assert(self.forward_lines[:,9,self.car_bit].argmax()==pull_line),("Max Wait!")
                self.out_transfer[self.transfer_work_bit:self.transfer_work_bit+self.car_bit] =\
                     self.forward_lines[pull_line, -1, :self.car_bit].copy()
                self.forward_lines[pull_line, -1, :] = np.zeros(self.forward_lines\
                    [pull_line, -1, :].shape, dtype=np.float32)

            if self.out_transfer[self.transfer_work_bit+self.car_bit] == time_notes[1]:
                assert (time_notes[1]!=0),("time_notes[1]")
                if push_back != 0:
                    assert (self.backward_lines[0, -1].sum() == 0), ("backward_lines[0]!")
                    self.backward_lines[0, -1, :self.car_bit] = self.out_transfer\
                        [self.transfer_work_bit:self.transfer_work_bit+self.car_bit].copy()
                    self.out_transfer[self.transfer_work_bit:self.transfer_work_bit+\
                        self.car_bit] = np.zeros((self.car_bit,), dtype=np.float32)
                    self.num_back_usage += 1
            if self.out_transfer[self.transfer_work_bit+self.car_bit] == time_notes[2]:
                assert (time_notes[2]!=0),("time_notes[2]")
                if push_back == 0:
                    assert (self.out_transfer[self.transfer_work_bit:self.transfer_work_bit\
                        +self.car_bit].sum() != 0), ("car type!")
                    self.output_push(self.out_transfer[self.transfer_work_bit:\
                        self.transfer_work_bit+self.car_bit].copy())
                self.out_transfer = np.zeros(self.out_transfer.shape, dtype=np.float32)
                self.out_transfer[0] = 1
            if self.out_transfer[self.transfer_work_bit+self.car_bit] > time_notes[2]:
                assert (False), ("超时!")

                # if out_transfer_action_index != 0: # 如果out_transfer空闲且动作有效，赋予out_transfer动作，倒计时
                #     self.out_transfer[:self.transfer_work_bit] = out_transfer_action.copy()

    def foward_line_step(self,):
        # forward lines
        for i_lines in range(self.forward_lines_num):
            for i_step in reversed(range(10)):
                if self.forward_lines[i_lines, i_step].sum() != 0:  # 该停车位有车
                    self.forward_lines[i_lines, i_step, self.car_bit] += 1 # 时间 + 1
                    if i_step<=8 and self.forward_lines[i_lines, i_step, self.car_bit] \
                        >= 9 and self.forward_lines[i_lines, i_step + 1].sum() == 0: # 如果时间结束，跳转
                        # reset
                        self.forward_lines[i_lines, i_step + 1,:self.car_bit] = \
                            self.forward_lines[i_lines, i_step,:self.car_bit].copy()
                        self.forward_lines[i_lines, i_step] = np.zeros(self.forward_lines\
                            [i_lines, i_step].shape, dtype=np.float32)

    def backward_line_step(self,):
                # backward lines
        for i_step in range(0, 10):
            if self.backward_lines[0, i_step].sum() != 0:  # 该停车位有车
                self.backward_lines[0, i_step, self.car_bit] += 1# 时间 + 1
                if i_step>=1 and self.backward_lines[0, i_step, self.car_bit] >= 9 \
                    and self.backward_lines[0, i_step - 1].sum() == 0: # 如果时间结束，跳转
                        self.backward_lines[0, i_step - 1,:self.car_bit] = \
                            self.backward_lines[0, i_step,:self.car_bit].copy()
                        self.backward_lines[0, i_step] = np.zeros(self.backward_lines\
                            [0, i_step].shape, dtype=np.float32)
   
    def step(self, action, save=False):
        self.is_out = False
        """
            action: 
        """     
        in_transfer_action = np.zeros((self.transfer_work_bit,))
        out_transfer_action = np.zeros((self.transfer_work_bit,))
        in_transfer_action[action[0]] = 1 if action[0] != None else in_transfer_action
        out_transfer_action[action[1]] = 1 if action[1] != None else out_transfer_action
        self.in_transfer_time_plus1()
        self.out_transfer_time_plus1()
        self.foward_line_step()
        self.backward_line_step()
        self.in_transfer_step(in_transfer_action)
        self.out_transfer_step(out_transfer_action)
        next_state,available_action,done=self.get_state(out_done=True)
        reward=self.get_reward(action)

        if self.input_index != 0:
            self.times += 1
            if save:
                self.save_state(done=done)
       # get_reward(),done
        return next_state,reward, available_action, done


    def save_state(self,done=False):
        
        # 
        cars = - np.ones([len(self.input),1])

        # 进口
        if self.input_index < len(self.input):
            car_id = int(self.input[self.input_index,0])
            cars[car_id-1] = 0

        if self.in_transfer[self.transfer_work_bit] != 0:
            car_id = int(self.in_transfer[self.transfer_work_bit])
            cars[car_id-1] = 1

        if self.out_transfer[self.transfer_work_bit] != 0:
            car_id = int(self.out_transfer[self.transfer_work_bit])
            cars[car_id-1] = 2

        # 出口
        if len(self.output) != 0:
            car_id = int(self.output[-1][0])
            cars[car_id-1] = 3
            
        for line in range(self.forward_lines.shape[0]):
            for i in range(self.forward_lines.shape[1]):
                if self.forward_lines[line, i, 0] != 0:
                    car_id = int(self.forward_lines[line, i, 0])
                    cars[car_id-1] = 10 ** ((10 - i) // 10+1) * (line+1) + (10 - i)

        for i in range(self.backward_lines.shape[1]):
            if self.backward_lines[0, i, 0] != 0:
                car_id = int(self.backward_lines[0, i, 0])
                cars[car_id-1] = 10 ** ((10 - i) // 10+1) * (6+1) + (10 - i)


        if self.cars_excel is None:
            self.cars_excel = cars
        else:
            self.cars_excel = np.concatenate((self.cars_excel, cars),axis=1)



    



