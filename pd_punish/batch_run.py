# -*- coding: UTF-8 -*-
import itertools, time
from mesa import Model
from mesa.batchrunner import batch_run
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from pd_punish.agent import PDAgent
import numpy as np
import pandas as pd
from multiprocessing import cpu_count

def track_run(model):
    return model.uid

def cooperate(model):
    return len([a for a in model.schedule.agents if a.move == 0])/model.num_agents
#    return len([a for a in model.schedule.agents if a.move == 0])

def defect(model):
    return len([a for a in model.schedule.agents if a.move == 1])/model.num_agents
#    return len([a for a in model.schedule.agents if a.move == 1])

def punish(model):
    return len([a for a in model.schedule.agents if a.move == 2])/model.num_agents
#    return len([a for a in model.schedule.agents if a.move == 2])

class PdNetworkModel(Model):

    id_gen = itertools.count(1)
    # 博弈顺序选择
    schedule_types = {"Sequential": BaseScheduler,
                      "Random": RandomActivation,
                      "Simultaneous": SimultaneousActivation}

    def __init__(self, num_agents=2000, intense=0.1, schedule_type="Random", b=2, c=2, k=1.5, f=1, e=5, noise=0.7, mu=0.00015, t=1, seed=None):    # 删除ratio
       # 博弈顺序确定
        super().__init__()
        #
        self.num_agents = num_agents        
        self.intense = intense

        # 记录第几次运行模型
        self.uid = next(self.id_gen) 
        ## 
        self.noise = noise                                              
        self.mu = mu
        self.f = f
        self.e = e
        self.t = t

        self.k = k 
        self.c = c
        self.b = b
                                    
        self.pd_payoff = ((b-c,        -c,      -c-e),
                          (  b,         0,      -e),
                          (b-f,        -f,      -f-e))                                      
        ##
        self.schedule_type = schedule_type
        self.schedule = self.schedule_types[self.schedule_type](self)
        ##
        
       # 初始化agent
        for i in range(self.num_agents):
            agent = PDAgent(i, self)
            self.schedule.add(agent)
       # 初始化需要手机的数据和内容
        self.datacollector = DataCollector(model_reporters={
                                           "Cooperation": cooperate,
                                           "Defect": defect,
                                           "Punisher": punish,
                                           })

        self.running = True

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)



def run():

    br_params = {
        'noise':[0.1],
        'intense':[0.1],
        'b':[16],
        'c':[15],
        'k':[6],
        'f':[5],
        'e':[5],

        'mu':[0.01]
    }

    br = batch_run(PdNetworkModel,
                       number_processes=cpu_count(),  
                       parameters=br_params,
                       iterations=10,
                       max_steps=300,
                       data_collection_period=10

                       )

    result = pd.DataFrame(br)
    

    result.to_csv("../new.csv")


if __name__ == '__main__':
    start=time.time()
    run()
    end=time.time()
    print(end-start)
    exit()
