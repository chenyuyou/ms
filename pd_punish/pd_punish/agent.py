from mesa import Agent
import random
import numpy as np

    
def fitness(intense, payoff, action, noise, k, f, c, t):
    '''
    带惩罚机制的适应度计算  20210512
    '''
    if action == 0:   # 合作的适应度计算，只有合作的信号和合作的行为收益
        f=0
        k=0
    elif action == 1: # 背叛的适应度计算，只有背叛信号和背叛行为的收益
        c=0
        f=0
        k=0
    else:   # 惩罚的适应度计算，只有惩罚行为和惩罚信号的收益
        c=0
                    # 善意惩罚的适应度计算
    return np.exp(intense*payoff*(1-np.exp(-(noise+((1-noise)*(k*f+c))))))
#    return np.exp(intense*payoff*(1-np.exp(-(1-noise)*(k*f+c))))




class PDAgent(Agent):
    def __init__(self, unique,  model, starting_move=None) -> None:
        super().__init__(unique, model) 
        self.score = 0
        if starting_move:
            self.move = starting_move
        else:  
            self.move = random.choice([0,1,2])                                         
        self.next_move = random.choice([0,1,2])


    def step(self):
#        if self.model.schedule_type != "Simultaneous":
#            self.advance()
        self.advance()
        if random.uniform(0, 1) < self.model.mu:
            self.next_move = random.choice([0,1,2])


    
    def advance(self):
        self.move = self.next_move
#        self.next_move = self.move
        self.score =self.score+self.increment_score()
#        self.score = self.increment_score()

    def increment_score(self):
        self.other_agent = random.choice(self.model.schedule.agents)
#        if self.model.schedule_type == "Simultaneous":
#            move = self.other_agent.next_move 
#        else:
        move = self.other_agent.move
#            move = self.other_agent.move 
        if self.other_agent.score > self.score:
            self.next_move = self.other_agent.move
        else:
            self.next_move = self.move
        return fitness(self.model.intense, self.model.pd_payoff[self.move][move], self.move, self.model.noise, self.model.k, self.model.f, self.model.c, self.model.t) 
            


 
