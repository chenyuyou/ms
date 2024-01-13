from mesa import Agent
import random
import numpy as np


class Household(Agent):
    def __init__(self, unique,  model, starting_move=None) -> None:
        super().__init__(unique, model) 
        self.score = 0

class CapitalGoodFirm(Agent):
    '''
    A_LaborProductivity:企业为消费品行业所生产的机床的劳动生产率。
    B_LaborProductivity:企业自身所采用的生产技术的劳动生产率。
    Tau:当前的技术年限。
    '''
    def __init__(self, unique,  model, Wage, B_LaborProductivity) -> None:
        super().__init__(unique, model) 
        self.Wage = Wage
        self.B_LaborProductivity = B_LaborProductivity
        self.Cost = 0

    def cost(self):
        self.Cost = self.Wage/self.B_LaborProductivity

    

class ConsumptionGoodFirm(Agent):
    def __init__(self, unique,  model, starting_move=None) -> None:
        super().__init__(unique, model) 
        self.score = 0


class Bank(Agent):
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
            


 
