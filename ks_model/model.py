from mesa import Model
from .scheduler import RandomActivationByTypeFiltered
from mesa.datacollection import DataCollector
from agent import MachineFactoryAgent, ConsumerGoodsFirmAgent

class EconomicModel(Model):
    def __init__(self, N_magent=50, N_cagent=200, mu_1=0.04, v=0.04, xi=0.50, zeta_1=0.30, zeta_2=0.30, alpha_1=3, beta_1=3, x1_lower=-0.15, x1_upper=0.15, b=3,
                 
                 ):
#        self.num_magents = N_magent
#        self.num_cagents = N_cagent
#        self.mu_1 = mu_1
#        self.v = v
#        self.zeta_1 = zeta_1
#        self.zeta_2 = zeta_2  # 模仿成功概率的参数


        self.schedule = RandomActivationByTypeFiltered(self)        
        # 创建资本品公司
        for i in range(N_magent):
            mfirm = MachineFactoryAgent(i, self, mu_1, v, xi, zeta_1, zeta_2, alpha_1, beta_1, x1_lower, x1_upper, b)
            self.schedule.add(mfirm)
        # 创建消费品公司
        for i in range(N_cagent):
            cfirm = ConsumerGoodsFirmAgent(i, self, )
            self.schedule.add(cfirm)


    #    self.datacollector = DataCollector(model_reporters={
    #                                       "Cooperation": cooperate,
    #                                       "Defect": defect,
    #                                       "Punisher": punish,
    #                                       })


    def step(self):
        # 在每个时间步骤，模型让所有的agents行动一次
        self.schedule.step()
        self.datacollector.collect(self)