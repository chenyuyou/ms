# -*- coding: UTF-8 -*-
import itertools, time
from mesa import Model
from mesa.batchrunner import batch_run
from mesa.datacollection import DataCollector
from mesa.time import RandomActivationByType
## from pd_punish.agent import PDAgent
import numpy as np
import pandas as pd
from multiprocessing import cpu_count

def track_run(model):
    return model.uid

def cooperate(model):
    return len([a for a in model.schedule.agents if a.move == 0])/model.num_agents
#    return len([a for a in model.schedule.agents if a.move == 0])



class PdNetworkModel(Model):

    id_gen = itertools.count(1)

    def __init__(self, capital_firms=50, consumption_firm=200, capital_goods_firm_rule_mu1=0.04,
                 RD_investment_propersity_v=0.04, RD_allocation_xi=0.50,
                 firm_serach_capability_zeta1=0.30, firm_serach_capability_zeta2=0.30,
                 beta_parameter_alpha=3, beta_parameter_beta=3,
                 beta_support_underline=-0.15, beta_support_bar=0.15,
                 new_customer_sample_gamma=0.50,
                 desired_inventories_iota=0.10, payback_period_b=3,
                 scrapping_age_eta=20, mark_up_coefficient_v=0.04,
                 competitiveness_weights_omega1=1, competitiveness_weights_omega2=1,
                 replicator_dynamics_coefficient_chi=1, maximum_debt_sales_ratio_Lambda=2,
                 interest_rate_r=0.01, uniform_distribution_supports_phi1=0.10,
                 uniform_distribution_supports_phi2=0.90, uniform_distribution_supports_phi3=0.10,
                 uniform_distribution_supports_phi4=0.90, beta_distribution_parameters_alpha2=2,
                 beta_distribution_parameters_beta2=4, wage_setting_AB_weight_psi1=1,
                 wage_setting_cpi_weight_psi2=0, wage_setting_u_weight_psi3=0,
                 tax_rate_tr=0.10, unemployment_subsidy_rate_varphi=0.40,
                 seed=None):
        super().__init__()
        self.capital_firms = capital_firms  # 资本品厂家数
        self.consumption_firm = consumption_firm    # 消费品厂家数
        self.capital_goods_firm_rule_mu1 = capital_goods_firm_rule_mu1    # 产品售价与成本的比例系数。
        self.RD_investment_propersity_v = RD_investment_propersity_v    # 研发费用占产品线销售额的比例。
        self.RD_allocation_xi = RD_allocation_xi    # 研发费用中用于创新的比率， 1-xi是研发费用中用于模仿的比率。
        
        self.firm_serach_capability_zeta1 = firm_serach_capability_zeta1    # 企业创新能力搜索参数
        self.firm_serach_capability_zeta2 = firm_serach_capability_zeta2    # 企业模仿能力搜索参数
        self.beta_parameter_alpha = beta_parameter_alpha    # 创新过程Beta分布参数中的alpha
        self.beta_parameter_beta = beta_parameter_beta  # 创新过程Beta分布参数中的beta
        self.beta_support_underline = beta_support_underline    # 贝塔分布的支持区间下限。
        self.beta_support_bar = beta_support_bar    # 贝塔分布的支持区间上限。
        self.new_customer_sample_gamma = new_customer_sample_gamma  # 潜在新客户占历史客户的比例系数。
        self.desired_inventories_iota = desired_inventories_iota    # 期望库存
        
        self.payback_period_b = payback_period_b  # 投资的回收周期。
        self.scrapping_age_eta = scrapping_age_eta    # 报废周期
            # 

        # 记录第几次运行模型
        self.uid = next(self.id_gen) 
   
                                    
        self.schedule = RandomActivationByType(self)
       # 初始化agent
        for i in range(self.num_agents):
            agent = PDAgent(i, self)
            self.schedule.add(agent)

        self.datacollector = DataCollector(model_reporters={
                                           "Cooperation": cooperate,
                                           "Defect": defect,
                                           "Punisher": punish,
                                           })

        self.running = True

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


#def run(input):
def run():
#    a,b=input
    br_params = {
        'noise':[0.1],
        'intense':[0.1],
#        'b':[a],
#        'c':[b],
        'b':[16],
        'c':[15],
        'k':[6],
        'f':[5],
        'e':[5],
#        'mu':[i/10000 for i in range(0,51,5)]
        'mu':[0.01]
    }

    br = batch_run(PdNetworkModel,
                       number_processes=cpu_count(),  
                       parameters=br_params,
                       iterations=10,
                       max_steps=300,
                       data_collection_period=10
#                       model_reporters={"Data Collector": lambda m: m.datacollector}
                       )

    result = pd.DataFrame(br)
    result.to_csv("../new.csv")


if __name__ == '__main__':

    start=time.time()
    run()
    end=time.time()
    print(end-start)
    exit()