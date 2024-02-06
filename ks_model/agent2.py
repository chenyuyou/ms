from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np
# 定义四种类型的Agent

class MachineFactoryAgent(Agent):
    """生产机器的工厂Agent，包括技术属性、计算生产成本、定价规则、研发投资以及创新和模仿的能力"""
    def __init__(self, unique_id, model, A_i_tau, B_i_tau, w_t, mu_1, v, S_i_previous, zeta, zeta_1, zeta_2, alpha_1, beta_1, x_bar, x_underline):
        super().__init__(unique_id, model)

        self.A_i_tau = A_i_tau  # 机器工具的劳动生产率
        self.B_i_tau = B_i_tau  # 工厂自身使用的生产技术的劳动生产率
        self.w_t = w_t  # 当前工资水平
        self.mu_1 = mu_1  # 定价规则的固定加成率
        self.v = v  # 研发投资的比例系数
        self.S_i_previous = S_i_previous  # 上一周期的销售额
        self.zeta = zeta  # 创新和模仿的分配参数
        self.RD_i = 0  # 初始化研发投资额
        self.IN_i = 0  # 初始化创新投资额
        self.IM_i = 0  # 初始化模仿投资额
        self.zeta_1 = zeta_1  # 创新成功概率的参数
        self.alpha_1 = alpha_1  # Beta分布参数
        self.beta_1 = beta_1
        self.x_bar = x_bar  # Beta分布支持的上限
        self.x_underline = x_underline  # Beta分布支持的下限
        self.zeta_2 = zeta_2  # 模仿成功概率的参数



    def calculate_unit_cost(self):
        """计算单位生产成本"""
        return self.w_t / self.B_i_tau

    def calculate_price(self):
        """根据固定加成率定价"""
        unit_cost = self.calculate_unit_cost()
        return (1 + self.mu_1) * unit_cost

    def invest_in_RD(self):
        """计算研发投资以及创新和模仿的投资额"""
        self.RD_i = self.v * self.S_i_previous
        self.IN_i = self.zeta * self.RD_i
        self.IM_i = (1 - self.zeta) * self.RD_i

    def calculate_innovation_probability(self):
        """计算创新成功的概率"""
        self.theta_in_i = 1 - np.exp(-self.zeta_1 * self.IN_i)

    def determine_innovation_opportunity(self):
        """决定是否获得创新机会"""
        self.calculate_innovation_probability()
        # 使用伯努利分布来决定是否获得创新机会
        self.has_innovation_opportunity = np.random.binomial(1, self.theta_in_i)

    def update_technology(self):
        """如果创新成功，更新技术参数"""
        if self.has_innovation_opportunity:
            # 从Beta分布中抽取x_iA和x_iB
            x_iA = np.random.beta(self.alpha_1, self.beta_1) * (self.x_bar - self.x_underline) + self.x_underline
            x_iB = np.random.beta(self.alpha_1, self.beta_1) * (self.x_bar - self.x_underline) + self.x_underline
            
            # 更新技术参数
            self.A_i_tau = self.A_i_tau * (1 + x_iA)
            self.B_i_tau = self.B_i_tau * (1 + x_iB)

    def calculate_imitation_probability(self):
        """计算模仿成功的概率"""
        self.theta_im_i = 1 - np.exp(-self.zeta_2 * self.IM_i)

    def determine_imitation_opportunity(self):
        """决定是否获得模仿机会"""
        self.calculate_imitation_probability()
        # 使用伯努利分布来决定是否获得模仿机会
        self.has_imitation_opportunity = np.random.binomial(1, self.theta_im_i)

    def calculate_technological_distance(self, other_agent):
        """计算与另一企业的技术距离"""
        distance = np.sqrt((self.A_i_tau - other_agent.A_i_tau)**2 + (self.B_i_tau - other_agent.B_i_tau)**2)
        return distance

    def imitate_technology(self):
        """尝试模仿竞争对手的技术"""
        if self.has_imitation_opportunity:
            # 获取所有可能的模仿目标
            possible_targets = [agent for agent in self.model.schedule.agents if isinstance(agent, MachineFactoryAgent) and agent.unique_id != self.unique_id]
            
            # 计算与每个目标的技术距离
            distances = [self.calculate_technological_distance(target) for target in possible_targets]
            total_distance = sum(distances)
            
            # 根据距离加权选择模仿目标
            probabilities = [distance / total_distance for distance in distances]
            target = np.random.choice(possible_targets, p=probabilities)
            
            # 更新技术参数为所选目标的参数
            self.A_i_tau = target.A_i_tau
            self.B_i_tau = target.B_i_tau



    def step(self):
        # 示例中仅计算并更新单位生产成本、价格、研发投资及其在创新和模仿上的分配
        self.unit_cost = self.calculate_unit_cost()
        self.price = self.calculate_price()
        self.invest_in_RD()
        self.determine_innovation_opportunity()
        self.update_technology()
        self.determine_imitation_opportunity()
        self.imitate_technology()

class ConsumerGoodsFactoryAgent(Agent):
    """消费品工厂Agent，包括生产决策和需求预期的计算"""
    def __init__(self, unique_id, model, A_i_tau, w_t, h, iota, initial_capital_stock):
        super().__init__(unique_id, model)
#        self.A_i_tau = A_i_tau  # 从机器工厂获得的机器的劳动生产率
#        self.w_t = w_t  # 当前工资水平
        self.h = h  # 考虑过去需求的时期数
        self.demand_history = []  # 过去需求的历史记录
        self.iota = iota  # 期望库存与预期需求的比例
        self.demand_history = []  # 过去需求的历史记录
        self.inventory = 0  # 当前实际库存
        self.capital_stock = initial_capital_stock  # 当前资本存量
        self.machines = []  # 企业拥有的所有机器集合
        self.machines = initial_machines  # 初始机器列表


    def calculate_unit_labor_cost(self):
        """计算单位劳动成本"""
        self.unit_labor_cost = self.w_t / self.A_i_tau

    def update_demand_history(self, demand):
        """更新需求历史记录"""
        self.demand_history.append(demand)
        # 保持历史记录长度为h
        if len(self.demand_history) > self.h:
            self.demand_history.pop(0)

    def calculate_adaptive_demand_expectation(self):
        """计算适应性需求预期"""
        if len(self.demand_history) == 0:
            return 0  # 如果没有历史数据，预期需求为0
        # 这里简化为使用过去h期需求的平均值作为预期需求
        return sum(self.demand_history) / len(self.demand_history)

    def plan_production(self):
        """根据适应性需求预期来计划生产量"""
        demand_expectation = self.calculate_adaptive_demand_expectation()
        # 简化模型：直接将预期需求作为生产计划
        self.production_plan = demand_expectation

    def calculate_desired_production_level(self):
        """计算期望生产水平"""
        demand_expectation = self.calculate_adaptive_demand_expectation()
        desired_inventory = self.iota * demand_expectation  # 计算期望库存
        if len(self.demand_history) < 1:
            actual_inventory_last_period = 0
        else:
            actual_inventory_last_period = self.inventory  # 假设inventory已经更新为上一时期的实际库存
        self.production_plan = demand_expectation + desired_inventory - actual_inventory_last_period

    def calculate_desired_capital_stock(self):
        """基于期望生产水平计算期望的资本存量"""
        # 这里简化为直接将期望生产量视为期望资本存量，实际应用中可能需要更复杂的函数
#########  这里有问题
        self.desired_capital_stock = self.production_plan

    def determine_investment(self):
        """确定是否需要进行投资以扩大产能"""
        self.calculate_desired_capital_stock()
        if self.desired_capital_stock > self.capital_stock:
            self.investment_needed = self.desired_capital_stock - self.capital_stock
        else:
            self.investment_needed = 0

    def scrap_machines(self, new_machine_price, new_machine_cost, payback_period_b):
        """根据技术陈旧程度和新机器的价格决定替换旧机器"""
        machines_to_scrap = [machine for machine in self.machines if (new_machine_price / (new_machine_cost - machine.production_cost)) <= payback_period_b]
        
        # 移除需要报废的机器
        for machine in machines_to_scrap:
            self.machines.remove(machine)
        
        # 计算替换投资，即需要购买的新机器数量
        self.replacement_investment = len(machines_to_scrap)

    def choose_machine_supplier(self, known_suppliers):
        """从已知供应商中选择机器"""
        # 假设known_suppliers是一个包含(machine_price, machine_productivity, supplier_id)元组的列表
        # 选择最低价格和单位生产成本的机器
        best_choice = min(known_suppliers, key=lambda x: x[0] + self.bc * x[1])
        self.order_machine(best_choice[2], best_choice[0], best_choice[1])

    def order_machine(self, supplier_id, price, productivity):
        """向选定的供应商订购机器"""
        # 这里简化处理，假设即时订购并接收机器
        new_machine = Machine(productivity, price, self.model.schedule.time)
        self.machines.append(new_machine)
        # 更新投资额
        self.investment += price

    def calculate_total_investment(self):
        """计算总投资"""
        # 总投资已通过订购机器时更新，这里可以进行其他相关计算或统计

    def step(self):
        # 示例中仅计算并更新单位劳动成本
#        self.calculate_unit_labor_cost()
        self.update_demand_history()
        self.plan_production()
        self.calculate_desired_production_level()
        self.determine_investment()
        self.scrap_machines(new_machine_price, new_machine_cost, payback_period_b)

class Machine:
    """定义一个简单的Machine类来表示不同年代的机器"""
    def __init__(self, A_i_tau, production_cost, vintage):
        self.A_i_tau = A_i_tau  # 机器的生产率
        self.production_cost = production_cost  # 机器的单位生产成本
        self.vintage = vintage  # 机器的年代


class ConsumerWorkerAgent(Agent):
    """消费者/工人Agent"""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # 初始化属性，比如收入、所购买的消费品数量等

    def step(self):
        # 定义每个时间步的行为，比如工作赚钱、购买消费品
        pass

class PublicSectorAgent(Agent):
    """公共部门Agent"""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # 初始化属性，比如税收、公共服务支出等

    def step(self):
        # 定义每个时间步的行为，比如征税、提供公共服务
        pass

# 定义模型

class EconomicModel(Model):
    """经济模型"""
    def __init__(self, N):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        # 可以添加更多初始化代码，比如创建agents、设置空间等

    def step(self):
        # 模拟一个时间步
        self.schedule.step()


