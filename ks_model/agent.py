from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np

class MachineFactoryAgent(Agent):
    """生产机器的工厂"""
    def __init__(self, unique_id, model, mu_1, v, xi, zeta_1, zeta_2, alpha_1, beta_1, x1_lower, x1_upper, b):
        super().__init__(unique_id, model)

        self.mu_1 = mu_1  # 固定加成率
        self.v = v
        self.xi = xi  # 研发费用中用于创新的比率， 1-xi是研发费用中用于模仿的比率。
        self.zeta_1 = zeta_1  # 创新成功概率的参数
        self.zeta_2 = zeta_2  # 模仿成功概率的参数
        self.alpha_1 = alpha_1  # Beta分布参数α
        self.beta_1 = beta_1  # Beta分布参数β
        self.x1_lower = x1_lower  # Beta分布支持的下限
        self.x1_upper = x1_upper  # Beta分布支持的上限
        self.b = b      # 资本品投资的回收周期。

        self.A_tau = 1  # 消费品行业的劳动生产率
        self.B_tau = 1  # 公司自身使用的生产技术的劳动生产率
        self.tau = 1  # 当前技术年代
        self.wage = 1  # 初始化工资，具体值将在后续调整
        self.unit_cost = 0  # 初始化单位成本
        self.price = 0  # 初始化价格
        self.unit_labor_cost = 0  # 初始化消费品生产的单位劳动成本
        self.past_sales = 0  # 过去的销售额
        self.RD_investment = 0  # R&D投资额
        self.IN_investment = 0  # 创新投资额
        self.IM_investment = 0  # 模仿投资额
        self.innovation_chance = 0  # 创新机会的概率

    def compute_unit_cost(self):
        """计算生产单位成本"""
        if self.B_tau > 0:  # 确保分母不为零
            self.unit_cost = self.wage / self.B_tau
        else:
            self.unit_cost = float('inf')  # 如果B_tau为0，设定单位成本为无限大，代表生产不可行

    def compute_price(self):
        """根据固定加成规则计算价格"""
        self.price = (1 + self.mu_1) * self.unit_cost

    def compute_unit_labor_cost(self):
        """计算消费品生产的单位劳动成本"""
        if self.A_tau > 0:
            self.unit_labor_cost = self.wage / self.A_tau
        else:
            self.unit_labor_cost = float('inf')  # 如果A_tau为0，设定单位劳动成本为无限大，代表无法生产
#########################   支出
    def compute_RD_investment(self):
        """计算R&D投资额"""
        self.RD_investment = self.v * self.past_sales

    def allocate_RD_investment(self):
        """根据比例分配R&D投资到创新和模仿"""
        self.IN_investment = self.xi * self.RD_investment
        self.IM_investment = (1 - self.xi) * self.RD_investment
#################   第一部分支出，用于创新
    def compute_innovation_chance(self):
        """计算获得创新机会的概率"""
        self.innovation_chance = 1 - np.exp(-self.zeta_1 * self.IN_investment)

    def attempt_innovation(self):
        """决定是否获得创新机会"""
        self.compute_innovation_chance()
        return np.random.binomial(1, self.innovation_chance)
    
    def draw_technological_advance(self):
        """抽取技术进步的幅度"""
        xA = np.random.beta(self.alpha_1, self.beta_1) * (self.x1_upper - self.x1_lower) + self.x1_lower
        xB = np.random.beta(self.alpha_1, self.beta_1) * (self.x1_upper - self.x1_lower) + self.x1_lower
        return xA, xB

    def innovate(self):
        """执行创新逻辑"""
        xA, xB = self.draw_technological_advance()
        self.A_tau = self.A_tau * (1 + xA)
        self.B_tau = self.B_tau * (1 + xB)
#################   第二部分支出，用于模仿
    def compute_imitation_chance(self):
        """计算获得模仿机会的概率"""
        self.imitation_chance = 1 - np.exp(-self.zeta_2 * self.IM_investment)

    def attempt_imitation(self):
        """决定是否获得模仿机会"""
        self.compute_imitation_chance()
        return np.random.binomial(1, self.imitation_chance)

    def compute_technological_distance(self, other):
        """计算与另一个公司的技术距离"""
        distance = np.sqrt((self.A_tau - other.A_tau)**2 + (self.B_tau - other.B_tau)**2)
        return distance

    def choose_firm_to_imitate(self):
        """选择一个公司进行模仿，偏向技术水平相近的公司"""
        distances = [(self.compute_technological_distance(other), other) for other in self.model.schedule.agents if other != self]
        if distances:
            # 计算加权概率，这里简化为按距离的倒数加权，需要进一步标准化
            weights = [1 / distance for distance, _ in distances]
            sum_weights = sum(weights)
            normalized_weights = [w / sum_weights for w in weights]
            chosen_one = np.random.choice([other for _, other in distances], p=normalized_weights)
            return chosen_one
        else:
            return None

    def imitate(self):
        """执行模仿逻辑，复制选择的公司的技术"""
        firm_to_imitate = self.choose_firm_to_imitate()
        if firm_to_imitate:
            self.A_tau = firm_to_imitate.A_tau
            self.B_tau = firm_to_imitate.B_tau
#############################################
    def evaluate_technologies(self, b, gamma):
        """评估并选择要生产的机器技术"""
        # 计算现有技术、潜在创新和模仿技术的成本效益
        current_cost_efficiency = self.price + b * self.compute_unit_cost()
        innovation_cost_efficiency = self.compute_innovation_cost_efficiency(b)
        imitation_cost_efficiency = self.compute_imitation_cost_efficiency(b)

        # 选择成本效益最高的技术
        min_cost_efficiency = min(current_cost_efficiency, innovation_cost_efficiency, imitation_cost_efficiency)
        
        if min_cost_efficiency == innovation_cost_efficiency:
            self.innovate()
        elif min_cost_efficiency == imitation_cost_efficiency:
            self.imitate()
        # 如果当前技术最优，不需要操作
        
    def compute_innovation_cost_efficiency(self, b):
        """计算创新技术的成本效益"""
        # 假设创新后的价格和成本可以通过某种方法计算得到
        innovation_price = self.compute_innovation_price()
        innovation_unit_cost = self.compute_innovation_unit_cost()
        return innovation_price + b * innovation_unit_cost

    def compute_imitation_cost_efficiency(self, b):
        """计算模仿技术的成本效益"""
        # 假设模仿后的价格和成本可以通过某种方法计算得到
        imitation_price = self.compute_imitation_price()
        imitation_unit_cost = self.compute_imitation_unit_cost()
        return imitation_price + b * imitation_unit_cost

    def step(self):
        # 实现研发活动，尝试发现新产品和更高效的生产技术
        # 计算机器生产单位成本
        self.compute_unit_cost()
        # 根据固定加成规则计算价格
        self.compute_price()
        # 计算消费品生产的单位劳动成本
        self.compute_unit_labor_cost()
        # 这里可以添加更多与时间步骤相关的行为
        self.compute_RD_investment()
        self.allocate_RD_investment()
        # 决定是否获得创新机会
        # 包括决定是否获得创新机会、创新成功后更新技术水平等
        if self.attempt_innovation():
            self.innovate()
        if self.attempt_imitation():
            # 执行模仿逻辑，可能需要更新技术水平等
            self.imitate()
        # 在每个时间步骤，企业需要决定生产哪种技术的机器
        self.evaluate_technologies(b=0.1, gamma=0.5)  # 示例中b和gamma为示例值


class ConsumerGoodsFirmAgent(Agent):
    """消费品生产公司"""
    def __init__(self, unique_id, model, h, i_desired, initial_capital_stock,
                initial_machines, initial_liquid_assets, interest_rate,
                max_debt_sales_ratio, initial_markup, v, omega_1, omega_2,
                chi, initial_market_share, initial_debt, ):
        super().__init__(unique_id, model, h)
        self.i_desired = i_desired  # 期望库存与需求的比例
        self.capital_stock = initial_capital_stock  # 初始化资本存量
        self.machines = initial_machines  # 初始机器列表
        self.liquid_assets = initial_liquid_assets  # 初始化流动资产
        self.interest_rate = interest_rate  # 贷款利率
        self.max_debt_sales_ratio = max_debt_sales_ratio  # 最大负债/销售比率
        self.markup = initial_markup  # 初始化加成率
        self.v = v  # 加成率调整的灵敏度参数
        self.chi = chi  # 市场份额演化的灵敏度参数
        self.market_share = initial_market_share  # 初始化市场份额
        self.debt = initial_debt  # 初始化债务水平
        self.liquid_assets = initial_liquid_assets  # 初始化流动资产
        self.omega_1 = omega_1  # 价格的权重
        self.omega_2 = omega_2  # 未满足需求的权重

        self.actual_inventory = 0  # 初始化实际库存水平
        self.desired_inventory = 0  # 初始化期望库存水平
        self.expected_demand = 0  # 初始化预期需求
        self.desired_production = 0  # 初始化期望生产量
        self.desired_capital_stock = 0  # 初始化期望资本存量
        self.investment = 0  # 初始化投资额
        self.b = 0.1  # 回报期参数示例值
        self.suppliers_brochures = []  # 存储供应商的宣传册信息
        self.gross_investment = 0  # 初始化总投资
        self.markup = 0.2  # 初始化加成率为20%
        self.unit_cost = 0  # 初始化单位生产成本
        self.price = 0  # 初始化产品价格
        self.market_shares = []  # 初始化市场份额历史列表
        self.unfilled_demand = 0  # 初始化上一期的未满足需求
        self.profit = 0  # 初始化利润
        self.sales_revenue = 0  # 初始化销售收入
        self.actual_sales = 0  # 初始化实际销售量
        self.internal_funds_for_investment = 0  # 初始化用于投资的内部资金金额

    def compute_demand_expectation(self):
        """计算适应性需求预期"""
        if len(self.demand_history) < self.h:
            # 如果历史数据少于h期，则使用所有可用数据的平均值
            expected_demand = sum(self.demand_history) / len(self.demand_history) if self.demand_history else 0
        else:
            # 使用最近h期的数据计算平均需求
            expected_demand = sum(self.demand_history[-self.h:]) / self.h
        return expected_demand

    def plan_production(self):
        """根据需求预期规划生产量"""
        expected_demand = self.compute_demand_expectation()
        # 此处可以添加进一步的生产规划逻辑，例如根据预期需求和当前库存来确定生产量
        # 例如:
        # self.production_plan = ...

    def update_desired_inventory(self):
        """更新期望库存水平"""
        self.desired_inventory = self.i_desired * self.expected_demand

    def compute_desired_production(self):
        """计算期望生产量"""
        self.expected_demand = self.compute_demand_expectation()
        self.update_desired_inventory()
        self.desired_production = self.expected_demand + self.desired_inventory - self.actual_inventory

    def compute_desired_capital_stock(self):
        """计算期望的资本存量"""
        # 假设期望的资本存量直接与期望的生产量相关
        # 这里的计算方法可以根据具体的经济模型进行调整
        self.desired_capital_stock = self.desired_production * some_factor

    def compute_investment_needed(self):
        """计算扩展生产能力所需的投资额"""
        self.compute_desired_capital_stock()
        if self.desired_capital_stock > self.capital_stock:
            self.investment = self.desired_capital_stock - self.capital_stock
        else:
            self.investment = 0

    def compute_replacement_decision(self, new_machine_price, new_machine_unit_cost):
        """计算替换决策，确定哪些旧机器应该被替换"""
        machines_to_replace = []
        for machine in self.machines:
            old_machine_effectiveness = new_machine_price / (machine.productivity - new_machine_unit_cost)
            if old_machine_effectiveness <= self.b:
                machines_to_replace.append(machine)
        return machines_to_replace

    def receive_suppliers_brochures(self, brochures):
        """接收来自一部分资本货物供应商的宣传册"""
        self.suppliers_brochures = brochures

    def choose_machine(self, b):
        """选择成本效益最高的机器进行投资"""
        best_choice = None
        lowest_cost = float('inf')
        for brochure in self.suppliers_brochures:
            # 假设宣传册中包含价格p和单位生产成本c
            total_cost = brochure['price'] + b * brochure['unit_cost']
            if total_cost < lowest_cost:
                best_choice = brochure
                lowest_cost = total_cost
        return best_choice

    def make_investment_decision(self, b):
        """做出投资决策，并计算总投资"""
        chosen_machine = self.choose_machine(b)
        if chosen_machine:
            # 假设宣传册中包含了机器的购买成本
            self.gross_investment += chosen_machine['price']
            # 发送订单到机器制造商等逻辑

    def finance_production(self, production_cost):
        """融资生产活动"""
        if self.liquid_assets >= production_cost:
            # 如果流动资产足够覆盖生产成本，则直接使用流动资产
            self.liquid_assets -= production_cost
            borrowed_amount = 0
        else:
            # 如果流动资产不足，尝试贷款以覆盖剩余成本
            borrowed_amount = min(production_cost - self.liquid_assets, self.max_debt_sales_ratio * self.estimated_sales - self.current_debt)
            self.liquid_assets = 0  # 使用所有流动资产
            # 假设当前负债已更新为包括此次贷款
            self.current_debt += borrowed_amount
        return borrowed_amount

    def finance_investment(self, investment_cost):
        """融资投资活动"""
        # 首先使用剩余的流动资产
        if self.liquid_assets >= investment_cost:
            self.liquid_assets -= investment_cost
            additional_borrowing = 0
        else:
            additional_borrowing_needed = investment_cost - self.liquid_assets
            self.liquid_assets = 0  # 使用所有流动资产
            # 然后尝试贷款以覆盖剩余投资成本
            additional_borrowing = min(additional_borrowing_needed, self.max_debt_sales_ratio * self.estimated_sales - self.current_debt)
            self.current_debt += additional_borrowing
        return additional_borrowing

    def compute_average_productivity(self):
        """计算平均生产率"""
        total_productivity = sum(machine.productivity for machine in self.machines)
        if self.machines:
            return total_productivity / len(self.machines)
        else:
            return 0

    def compute_unit_cost_of_production(self):
        """计算单位生产成本"""
        average_productivity = self.compute_average_productivity()
        # 假设单位劳工成本为固定值，可以根据模型需求调整
        unit_labor_cost = 10
        if average_productivity > 0:
            self.unit_cost = unit_labor_cost / average_productivity
        else:
            self.unit_cost = float('inf')  # 如果没有生产力，则设置成本为无限大

    def set_price(self):
        """设置产品价格"""
        self.compute_unit_cost_of_production()
        self.price = (1 + self.markup) * self.unit_cost

    def update_markup_based_on_market_share(self):
        """基于市场份额变化更新加成率"""
        if len(self.market_shares) >= 2:
            # 计算市场份额的变化比例
            share_change_ratio = (self.market_shares[-1] - self.market_shares[-2]) / self.market_shares[-2]
            # 更新加成率
            self.markup *= (1 + self.v * share_change_ratio)
            # 确保加成率保持在合理范围内，例如不允许负值
            self.markup = max(self.markup, 0)

    def compute_competitiveness(self):
        """计算公司的竞争力指标"""
        self.competitiveness = -self.omega_1 * self.price - self.omega_2 * self.unfilled_demand

    def update_market_share(self, average_competitiveness):
        """更新公司市场份额"""
        share_change = self.chi * (self.competitiveness - average_competitiveness) / average_competitiveness
        self.market_share *= (1 + share_change)
        # 确保市场份额保持在合理范围内
        self.market_share = max(self.market_share, 0)
        self.market_share = min(self.market_share, 1)

    def compute_profit(self, interest_rate):
        """计算公司利润"""
        # 计算销售收入
        self.sales_revenue = self.price * self.actual_sales
        # 计算总成本，包括生产成本和债务利息
        total_cost = self.unit_cost * self.actual_sales + interest_rate * self.debt
        # 计算利润
        self.profit = self.sales_revenue - total_cost

    def update_liquid_assets(self):
        """更新公司的流动资产存量"""
        self.liquid_assets = self.liquid_assets + self.profit - self.internal_funds_for_investment
        # 在实际应用中，可能需要确保流动资产不为负值
        self.liquid_assets = max(self.liquid_assets, 0)


    def step(self):
        # 在每个时间步更新代理的状态
        self.plan_production()
        self.compute_desired_production()
        self.compute_investment_needed()
        new_machine_price = 100  # 新机器价格示例值
        new_machine_unit_cost = 5  # 新机器单位生产成本示例值
        machines_to_replace = self.compute_replacement_decision(new_machine_price, new_machine_unit_cost)
        self.receive_suppliers_brochures([...])  # 接收宣传册的示例数据
        self.make_investment_decision(b)
        production_cost = 1000  # 生产成本示例值
        investment_cost = 500  # 投资成本示例值
        self.finance_production(production_cost)
        self.finance_investment(investment_cost)
        self.update_markup_based_on_market_share()
        self.set_price()
        self.compute_competitiveness()
        self.compute_competitiveness()  # 假设之前已定义此方法计算竞争力
        average_competitiveness = self.model.compute_average_competitiveness()  # 从模型获取平均竞争力
        self.update_market_share(average_competitiveness)
        interest_rate = 0.05  # 假设利率为5%
        self.set_price()  # 假设之前已定义此方法计算价格
        self.compute_competitiveness()  # 假设之前已定义此方法计算竞争力
        average_competitiveness = self.model.compute_average_competitiveness()  # 从模型获取平均竞争力
        self.update_market_share(average_competitiveness)
        # 假设actual_sales和unit_cost根据模型逻辑在此之前已更新
        self.compute_profit(interest_rate)
        # 假设在计算利润之后，根据投资决策更新了用于投资的内部资金金额
        self.update_liquid_assets()

class ConsumptionGoodsMarketModel(Model):
    """消费品市场模型"""
    
    def __init__(self, N, omega_1, omega_2):
        self.schedule = RandomActivation(self)
        # 初始化消费品公司代理...
        for i in range(N):
            agent = ConsumerGoodsFirmAgent(i, self, omega_1, omega_2)
            self.schedule.add(agent)
    
    def compute_average_competitiveness(self):
        """计算消费品部门的平均竞争力"""
        total_competitiveness = 0
        total_market_share = 0
        for agent in self.schedule.agents:
            total_competitiveness += agent.competitiveness * agent.market_share
            total_market_share += agent.market_share
        if total_market_share > 0:
            average_competitiveness = total_competitiveness / total_market_share
        else:
            average_competitiveness = 0
        return average_competitiveness

    def step(self):
        # 在每个时间步骤执行代理的行为
        self.schedule.step()
        # 计算平均竞争力
        avg_competitiveness = self.compute_average_competitiveness()
  

class MarketModel(Model):
    """市场模型"""
    
    def __init__(self, N, omega_1, omega_2,psi_1, psi_2, psi_3):
        self.schedule = RandomActivation(self)
        self.LS = 1000  # 假设劳动供应是固定的
        self.wage = 10  # 初始化工资率
        self.psi_1 = psi_1  # 工资调整参数
        self.psi_2 = psi_2
        self.psi_3 = psi_3
        self.wage_rate = 10  # 初始化工资率
        self.phi = phi  # 失业补贴占当前市场工资的比例
        self.tax_rate = 0.2  # 假设的税率
        self.subsidy_rate = self.phi * self.wage_rate  # 计算失业补贴

    
    def remove_and_add_firms(self):
        """移除公司并添加新公司"""
        # 移除具有准零市场份额或负净资产的公司
        for agent in self.schedule.agents[:]:
            if agent.market_share <= 0.01 or agent.liquid_assets < 0:
                self.schedule.remove(agent)
                
        # 添加新公司以保持公司数量不变
        while len(self.schedule.agents) < N:
            new_agent = self.create_new_firm()
            self.schedule.add(new_agent)
    
    def create_new_firm(self):
        """创建新公司"""
        # 假设根据现有公司平均水平的一定比例确定新公司的资本存量和流动资产
        average_capital = np.mean([agent.capital_stock for agent in self.schedule.agents])
        average_liquid_assets = np.mean([agent.liquid_assets for agent in self.schedule.agents])
        
        new_capital_stock = average_capital * 0.5  # 例如，新公司资本存量为平均值的50%
        new_liquid_assets = average_liquid_assets * 0.5  # 新公司流动资产为平均值的50%
        
        # 创建新公司实例，这里需要根据实际情况来设置适当的参数
        new_agent = ConsumerGoodsFirmAgent(self.next_id(), self, new_capital_stock, new_liquid_assets, ...)
        return new_agent

    def compute_aggregate_labor_demand(self):
        """计算总的劳动需求"""
        LD = sum(agent.labor_demand for agent in self.schedule.agents if hasattr(agent, 'labor_demand'))
        return LD

    def update_employment_and_wage(self, AB_change, cpi_change, unemployment_change):
        """更新就业水平和工资率"""
        LD = self.compute_aggregate_labor_demand()
        L = min(LD, self.LS)  # 总就业水平是劳动需求和供应之间的最小值
        # 更新工资率
        self.wage += self.wage * (1 + self.psi_1 * AB_change + self.psi_2 * cpi_change + self.psi_3 * unemployment_change)
        # 更新其他状态...

    def compute_aggregate_consumption(self, LD, LS):
        """计算聚合消费"""
        employed_income = self.wage_rate * LD  # 雇佣工人的总收入
        unemployed_income = self.subsidy_rate * (LS - LD)  # 失业工人的总收入
        aggregate_consumption = employed_income + unemployed_income
        return aggregate_consumption

    def compute_national_account(self, LD, LS):
        """计算国民经济核算标准满足的条件"""
        # 计算总生产Y，这里简化为所有公司生产量的总和
        Y = sum(agent.production for agent in self.schedule.agents if hasattr(agent, 'production'))
        # 计算聚合消费C
        C = self.compute_aggregate_consumption(LD, LS)
        # 假设投资I和库存变化Delta N，这里用示例值
        I = 1000  # 投资的示例值
        Delta_N = 500  # 库存变化的示例值
        # 验证是否满足Y = C + I + Delta N
        assert Y == C + I + Delta_N, "National account identity does not hold"


    def step(self):
        # 执行每个代理的行为
        self.schedule.step()
        # 处理公司退出和新公司进入
        self.remove_and_add_firms()
        # 其他市场动态逻辑...
        AB_change = 0.02  # 示例值
        cpi_change = 0.03
        unemployment_change = -0.01
        # 更新就业水平和工资率
        self.update_employment_and_wage(AB_change, cpi_change, unemployment_change)
        # 更新工资率等逻辑...
        LD = self.compute_aggregate_labor_demand()  # 计算总的劳动需求
        LS = 1000  # 假设劳动供应是固定的
        # 计算并验证国民经济核算标准
        self.compute_national_account(LD, LS)

class Machine:
    """代表一台机器的类，包含年代、生产效率和购买成本"""
    def __init__(self, vintage, productivity, cost):
        self.vintage = vintage
        self.productivity = productivity
        self.cost = cost