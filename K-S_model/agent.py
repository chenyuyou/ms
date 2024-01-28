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
    def __init__(self, unique, model, mu1, v, xi, b, zeta1, zeta2, alpha_1, beta_1,
                 alpha_2, beta_2, x1_bar, x1_underline, gamma, iota, eta, v_, omega1, omega2,
                 chi, lambda_, r, phi_1, phi_2, phi_3, phi_4, psi_1, psi_2, psi_3, tr, varphi,
                 Wage, A_LaborProductivity, B_LaborProductivity
                 ) -> None:
        super().__init__(unique, model)
        self.mu1 = mu1  # 产品售价与成本的比例系数。
        self.v = v      # 研发费用占产品线销售额的比例。
        self.xi = xi    # 研发费用中用于创新的比率， 1-xi是研发费用中用于模仿的比率。
        self.b = b      # 投资的回收周期。
        self.zeta1 = zeta1  # 企业创新能力搜索参数
        self.zeta2 = zeta2  # 企业模仿能力搜索参数
        self.alpha_1 = alpha_1  # 创新过程Beta分布参数中的alpha
        self.beta_1 = beta_1    # 创新过程Beta分布参数中的beta
        self.alpha_2 = alpha_2  # Beta分布参数中的alpha
        self.beta_2 = beta_2    # Beta分布参数中的beta
        self.x1_bar = x1_bar
        self.x1_underline = x1_underline
        self.gamma = gamma
        self.iota = iota    # 期望库存
        self.eta = eta  # 报废周期
        self.v_ = v_    # 加成系数
        self.omega1 = omega1    # 竞争力权重
        self.omega2 = omega2    # 竞争力权重
        self.chi = chi  # 复制动力学系数
        self.lambda_ = lambda_  # 最大负债/销售比率
        self.r = r  # 利率
        self.phi_1 = phi_1  # 均匀分布支撑(消费-商品的进入资本)
        self.phi_2 = phi_2  # 均匀分布支撑(消费-商品的进入资本)
        self.phi_3 = phi_3  # 均匀分布支撑(流动资产的进入存量)
        self.phi_4 = phi_4  # 均匀分布支撑(流动资产的进入存量)
        self.psi_1 = psi_1  # 工资AB设定权重
        self.psi_2 = psi_2  # 工资cpi设定权重
        self.psi_3 = psi_3  # 工资U设定权重
        self.tr = tr    # 税率
        self.varphi = varphi    # 失业补贴率

        self.Wage_t = Wage
        self.A_LaborProductivity_tau = A_LaborProductivity
        self.B_LaborProductivity_tau = B_LaborProductivity
        self.CostCapital_t = 0
        self.CostConsumption_t = 0
        self.Price_t = 0 
        self.Sale_minus_1 = 0
        self.Research_t = 0
        self.Innovation_t = 0
        self.Imitation_t = 0
        self.SkillImproced_A_t = 0
        self.SkillImproced_B_t = 0
        self.Tech_A_t = 0
        self.Tech_B_t = 0
        self.TechInnovation_A_t = 0
        self.TechInnovation_B_t = 0
        self.TechImitation_A_t = 0
        self.TechImitation_B_t = 0

    def cost_for_capital(self):
        """
        计算资本品企业的单位生产成本。

        参数:
        w_t (float): 时间 t 的货币工资水平。
        B_i_tau (float): 企业 i 使用的生产技术的劳动生产率。

        返回:
        float: 单位生产成本。
        """
        self.CostCapital_t = self.Wage_t/self.B_LaborProductivity_tau

    def cost_for_consumpiton(self):
        """
        计算消费品部门的单位劳动成本。

        参数:
        w_t (float): 时间 t 的货币工资水平。
        A_i_tau (float): 企业 i 生产的机器的劳动生产率。

        返回:
        float: 单位劳动成本。
        """
        self.CostConsumption_t = self.Wage_t/self.A_LaborProductivity_tau

    def price_for_capital(self):
        """
        根据固定加成定价规则计算价格。

        参数:
        c_i_t (float): 单位生产成本。
        mu_1 (float): 固定加成比例。

        返回:
        float: 计算出的价格。
        """
        self.Price_t = (1 + self.mu1)*self.CostCapital_t

    def research_investment(self):
        """
        计算资本品行业企业的研发投资。

        参数:
        S_i_t_minus_1 (float): 企业过去的销售额。
        v (float): 销售额投资于研发的比例系数。

        返回:
        float: 研发投资。
        """    
        self.Research_t = self.v*self.Sale_minus_1

    def research_allocation(self):
        """
        计算企业在创新和模仿上的研发支出。

        参数:
        RD_i_t (float): 研发总支出。
        xi (float): 分配给创新的研发比例。

        返回:
        tuple: (创新研发支出, 模仿研发支出)。
        """
        self.Innovation_t = self.xi * self.Research_t
        self.Imitation_t = (1 - self.xi) * self.Research_t
####################################################################
    def innovation(self):
        """
        模拟创新过程。

        参数:
        IN_i_t (float): 创新研发支出。
        zeta_1 (float): 研发支出与创新机会关系的系数。
        A_i (float): 现有技术 A 的参数。
        B_i (float): 现有技术 B 的参数。
        alpha_1, beta_1 (float): 贝塔分布的参数。
        x1_bar, x1_underline (float): 贝塔分布的支持区间。

        返回:
        tuple: (是否获得创新机会, 新技术 A 的参数, 新技术 B 的参数)。
        """    
        # 判断是否获得创新机会
        self.Theta_innovation_t = 1 - np.exp(-self.zeta1 * self.Innovation_t)
        Innovation_chance = np.random.binomial(1, self.Theta_innovation_t)
        if Innovation_chance:
            self.SkillImproced_A_t = np.random.beta(self.alpha_1, self.beta_1) * (self.x1_bar - self.x1_underline) + self.x1_underline
            self.SkillImproced_B_t = np.random.beta(self.alpha_1, self.beta_1) * (self.x1_bar - self.x1_underline) + self.x1_underline

    def technological_distance(tech1, tech2):
        """
        计算两个技术之间的欧几里得距离。

        参数:
        tech1, tech2 (tuple): 技术参数 (A, B)。

        返回:
        float: 技术之间的欧几里得距离。
        """
        return np.sqrt((tech1[0] - tech2[0])**2 + (tech1[1] - tech2[1])**2)

    def simulate_imitation(self):
        """
        模拟模仿过程，考虑技术距离和加权模仿概率。

        参数:
        IM_i_t (float): 模仿研发支出。
        zeta_2 (float): 研发支出与模仿机会关系的系数。
        own_technology (tuple): 企业自身的技术 (A, B)。
        competitors_technologies (list of tuples): 竞争对手的技术列表，每个元素为 (A, B)。

        返回:
        tuple: (是否获得模仿机会, 模仿的技术 A, 模仿的技术 B)。
        """
        # 判断是否获得模仿机会
        self.Theta_imitation_t = 1 - np.exp(-self.zeta2 * self.Imitation_t)
        Imitation_chance = np.random.binomial(1, self.Theta_imitation_t)

        if Imitation_chance and competitors_technologies:
        # 计算与每个竞争对手的技术距离
            Distances = [calculate_technological_distance(own_technology, tech) for tech in competitors_technologies]
        # 转换为概率（距离越小，概率越高）
            Probabilities = [1 / d if d != 0 else 1.0 for d in Distances]
            Probabilities = np.array(Probabilities) / sum(Probabilities)
        # 加权选择模仿的技术
            chosen_tech_index = np.random.choice(len(competitors_technologies), p=Probabilities)
            self.TechImitation_A_t, self.TechImitation_B_t = competitors_technologies[chosen_tech_index]
    
    def choose_machine_to_produce(potential_machines, payback_period_parameter):
        """
        资本品企业选择生产哪种机器。

        参数:
        potential_machines (list of tuples): 潜在机器的列表，格式为 (price, unit_cost, technology_type)。
        payback_period_parameter (float): 回报期参数。

        返回:
        tuple: 被选择生产的机器信息。
        """
    # 计算每种机器的总成本（价格 + 回报期参数 * 单位成本）
        total_costs = [(price + payback_period_parameter * unit_cost, technology_type) for price, unit_cost, technology_type in potential_machines]

    # 选择总成本最低的机器
        return min(total_costs, key=lambda x: x[0])
################################################################################



class ConsumptionGoodFirm(Agent):

    def __init__(self, unique, Demand, model) -> None:
        super().__init__(unique, model) 
        self.Demand = Demand


    def demand_expectation(past_demands, h):
        """
        根据适应性需求预期计算预期需求。

        参数:
        past_demands (list): 过去 h 个时期的实际需求列表。
        h (int): 考虑的时期数。

        返回:
        float: 预期需求。
        """
        if len(past_demands) < h:
            raise ValueError("提供的过去需求数据少于 h 个时期。")

    # 计算过去 h 个时期的需求平均值
        return sum(past_demands[-h:]) / h

    def desired_production_level(expected_demand, desired_inventory_coefficient, actual_inventory_last_period):
        """
        计算消费品企业的期望生产水平。

        参数:
        expected_demand (float): 预期需求。
        desired_inventory_coefficient (float): 期望库存系数。
        actual_inventory_last_period (float): 上一时期末的实际库存。

        返回:
        float: 期望生产水平。
        """
        desired_inventory = desired_inventory_coefficient * expected_demand
        return expected_demand + desired_inventory - actual_inventory_last_period

    def replacement_decision(machine_vintages, new_machine_price, new_machine_unit_cost, b):
        """
        计算消费品企业的机器替换决策。

        参数:
        machine_vintages (list of tuples): 企业拥有的机器年代及其单位生产成本，格式为 (A_i_tau, unit_cost)。
        new_machine_price (float): 新机器的价格。
        new_machine_unit_cost (float): 新机器的单位生产成本。
        b (float): 替换决策的阈值。

        返回:
        list: 需要替换的机器年代。
        """
        to_replace = []
        for A_i_tau, unit_cost in machine_vintages:
            if new_machine_price / (unit_cost - new_machine_unit_cost) <= b:
                to_replace.append(A_i_tau)
        return to_replace

    def calculate_price(unit_cost, markup):
        """
        根据单位成本和加成率计算价格。

        参数:
        unit_cost (float): 单位生产成本。
        markup (float): 加成率。

        返回:
        float: 计算出的价格。
        """
        return (1 + markup) * unit_cost

    def calculate_markup(last_markup, market_share_last, market_share_before_last, v):
        """
        根据市场份额的变化计算新的加成率。

        参数:
        last_markup (float): 上一时期的加成率。
        market_share_last (float): 上一时期的市场份额。
        market_share_before_last (float): 前一个时期的市场份额。
        v (float): 调整强度系数。

        返回:
        float: 新的加成率。
        """
        if market_share_before_last == 0:
            return last_markup  # 避免除以零

        return last_markup * (1 + v * (market_share_last - market_share_before_last) / market_share_before_last)


    def calculate_competitiveness(price, unfilled_demand, omega_1, omega_2):
        """
        计算企业的竞争力。

        参数:
        price (float): 价格。
        unfilled_demand (float): 上一时期未满足的需求。
        omega_1, omega_2 (float): 影响竞争力的参数。

        返回:
        float: 企业的竞争力。
        """
        return -omega_1 * price - omega_2 * unfilled_demand

    def calculate_average_competitiveness(firms_competitiveness, past_market_shares):
        """
        计算消费品行业的平均竞争力。

        参数:
        firms_competitiveness (list of float): 各企业的竞争力。
        past_market_shares (list of float): 各企业的过去市场份额。

        返回:
        float: 消费品行业的平均竞争力。
        """
        if len(firms_competitiveness) != len(past_market_shares):
            raise ValueError("企业数量和市场份额数量不匹配。")

        total_competitiveness = sum(E_j * f_j for E_j, f_j in zip(firms_competitiveness, past_market_shares))
        return total_competitiveness

    def calculate_market_share_evolution(current_market_share, firm_competitiveness, average_competitiveness, chi):
        """
        计算市场份额的演变。

        参数:
        current_market_share (float): 当前市场份额。
        firm_competitiveness (float): 企业的竞争力。
        average_competitiveness (float): 行业平均竞争力。
        chi (float): 调整系数。

        返回:
        float: 新的市场份额。
        """
        if average_competitiveness == 0:
            return current_market_share  # 避免除以零

        return current_market_share * (1 + chi * (firm_competitiveness - average_competitiveness) / average_competitiveness)

    def calculate_profit(price, actual_sales, unit_cost, production_quantity, interest_rate, debt):
        """
        计算消费品企业的利润。

        参数:
        price (float): 价格。
        actual_sales (float): 实际销售量。
        unit_cost (float): 单位生产成本。
        production_quantity (float): 生产量。
        interest_rate (float): 利率。
        debt (float): 债务。

        返回:
        float: 企业的利润。
        """
        sales_revenue = price * actual_sales
        production_cost = unit_cost * production_quantity
        interest_expense = interest_rate * debt
        return sales_revenue - production_cost - interest_expense


    def calculate_liquid_assets_evolution(previous_liquid_assets, profit, internal_funds_for_investment):
        """
        计算消费品企业流动资产存量的演变。

        参数:
        previous_liquid_assets (float): 上一时期的流动资产存量。
        profit (float): 利润。
        internal_funds_for_investment (float): 用于融资投资的内部资金。

        返回:
        float: 新的流动资产存量。
        """
        return previous_liquid_assets + profit - internal_funds_for_investment


class Public(Agent):
    def __init__(self, unique, model) -> None:
        super().__init__(unique, model) 

   
    def demand(self):
        pass

    def produce(self):
        pass

    def investigate(self):
        pass


class Public(Agent):
    def __init__(self, unique, model) -> None:
        super().__init__(unique, model) 

