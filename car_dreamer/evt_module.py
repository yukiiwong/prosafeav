from scipy.stats import genpareto
from copulas.bivariate import Frank
import numpy as np
from collections import deque

class CopulaEVTModel:
    def __init__(self, threshold_ttc=3.0, threshold_drac=1.5, buffer_size=10000, min_sample=100):
        """
        初始化 EVT + Copula 模型
        - threshold_ttc: 小于该值视为 extreme（反转后）
        - threshold_drac: 大于该值视为 extreme
        """
        self.threshold_ttc = threshold_ttc      # 实际上作用于 -ttc
        self.threshold_drac = threshold_drac
        self.buffer = deque(maxlen=buffer_size)
        self.min_sample = min_sample
        self.copula = None
        self.gpd_ttc_params = None
        self.gpd_drac_params = None

    def add_sample(self, ttc, drac):
        """
        添加样本（排除 None 和过大值）
        - TTC 进行反转存储
        """
        if ttc is not None and drac is not None and abs(ttc) < 10 and abs(drac) < 10:
            self.buffer.append((-ttc, drac))  # 反转 TTC，使得“越大越危险”

    def _fit_gpd(self, data, threshold):
        """
        基于 POT 模型拟合 Generalized Pareto Distribution (GPD)
        """
        excess = [x - threshold for x in data if x > threshold]
        if len(excess) < self.min_sample:
            return None
        return genpareto.fit(excess)

    def _transform_to_uniform(self, data, threshold, gpd_params):
        """
        将原始数据转为[0,1]上的伪 U(0,1) 分布值，用于 copula 建模
        """
        c, loc, scale = gpd_params
        return [
            1.0 if x <= threshold else 1 - genpareto.cdf(x - threshold, c, loc=loc, scale=scale)
            for x in data
        ]

    def update_model(self):
        """
        拟合边缘 GPD 模型后，联合拟合 Frank Copula 模型
        """
        ttc_vals = [x[0] for x in self.buffer]
        drac_vals = [x[1] for x in self.buffer]

        self.gpd_ttc_params = self._fit_gpd(ttc_vals, self.threshold_ttc)
        self.gpd_drac_params = self._fit_gpd(drac_vals, self.threshold_drac)

        if self.gpd_ttc_params is None or self.gpd_drac_params is None:
            print("[CopulaEVT] Not enough data to update EVT margins.")
            return

        u_ttc = self._transform_to_uniform(ttc_vals, self.threshold_ttc, self.gpd_ttc_params)
        u_drac = self._transform_to_uniform(drac_vals, self.threshold_drac, self.gpd_drac_params)

        data_uniform = np.array(list(zip(u_ttc, u_drac)))

        try:
            self.copula = Frank()
            self.copula.fit(data_uniform)
            print("[CopulaEVT] Bivariate Frank copula updated with", len(data_uniform), "samples.")
        except Exception as e:
            print(f"[CopulaEVT] Copula fitting failed: {e}")
            self.copula = None

    def get_joint_risk(self, ttc, drac):
        """
        估计联合尾部风险（越接近1越危险）
        """
        if self.copula is None or self.gpd_ttc_params is None or self.gpd_drac_params is None:
            return 0.0

        if len(self.buffer) < self.min_sample:
            return 0.0

        ttc_rev = -ttc
        if ttc_rev <= self.threshold_ttc or drac <= self.threshold_drac:
            return 0.0

        try:
            u1 = 1 - genpareto.cdf(ttc_rev - self.threshold_ttc, *self.gpd_ttc_params)
            u2 = 1 - genpareto.cdf(drac - self.threshold_drac, *self.gpd_drac_params)

            joint_prob = self.copula.cumulative_distribution(np.array([[u1, u2]]))[0]
            return 1 - joint_prob
        except Exception as e:
            print(f"[CopulaEVT] Risk estimation failed: {e}")
            return 0.0

    def get_evt_reward(self, ttc, drac, weight=1.0):
        """
        转化为负风险惩罚，用于 RL 奖励函数
        """
        risk = self.get_joint_risk(ttc, drac)
        return -weight * risk
