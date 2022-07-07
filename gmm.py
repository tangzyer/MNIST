import numpy as np
from scipy.stats import multivariate_normal
np.random.seed(None)
from scipy.special import logsumexp
import copy

class MyGMM(object):
    def __init__(self, K=3):
        """
        高斯混合模型，用EM算法进行求解
        :param K: 超参数，分类类别
        涉及到的其它参数：
        :param N: 样本量
        :param D: 单个样本的维度
        :param alpha: 模型参数，高斯函数的系数，决定高斯函数的高度，维度（K）
        :param mu: 模型参数，高斯函数的均值，决定高斯函数的中型位置，维度（K,D）
        :param Sigma: 模型参数，高斯函数的方差矩阵，决定高斯函数的形状，维度（K,D,D）
        :param gamma: 模型隐变量，决定单个样本具体属于哪一个高斯分布，维度(N,K)
        """
        self.K = K
        self.params = {
            'alpha': None,
            'mu': None,
            'Sigma': None,
            'gamma': None
        }

        self.N = None
        self.D = None

    def __init_params(self):
        # alpha 需要满足和为1的约束条件
        alpha = np.ones((self.K, 1))
        alpha = alpha / np.sum(alpha)
        mu = np.random.rand(self.K, self.D)
        Sigma = np.diag([0.2]*self.D).astype(np.float64)
        # 虽然gamma有约束条件，但是第一步E步时会对此重新赋值，所以可以随意初始化
        gamma = np.random.rand(self.N, self.K)

        self.params = {
            'alpha': alpha,
            'mu': mu,
            'Sigma': Sigma,
            'gamma': gamma
        }

    def _gaussian_function(self, y_j, mu_k, Sigma_k):
        '''
        计算高纬度高斯函数
        :param y_j: 第j个观测值
        :param mu_k: 第k个mu值
        :param Sigma_k: 第k个Sigma值
        :return:
        '''
        # 先取对数
        n_1 = self.D * np.log(2 * np.pi)
        # 计算数组行列式的符号和（自然）对数。

        _, n_2 = np.linalg.slogdet(Sigma_k)

        # 计算矩阵的（乘法）逆矩阵。
        n_3 = np.dot(np.dot((y_j - mu_k).T, np.linalg.inv(Sigma_k)), y_j - mu_k)

        # 返回是重新取指数抵消前面的取对数操作
        return np.exp(-0.5 * (n_1 + n_2 + n_3)+200)

    def _E_step(self, x, label):
        mu = self.params['mu']
        Sigma = self.params['Sigma']
        # for i in range(self.K):
        #     pr_x_i = multivariate_normal.pdf(x, mean=mu[i], cov=Sigma[i], allow_singular=True)
        #     pr_a = []
        #     for j in range(self.N):
        #         pr_a.append(np.sum(self.params['alpha'][label[j]]))
        #     pr_a = np.array(pr_a)
        for i in range(self.K):
            # a = mu[i]
            # c = Sigma
            t = multivariate_normal.pdf(x, mean=mu[i], cov=Sigma, allow_singular=True)
            # t = self._gaussian_function(x[1], a, c)
            self.params['gamma'][:, i] = np.log(self.params['alpha'][i]) + multivariate_normal.logpdf(x, mean=mu[i], cov=Sigma, allow_singular=True)
            if np.isnan(self.params['gamma']).any():
                pass
        mask = np.array(np.array(label, dtype=bool))
        self.params['gamma'] = np.ma.array(self.params['gamma'], mask=~mask).filled(-np.inf)
        self.params['gamma'] -= logsumexp(self.params['gamma'], axis=1, keepdims=True)
        #self.params['']
        if np.isnan(self.params['gamma']).any():
            pass

    def _M_step(self, x, label):
        mu = self.params['mu']
        gamma = np.exp(self.params['gamma'])
        self.params['Sigma'][:,:]=0
        #Sigma_list = []
        for k in range(self.K):
            mu_k = mu[k]
            gamma_k = gamma[:, k]
            gamma_k_j_list = []
            mu_k_part_list = []
            for j in range(self.N):
                if label[j][k] == 1:
                    #omega_j = np.sum(self.params['alpha'][label[j]])
                    x_j = x[j]
                    gamma_k_j = gamma_k[j]
                    gamma_k_j_list.append(gamma_k_j)
                    # mu_k的分母的分母列表
                    mu_k_part_list.append(gamma_k_j * x_j)
                    sigma_j = np.outer((x_j - mu_k).T, (x_j-mu_k))
                    self.params['Sigma'] += gamma_k_j * sigma_j / self.N
                    if np.isnan(self.params['Sigma']).any():
                        pass
                    #Sigma_list.append(gamma_k_j * sigma_j)
                    # Sigma_k的分母列表
            # 对模型参数进行迭代更新
            self.params['mu'][k] = np.sum(mu_k_part_list, axis=0) / np.sum(gamma_k_j_list)
        min_eig = np.min(np.real(np.linalg.eigvals(self.params['Sigma'])))
        if min_eig < 0:
            self.params['Sigma'] -= 10* min_eig * np.eye(*self.params['Sigma'].shape)
        #self.params['Sigma'] = np.sum(Sigma_list, axis=0) / self.N
            #self.params['alpha'][k] = np.sum(gamma_k_j_list) / self.N

    def fit(self, x, label, max_iter=1000):
        x = np.array(x)
        self.N, self.D = x.shape
        self.__init_params()
        for i in range(max_iter):
            mu = copy.deepcopy(self.params['mu'])
            self._E_step(x, label)
            self._M_step(x, label)
            mu_new =  self.params['mu']
            a = np.linalg.norm(mu-mu_new)
            if a < 1e-4:
                print(a,i)
                break
            
            # if np.linalg.norm(mu_1-mu_2) < 0.05:
            #     break

    def predict(self, X):
        mu = self.params['mu']
        Sigma = self.params['Sigma']
        prediction = []
        for x in X:
            prob_singular = []
            for k in range(self.K):
                mu_k = mu[k]
                prob_singular.append(multivariate_normal.logpdf(x, mean=mu_k, cov=Sigma, allow_singular=True))
            prediction.append(np.array(prob_singular).argmax())
        return prediction


class bayesian(object):
    def __init__(self, centers, covs):
        self.centers = centers
        self.covs = covs
        self.K = len(self.centers)


    def predict(self, X):
        mu = self.centers
        Sigma = self.covs
        prediction = []
        for x in X:
            prob_singular = []
            for k in range(self.K):
                mu_k = mu[k]
                Sigma_k = Sigma[k]
                prob_singular.append(multivariate_normal.pdf(x, mean=mu_k, cov=Sigma_k, allow_singular=True))
            prediction.append(np.array(prob_singular).argmax())
        return prediction