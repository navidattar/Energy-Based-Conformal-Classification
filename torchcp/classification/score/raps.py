# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F
from torchcp.classification.score.aps import APS, EnergyAPS


class RAPS(APS):
    """
    Method: Regularized Adaptive Prediction Sets 
    Paper: Uncertainty Sets for Image Classifiers using Conformal Prediction (Angelopoulos et al., 2020)
    Link: https://arxiv.org/abs/2009.14193
    
    Args:
        penalty (float): The weight of regularization. When penalty = 0, RAPS=APS.
        kreg (int, optional): The rank of regularization which is an integer in [0, classes_num]. Default is 0.
        score_type (str, optional): The type of score to use. Default is "softmax".
        randomized (bool, optional): Whether to use randomized scores. Default is True.
        
    Examples::
        >>> raps = RAPS(penalty=0.1, kreg=1, score_type="softmax", randomized=True)
        >>> probs = torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
        >>> scores_all = raps._calculate_all_label(probs)
        >>> print(scores_all)
        >>> scores_single = raps._calculate_single_label(probs, torch.tensor([2, 1]))
        >>> print(scores_single)
        
    """

    def __init__(self, score_type="softmax", randomized=True, penalty=0, kreg=0):

        super().__init__(score_type=score_type, randomized=randomized)
        if penalty < 0:
            raise ValueError("The parameter 'penalty' must be a nonnegative value.")

        if type(kreg) != int or kreg < 0:
            raise TypeError("The parameter 'kreg' must be a nonnegative integer.")
        self.__penalty = penalty
        self.__kreg = kreg

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            U = torch.rand(probs.shape, device=probs.device)
        else:
            U = torch.zeros_like(probs)
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[-1] + 1, device=probs.device) - self.__kreg),
                            torch.tensor(0, device=probs.device))
        ordered_scores = cumsum - ordered * U + reg
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _calculate_single_label(self, probs, label):

        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            U = torch.rand(indices.shape[0], device=probs.device)
        else:
            U = torch.zeros(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        reg = torch.maximum(self.__penalty * (idx[1] + 1 - self.__kreg), torch.tensor(0).to(probs.device))
        scores = cumsum[idx] - U * ordered[idx] + reg
        return scores


class EnergyRAPS(EnergyAPS):

    def __init__(self, score_type="identity", randomized=True, penalty=0, kreg=0, temp_cal=1.0, temp_e=1.0, ent=False, softplus_beta=1.0):

        super().__init__(score_type=score_type, randomized=randomized, temp_cal=temp_cal, temp_e=temp_e, ent=ent, softplus_beta=softplus_beta)
        if penalty < 0:
            raise ValueError("The parameter 'penalty' must be a nonnegative value.")

        if type(kreg) != int or kreg < 0:
            raise TypeError("The parameter 'kreg' must be a nonnegative integer.")
        self.__penalty = penalty
        self.__kreg = kreg
        self.softplus_beta = softplus_beta

    def _calculate_all_label(self, probs):
        energy = torch.logsumexp(probs*self.temp_cal/self.temp_e, dim=-1)
        if self.score_type == "identity":
            probs = torch.softmax(probs, dim=-1) 
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            U = torch.rand(probs.shape, device=probs.device)
        else:
            U = torch.zeros_like(probs)
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[-1] + 1, device=probs.device) - self.__kreg),
                            torch.tensor(0, device=probs.device))
        ordered_scores = cumsum - ordered * U + reg 
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)

        if self.ent:
            entropy = torch.distributions.Categorical(probs).entropy()
            scores = scores / entropy.view(-1, 1)
        else:
            scores = scores * F.softplus(energy, beta=self.softplus_beta).view(-1, 1)
        
        return scores

    def _calculate_single_label(self, probs, label):
        energy = torch.logsumexp(probs*self.temp_cal/self.temp_e, dim=-1)
        if self.score_type == "identity":
            probs = torch.softmax(probs, dim=-1)
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            U = torch.rand(indices.shape[0], device=probs.device)
        else:
            U = torch.zeros(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        reg = torch.maximum(self.__penalty * (idx[1] + 1 - self.__kreg), torch.tensor(0).to(probs.device))
        scores = cumsum[idx] - U * ordered[idx] + reg 
        
        if self.ent:
            entropy = torch.distributions.Categorical(probs).entropy()
            scores = scores / entropy.view(-1)
        else:
            scores = scores * F.softplus(energy, beta=self.softplus_beta).view(-1)
        
        return scores