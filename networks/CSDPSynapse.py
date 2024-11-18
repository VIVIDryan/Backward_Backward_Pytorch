#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : CSDPSynapse.py
# Author            : TravisChen <chenxuqiangwork@outlook.com>
# Date              : 15.10.2024
# Last Modified Date: 15.10.2024
# Last Modified By  : TravisChen <chenxuqiangwork@outlook.com>
### Synapse of the SNN
from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats

class CSDPSynapse(DenseSynapse):

    def __init__(self, name, shape, eta=0., weight_init=None, bias_init=None,
                 w_bound=1., is_nonnegative=False, w_decay=0., update_sign=1.,
                 w_sign=1., is_hollow=False, soft_bound=False, gamma_depress=0.,
                 optim_type="sgd", p_conn=1., resist_scale=1., batch_size=1, **kwargs):
        super().__init__(name, shape, weight_init, bias_init, resist_scale,
                         p_conn, batch_size=batch_size, **kwargs)

        ## synaptic plasticity properties and characteristics
        self.shape = shape
        ## 电阻的缩放因子，影响突出输出的强度
        self.resist_scale = resist_scale
        ## 限制突触更新的范围
        self.w_bound = w_bound

        self.w_sign = w_sign
        self.w_decay = w_decay ## synaptic decay
        ## TODO
        ## 学习率，突触更新的速度，
        self.eta = eta
        ## 添加突触抑制
        self.gamma_depress = gamma_depress
        self.is_nonnegative = is_nonnegative
        self.is_hollow = is_hollow
        self.soft_bound = soft_bound
        self.update_sign = update_sign

        ## optimization / adjustment properties (given learning dynamics above)
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)

        ## 阻止神经元自连接，通过掩码确保侧向链接没有自连接
        self.weightMask = 1.
        if self.is_hollow:
            self.weightMask = 1. - jnp.eye(N=shape[0], M=shape[1])

        # compartments (state of the cell, parameters, will be updated through stateless calls)
        ## 前突和后突的脉冲和痕迹，用于在前向传播中存储突触状态的容器
        self.preVals = jnp.zeros((self.batch_size, shape[0]))
        self.postVals = jnp.zeros((self.batch_size, shape[1]))
        self.preSpike = Compartment(self.preVals)
        self.postSpike = Compartment(self.postVals)
        self.preTrace = Compartment(self.preVals)
        self.postTrace = Compartment(self.postVals)
        self.dWeights = Compartment(jnp.zeros(shape))
        self.dBiases = Compartment(jnp.zeros(shape[1]))

        #key, subkey = random.split(self.key.value)
        self.opt_params = Compartment(get_opt_init_fn(optim_type)(
            [self.weights.value, self.biases.value]
            if bias_init else [self.weights.value]))

    ## 前突脉冲、突触权重和偏差来计算突触的输出。它通过矩阵乘法来计算后突神经元的脉冲活动
    @staticmethod
    def _advance_state(resist_scale, w_sign, inputs, weights, biases):
        factor = w_sign * resist_scale
        outputs = (jnp.matmul(inputs, weights) * factor) + biases
        return outputs

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

    ## 脉冲时间依赖性可塑性（STDP）规则来计算突触的更新值。它根据前突和后突神经元的脉冲和痕迹来计算权重更新
    @staticmethod
    def _compute_update(w_bound, update_sign, w_decay, soft_bound, gamma_depress,
                        eta, preSpike, postSpike, preTrace, postTrace, weights,
                        biases):
        ## calculate synaptic update values
        if eta > 0.:
            dW = jnp.matmul(preSpike.T, postTrace) ## pre-syn-driven STDP product
            if gamma_depress > 0.: ## add in post-syn driven counter-term
                dW = dW - jnp.matmul(preTrace.T, postSpike) * gamma_depress
            if soft_bound:
                dW = dW * (w_bound - jnp.abs(weights))
            ## FIXME / NOTE: fix decay to be abs value of synaptic weights
            ## compute post-syn decay factor
            Wdecay_factor = -(jnp.matmul((1. - preSpike).T, (postSpike)) * w_decay)
            db = jnp.sum(postTrace, axis=0, keepdims=True)
        else:
            dW = weights * 0
            Wdecay_factor = dW
            db = biases * 0
        return dW * update_sign, db * update_sign, Wdecay_factor


    ## 负责应用学习规则更新突触的权重和偏差，同时保证突触更新后的权重保持在指定的范围内。
    ## 该函数会根据计算出的梯度更新权重，并使用优化器进行一步更新，同时应用权重衰减和约束（如非负性或空洞性）。
    @staticmethod
    def _evolve(opt, soft_bound, w_bound, resist_scale, is_nonnegative, update_sign,
                w_decay, bias_init, gamma_depress, is_hollow, eta,
                preSpike, postSpike, preTrace, postTrace, weights, biases, weightMask,
                opt_params):
        d_z = postTrace * resist_scale # 0.1 ## get modulated post-synaptic trace
        ## calculate synaptic update values
        dWeights, dBiases, weightDecay = CSDPSynapse._compute_update(
            w_bound, update_sign, w_decay, soft_bound, gamma_depress, eta,
            preSpike, postSpike, preTrace, d_z, weights, biases
        )
        ## conduct a step of optimization - get newly evolved synaptic weight value matrix
        if bias_init != None:
            opt_params, [weights, biases] = opt(opt_params, [weights, biases], [dWeights, dBiases])
        else:
            # ignore db since no biases configured
            opt_params, [weights] = opt(opt_params, [weights], [dWeights])
        ## apply decay to synapses and enforce any constraints
        weights = weights + weightDecay
        if w_bound > 0.:
            if is_nonnegative:
                weights = jnp.clip(weights, 0., w_bound)
            else:
                weights = jnp.clip(weights, -w_bound, w_bound)
        if is_hollow: ## enforce lateral hollow matrix masking
            weights = weights * weightMask

        return opt_params, weights, biases, dWeights, dBiases

    @resolver(_evolve)
    def evolve(self, opt_params, weights, biases, dWeights, dBiases):
        self.opt_params.set(opt_params)
        self.weights.set(weights)
        self.biases.set(biases)
        self.dWeights.set(dWeights)
        self.dBiases.set(dBiases)
    ## 重置突触的状态，将所有值（脉冲、痕迹、梯度）重置为零。这个函数在每一批数据训练开始时会被调用
    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        return (
            preVals, # inputs
            postVals, # outputs
            preVals, # pre
            postVals, # post
            jnp.zeros(shape), # dW
            jnp.zeros(shape[1]), # db
        )

    @resolver(_reset)
    def reset(self, inputs, outputs, preSpike, postSpike, preTrace, postTrace,
              dWeights, dBiases):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.preSpike.set(preSpike)
        self.postSpike.set(postSpike)
        self.preTrace.set(preTrace)
        self.postTrace.set(postTrace)
        self.dWeights.set(dWeights)
        self.dBiases.set(dBiases)

    ## 这是一个实用函数，用于打印突触状态的摘要，包括各个容器（如权重和脉冲）的统计信息（如均值、标准差）。
    def __repr__(self):
        comps = [varname for varname in dir(self) if Compartment.is_compartment(getattr(self, varname))]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines
