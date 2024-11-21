import torch
from jax import numpy as jnp


def SNNavg_aggregation(client_models, global_model):
    """
    通过FedAvg聚合多个服务器的模型参数，更新全局模型参数，并同步更新每个服务器的模型
    
    参数:
    global_model: CSDP_SNN 全局模型对象
    server_models: List[CSDP_SNN] 包含多个服务器模型的列表
    
    返回:
    global_model: 更新后的全局模型
    server_models: List[CSDP_SNN] 更新后的服务器模型列表
    """
    
    num_servers = len(client_models)  # 获取服务器的数量
    
    # 遍历每一个可以保存的组件（如权重矩阵等）
    for comp_name in global_model.saveable_comps:
        # 获取全局模型中对应组件的引用
        global_comp = global_model.circuit.components.get(comp_name)
        
        # 检查组件是否有 weights 属性
        if hasattr(global_comp, 'weights'):
            # 初始化一个与全局模型参数同形状的零矩阵
            aggregated_params = jnp.zeros_like(global_comp.weights.value)
            
            # 对每个服务器的模型参数进行累加
            for server_model in client_models:
                server_comp = server_model.circuit.components.get(comp_name)
                aggregated_params += server_comp.weights.value
            
            # 取平均值，并更新到全局模型中
            aggregated_params /= num_servers
            global_comp.weights.set(aggregated_params)
            
            # 用聚合后的全局参数更新每个服务器的模型参数
            for server_model in client_models:
                server_comp = server_model.circuit.components.get(comp_name)
                server_comp.weights.set(aggregated_params)
    
    print("Finish aggregation, update the global model and local model!\n")
    
    return client_models, global_model



def FedAvg(Userlist, Global_model):
    """
    :param w: the list of user
    :return: the userlist after aggregated and the global mode
    """


    l_user = len(Userlist)    # the number of user

    client_weights = [1/l_user for i in range(l_user)]
    with torch.no_grad():
        for key in Global_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                Global_model.state_dict()[key].data.copy_(
                    Userlist[0].state_dict()[key])
            else:
                temp = torch.zeros_like(
                    Global_model.state_dict()[key], dtype=torch.float32)

                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * \
                        Userlist[client_idx].state_dict()[
                        key]

                Global_model.state_dict()[
                    key].data.copy_(temp)

                for client_idx in range(len(client_weights)):
                    Userlist[client_idx].state_dict()[key].data.copy_(
                        Global_model.state_dict()[key])
    return Userlist, Global_model