from DeepOmix import DeepOmixNet
from SubNetwork import dropout_mask, s_mask
from Survival import R_set, neg_par_log_likelihood, c_index
import torch
import torch.optim as optim
import copy
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import math
dtype = torch.FloatTensor

def trainDeepOmixNet_without(train_x, train_ytime, train_yevent, \
			eval_x, eval_ytime, eval_yevent, pathway_mask, \
			In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, \
			Learning_Rate, L2, Num_Epochs, Dropout_Rate):
	print('there are ',Num_Epochs,'epochs in the training process!!')
	net = DeepOmixNet(In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, pathway_mask)
	###if gpu is being used
	if torch.cuda.is_available():
		net.cuda()
	opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay = L2)
	for epoch in range(Num_Epochs+1):
		net.train()
		opt.zero_grad() 
		net.do_m1 = dropout_mask(Pathway_Nodes, Dropout_Rate[0])
		net.do_m2 = dropout_mask(Hidden_Nodes, Dropout_Rate[1])

		pred, _ = net(train_x, train_yevent)
		loss = neg_par_log_likelihood(pred, train_ytime, train_yevent) 
		loss.backward()
		opt.step()

		net.sc1.weight.data = net.sc1.weight.data.mul(net.pathway_mask)

		do_m1_grad = copy.deepcopy(net.sc2.weight._grad.data)
		do_m2_grad = copy.deepcopy(net.sc3.weight._grad.data)
		do_m1_grad_mask = torch.where(do_m1_grad == 0, do_m1_grad, torch.ones_like(do_m1_grad))
		do_m2_grad_mask = torch.where(do_m2_grad == 0, do_m2_grad, torch.ones_like(do_m2_grad))

		net_sc2_weight = copy.deepcopy(net.sc2.weight.data)
		net_sc3_weight = copy.deepcopy(net.sc3.weight.data)

		net_state_dict = net.state_dict()

		copy_net = copy.deepcopy(net)
		copy_state_dict = copy_net.state_dict()
		for name, param in copy_state_dict.items():

			if not "weight" in name:
				continue
			if "sc1" in name:
				continue
			if "sc4" in name:
				break
			if "sc2" in name:
				active_param = net_sc2_weight.mul(do_m1_grad_mask)
			if "sc3" in name:
				active_param = net_sc3_weight.mul(do_m2_grad_mask)
			nonzero_param_1d = active_param[active_param != 0]
			if nonzero_param_1d.size(0) == 0: 
				break
			copy_param_1d = copy.deepcopy(nonzero_param_1d)
			S_set =  torch.arange(100, -1, -1)[1:]
			copy_param = copy.deepcopy(active_param)
			S_loss = []
			for S in S_set:
				param_mask = s_mask(sparse_level = S.item(), param_matrix = copy_param, nonzero_param_1D = copy_param_1d, dtype = dtype)
				transformed_param = copy_param.mul(param_mask)
				copy_state_dict[name].copy_(transformed_param)
				copy_net.train()
				y_tmp, _ = copy_net(train_x, train_yevent)
				loss_tmp = neg_par_log_likelihood(y_tmp, train_ytime, train_yevent)
				S_loss.append(loss_tmp)
			interp_S_loss = interp1d(S_set, S_loss, kind='cubic')
			interp_S_set = torch.linspace(min(S_set), max(S_set), steps=100)
			interp_loss = interp_S_loss(interp_S_set)
			optimal_S = interp_S_set[np.argmin(interp_loss)]
			optimal_param_mask = s_mask(sparse_level = optimal_S.item(), param_matrix = copy_param, nonzero_param_1D = copy_param_1d, dtype = dtype)
			if "sc2" in name:
				final_optimal_param_mask = torch.where(do_m1_grad_mask == 0, torch.ones_like(do_m1_grad_mask), optimal_param_mask)
				optimal_transformed_param = net_sc2_weight.mul(final_optimal_param_mask)
			if "sc3" in name:
				final_optimal_param_mask = torch.where(do_m2_grad_mask == 0, torch.ones_like(do_m2_grad_mask), optimal_param_mask)
				optimal_transformed_param = net_sc3_weight.mul(final_optimal_param_mask)
			copy_state_dict[name].copy_(optimal_transformed_param)
			net_state_dict[name].copy_(optimal_transformed_param)

		if epoch % 20 == 0: 
			net.train()
			train_pred, _ = net(train_x, train_yevent)
			train_loss = neg_par_log_likelihood(train_pred, train_ytime, train_yevent).view(1,).item()

			net.eval()
			eval_pred, _ = net(eval_x, eval_yevent)
			eval_loss = neg_par_log_likelihood(eval_pred, eval_ytime, eval_yevent).view(1,).item()

			train_cindex = c_index(train_pred, train_ytime, train_yevent)
			eval_cindex = c_index(eval_pred, eval_ytime, eval_yevent)
			print("The ",epoch,"th epoch : Loss in Train: ", train_loss, " Loss in Val: ", eval_loss)
			if (math.isnan(train_loss)):
				print(epoch,train_loss,"end_train")
				break
			torch.save(net.state_dict(), "{}.pt".format(epoch))

	return (train_loss, eval_loss, train_cindex, eval_cindex)
