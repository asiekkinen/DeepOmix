import torch
import torch.nn as nn

class DeepOmixNet(nn.Module):
	def __init__(self, In_Nodes, Pathway_Nodes, Hidden_Nodes,Out_Nodes, Pathway_Mask):
		super(DeepOmixNet, self).__init__()
		self.tanh = nn.Tanh()
		self.pathway_mask = Pathway_Mask
		self.sc1 = nn.Linear(In_Nodes, Pathway_Nodes)
		self.sc2 = nn.Linear(Pathway_Nodes, Hidden_Nodes)
		self.sc3 = nn.Linear(Hidden_Nodes, Out_Nodes, bias=False)
		self.sc4 = nn.Linear(Out_Nodes+1,1, bias = False)
		self.sc4.weight.data.uniform_(-0.001, 0.001)
		self.do_m1 = torch.ones(Pathway_Nodes)
		self.do_m2 = torch.ones(Hidden_Nodes)
		###if gpu is being used
		if torch.cuda.is_available():
			self.do_m1 = self.do_m1.cuda()
			self.do_m2 = self.do_m2.cuda()

	def forward(self, x_1, x_2):
		self.sc1.weight.data = self.sc1.weight.data.mul(self.pathway_mask)
		functional_output = self.sc1(x_1)
		x_1 = self.tanh(functional_output)
		if self.training == True: 
			x_1 = x_1.mul(self.do_m1)
		x_1 = self.tanh(self.sc2(x_1))
		if self.training == True: 
			x_1 = x_1.mul(self.do_m2)
		x_1 = self.tanh(self.sc3(x_1))
		x_cat = torch.cat((x_1, x_2), 1)
		lin_pred = self.sc4(x_cat)
		
		return lin_pred, functional_output

