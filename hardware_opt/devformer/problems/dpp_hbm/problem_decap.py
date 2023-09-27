from torch.utils.data import Dataset
import torch
import os
import pickle5 as pickle
from tqdm import tqdm
import numpy as np
from problems.dpp_hbm.state_decap import StateDecap
from problems.dpp_hbm.reward_function_serial import reward_gen


class Decap_hbm(object):
    NAME = "dpp_hbm"

    @staticmethod
    def get_costs(dataset, pi, raw_pdn=None, z_init_list=None, mask=None):
        # Check that tours are valid, i.e. contain 0 to n -1

        # Gather dataset in order of tour

        reward_list = []

        for i in tqdm(range(pi.size(0))):
            #             trial[0] = pi[i].cpu().numpy()

            reward = reward_gen(
                (dataset[i, :, 2] == 2).nonzero().item(), pi[i].cpu().numpy(), 5
            )

            reward_list.append(reward)

        # reward = reward_gen((dataset[:,:,2]==2).nonzero()[:,0], pi.cpu().numpy(),5, raw_pdn,z_init_list,mask)
        #         assert(False)
        #         reward_list.append(reward)

        reward = torch.FloatTensor(reward_list).cuda().view(-1, 1)

        # virtual reward
        # d = dataset.gather(1, pi.unsqueeze(-1))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return -reward, None
        # return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return DecapDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateDecap.initialize(*args, **kwargs)


class DecapDataset(Dataset):
    def __init__(
        self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None,decap_x = 15, decap_y = 20
    ):
        super(DecapDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"

            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.data = [
                    torch.FloatTensor(row)
                    for row in (data[offset : offset + num_samples])
                ]
        else:

            probing_port = range(0,decap_x*decap_y)
            num_data = int(num_samples/len(probing_port))
            dpp = np.zeros((num_samples,decap_x*decap_y, 3))
            normalization = max(decap_x,decap_y)
            for j in range(num_data):
                for i in range(len(probing_port)):
                    for x in range(decap_x):
                        num_restriction = 0
                        for y in range(decap_y):
                            dpp[i+j*decap_x*decap_y,x*decap_y+y, 0] = x/normalization
                            dpp[i+j*decap_x*decap_y,x*decap_y+y,1] = y/normalization
                            if np.random.uniform(0,1)>0.9 and num_restriction < 15:                        
                                dpp[i+j*decap_x*decap_y,x*decap_y+y,2] = 1
                                num_restriction += 1

                            if x*decap_y+y ==probing_port[i]:             
                                dpp[i+j*decap_x*decap_y,x*decap_y+y,2] = 2         


            self.data = [torch.from_numpy(dpp[i]).float() for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]