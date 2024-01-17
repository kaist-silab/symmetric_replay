import torch
from rdkit import Chem


def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles

def smiles_to_seq(smiles, voc):
    tokenized = voc.tokenize(smiles)
    seq = []
    for char in tokenized:
        seq.append(voc.vocab[char])
    return torch.tensor(seq).float().cuda()


def make_symmetric_trj(seqs, voc, do_random=False):
    padded_len = seqs.shape[1] +5
    smiles_list = seq_to_smiles(seqs, voc)

    symmetric_seq_list = []
    for i, smi in enumerate(smiles_list):

        mol = Chem.MolFromSmiles(smi)
        new_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, doRandom=do_random)
        # print(smi, new_smiles)
        
        try:
            new_seq = smiles_to_seq(new_smiles, voc)
            # print(len(new_seq), padded_len)
            new_seq = torch.cat([new_seq, torch.zeros(padded_len-len(new_seq)).cuda()], dim=0)
        except:
            new_seq = seqs[i]

        symmetric_seq_list.append(new_seq)

    return torch.stack(symmetric_seq_list)
