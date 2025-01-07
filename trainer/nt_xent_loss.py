import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature):
    """
    compute the contrastive loss between two arrays

    :param z1: the first input array, with torch.Size([batch_size , projection_dim]) 
    :param z2: the second input array, with torch.Size([batch_size , projection_dim]) 

    return loss
    """
    # Normalize the representations.
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)  

    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t())
    
    # Exponentiate the similarity matrix and mask out the self-similarity
    exp_sim_matrix = torch.exp(sim_matrix / temperature)
    mask = torch.eye(2 * z1.size(0), device=z.device).bool()
    exp_sim_matrix = exp_sim_matrix.masked_fill(mask, 0)

    # # Extract the positive pairs and scale by batch size
    pos_sim_1 = torch.diag(exp_sim_matrix, z1.size(0))
    pos_sim_2 = torch.diag(exp_sim_matrix, -z1.size(0))
    pos_sim = torch.cat([pos_sim_1, pos_sim_2], dim=0)

    # Compute the NT-Xent loss for each example
    sum_exp_sim = torch.sum(exp_sim_matrix, dim=1)
    loss = -torch.log(pos_sim / sum_exp_sim)

    return loss.mean()