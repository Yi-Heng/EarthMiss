import torch
import kmeans1d
import time
class CovMatrix_mmr:
    def __init__(self, dim, relax_denom=0, clusters=50):
        super(CovMatrix_mmr, self).__init__()

        self.dim = dim
        self.i = torch.eye(dim, dim).cuda()

     
        self.reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda()


        self.num_off_diagonal = torch.sum(self.reversal_i)
        self.num_sensitive = 0
        self.var_matrix = None
        self.count_var_cov = 0
        self.mask_matrix = None
        self.clusters = clusters
        if relax_denom == 0: 
            self.margin = 0
        else:             
            self.margin = self.num_off_diagonal // relax_denom

    def get_eye_matrix(self):
        return self.i, self.reversal_i

    def get_mask_matrix(self, mask=True):
        if self.mask_matrix is None:
            self.set_mask_matrix()
        return self.i, self.mask_matrix, 0, self.num_sensitive

    def reset_mask_matrix(self):
        self.mask_matrix = None
        
    def reset_var_matrix(self):
        self.var_matrix = None
    
    def set_mask_matrix(self):

        self.var_matrix = self.var_matrix / self.count_var_cov
        var_flatten = torch.flatten(self.var_matrix)

        if self.margin == 0:   
            clusters, centroids = kmeans1d.cluster(var_flatten, self.clusters) 
            num_sensitive = var_flatten.size()[0] - clusters.count(0)  
            _, indices = torch.topk(var_flatten, k=int(num_sensitive))
        else:                   # do not use
            num_sensitive = self.num_off_diagonal - self.margin
            _, indices = torch.topk(var_flatten, k=int(num_sensitive))
        mask_matrix = torch.flatten(torch.zeros(self.dim, self.dim).cuda())
        mask_matrix[indices] = 1

        if self.mask_matrix is not None:
            self.mask_matrix = mask_matrix.view(self.dim, self.dim).float()
        else:
            self.mask_matrix = mask_matrix.view(self.dim, self.dim)
        self.num_sensitive = torch.sum(self.mask_matrix)

        self.var_matrix = None
        self.count_var_cov = 0


    def set_variance_of_covariance(self, var_cov):
        if self.var_matrix is None:
            self.var_matrix = var_cov
        else:
            self.var_matrix = self.var_matrix + var_cov
        self.count_var_cov += 1


def instance_whitening_loss(f_map, eye, mask_matrix, num_remove_cov,margin = 0):
    f_cor, B = get_covariance_matrix(f_map, eye=eye)
    f_cor_masked = f_cor * mask_matrix

    off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1,2), keepdim=True) - margin # B X 1 X 1
    loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
    loss = torch.sum(loss) / B

    return loss


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape  
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)

    return f_cor, B