import torch
import random
from scipy.special import betainc

class LAM:
    def __init__(self, mark_bps=128, fpr=1e-6, user_number=1000000, latent_shape=(1,8,256,16), seed=2025):
        self.watermark = None
        self.latent_shape = latent_shape
        self.latent_c, self.latent_h, self.latent_w = latent_shape[1], latent_shape[2], latent_shape[3]
        self.latent_length = self.latent_c * self.latent_h * self.latent_w
        self.mark_bps = mark_bps
        self.times = self.latent_length // (2 * self.mark_bps) 
        self.extra_factor = 1
        self.tp_dec_count = 0
        self.tp_bits_count = 0
        self.tau_dec = None
        self.tau_bits = None
        self.s = seed
        self.device = device

        for i in range(self.mark_bps):
            fpr_dec = betainc(i + 1, self.mark_bps - i, 0.5)
            fpr_bits = fpr_dec * user_number
            if fpr_dec <= fpr and self.tau_dec is None:
                self.tau_dec = (i + 1) / self.mark_bps
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = (i + 1) / self.mark_bps
        
        if self.tau_dec is None: self.tau_dec = 0.6 
        if self.tau_bits is None: self.tau_bits = 0.6

    def shuffle_zm(self, zm):
        flattened = zm.flatten()
        perm = list(range(flattened.numel()))
        rng = random.Random(self.s)
        rng.shuffle(perm)
        perm_tensor = torch.tensor(perm, dtype=torch.long, device=self.device)
        return flattened[perm_tensor]

    def recover_zm(self, zw):
        perm = list(range(zw.numel()))
        rng = random.Random(self.s)
        rng.shuffle(perm)
        perm_tensor = torch.tensor(perm, dtype=torch.long, device=self.device)
        inv_perm = torch.argsort(perm_tensor)
        return zw.flatten()[inv_perm]

    def merge_z(self, zl, zs):
        zm = []
        for i in range(self.times):
            zm.extend(zs[i*self.mark_bps:(i+1)*self.mark_bps])
            zm.extend(zl[i*self.mark_bps:(i+1)*self.mark_bps])
        return torch.tensor(zm, device=self.device)

    @staticmethod
    def avg_group(arr, group_num):
        a = group_num * 2
        o = len(arr)//a
        group = []
        for i in range(group_num):
            tmp = []
            tmp.extend(arr[i*o:(i+1)*o])
            tmp.extend(arr[(a-i-1)*o:(a-i)*o])
            group.append(tmp)
        return group

    def create_watermark_and_return_w(self, device='cuda', dtype=torch.float16):
        self.device = device
        random_bits = [0]*(self.mark_bps//2) + [1]*(self.mark_bps//2)
        rng = random.Random(self.s + 1) 
        rng.shuffle(random_bits)
        self.watermark = ''.join(map(str, random_bits))

        X = torch.randn(self.latent_shape, device=device, dtype=dtype)
        N = X[X<0]
        P = X[X>=0]

        use_len = self.latent_length//4
        N1 = N[torch.topk(N, use_len, largest=False)[1].sort()[0]]
        P1 = P[torch.topk(P, use_len, largest=True)[1].sort()[0]]

        zl = []
        n_idx, p_idx = 0, 0
        bin_str = self.watermark * self.times
        for bit in bin_str:
            if bit=='0':
                zl.append(N1[n_idx].item())
                n_idx+=1
            else:
                zl.append(P1[p_idx].item())
                p_idx+=1

        rem_N = N[~torch.isin(torch.arange(len(N), device=device), torch.topk(N, use_len, largest=False)[1])]
        rem_P = P[~torch.isin(torch.arange(len(P), device=device), torch.topk(P, use_len, largest=True)[1])]
        
        remain = torch.cat([rem_N, rem_P]).tolist()
        mid = len(remain)//2
        Rn = sorted(remain[:mid])
        Rp = sorted(remain[mid:])
        Gn = self.avg_group(Rn, self.mark_bps//2)
        Gp = self.avg_group(Rp, self.mark_bps//2)

        zs = []
        n_idx, p_idx = 0, 0
        for k in self.watermark:
            if k=='0':
                zs.extend(Gn[n_idx])
                n_idx+=1
            else:
                zs.extend(Gp[p_idx])
                p_idx+=1

        zm = self.merge_z(zl, zs)
        target_len = self.latent_length
        if zm.numel() > target_len:
            zm = zm[:target_len]
        elif zm.numel() < target_len:
            pad_len = target_len - zm.numel()
            zm = torch.cat([zm, torch.zeros(pad_len, device=device)])

        zw = self.shuffle_zm(zm)
        return zw.reshape(self.latent_shape).to(device).to(dtype)

    def eval_watermark(self, zw):
        zm = self.recover_zm(zw) 
        zm = zm.tolist() 
        len_group = 2 * self.mark_bps
        valid_len = (len(zm) // len_group) * len_group
        zm = zm[:valid_len]
        group_zm = [zm[i:i+len_group] for i in range(0, len(zm), len_group)]
        
        zs_list = [group[:self.mark_bps] for group in group_zm]
        zl_list = [group[self.mark_bps:] for group in group_zm]

        zs = []
        for segment in zs_list:
            zs.extend([sum(segment[i:i+self.times]) for i in range(0, len(segment), self.times)])

        zl_w = ''.join(['0' if val<0 else '1' for sub in zl_list for val in sub])
        zs_w = ''.join(['0' if val<0 else '1' for val in zs])
        
        zw_all = zl_w + zs_w
            
        zw_list = [zw_all[i:i+self.mark_bps] for i in range(0, len(zw_all), self.mark_bps)]
        
        if not zw_list: return 0.0
        if len(zw_list[-1]) < self.mark_bps:
            zw_list[-1] = zw_list[-1] + '0'*(self.mark_bps - len(zw_list[-1]))
            
        zw_final = ''
        for i in range(self.mark_bps):
            cor_1 = sum(segment[i]=='1' for segment in zw_list)
            zw_final += '1' if cor_1 > len(zw_list)//2 else '0'

        correct_bits = sum(1 for x,y in zip(zw_final, self.watermark) if x==y)
        zw_correct = correct_bits / self.mark_bps
        
        #if zw_correct < 0.35:
        zw_correct = 1.0 - zw_correct
            
        return zw_correct