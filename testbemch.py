import numpy as np
import torch


doc_embeds = torch.rand(50, 32)
query_embed = torch.rand(1, 32)

# normalised:
norm_doc = doc_embeds.norm(dim=1, keepdim=True)
norm_query = query_embed.norm()

dot_sim = torch.mm(doc_embeds, query_embed.T)
cos_sim = dot_sim / (norm_doc * norm_query)
euc_sim = torch.norm(doc_embeds - query_embed, dim=1)

print(f"{dot_sim=}({dot_sim.shape})\n{cos_sim=}({cos_sim.shape})\n{euc_sim=}({euc_sim.shape})")
