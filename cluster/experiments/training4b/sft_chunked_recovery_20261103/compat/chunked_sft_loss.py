import torch
import torch.nn.functional as F
CHUNK=256
class ChunkedCE(torch.autograd.Function):
 @staticmethod
 def forward(ctx, logits, targets, denominator, ignore_index):
  ctx.save_for_backward(logits,targets,denominator);ctx.ignore_index=ignore_index
  total=torch.zeros((),device=logits.device,dtype=torch.float32)
  for i in range(0,logits.shape[0],CHUNK):
   total+=F.cross_entropy(logits[i:i+CHUNK].float(),targets[i:i+CHUNK],ignore_index=ignore_index,reduction='sum')
  return total/denominator
 @staticmethod
 def backward(ctx, upstream):
  logits,targets,denominator=ctx.saved_tensors
  grad=torch.empty_like(logits)
  for i in range(0,logits.shape[0],CHUNK):
   y=targets[i:i+CHUNK];valid=y!=ctx.ignore_index
   p=torch.softmax(logits[i:i+CHUNK].float(),dim=-1)
   rows=torch.arange(y.numel(),device=y.device)[valid]
   p[rows,y[valid]]-=1
   p[~valid]=0
   if denominator.item()!=0:p.mul_(upstream/denominator)
   else:p.zero_()
   grad[i:i+CHUNK].copy_(p)
  return grad,None,None,None

def chunked_causal_loss(logits,labels,vocab_size,num_items_in_batch=None,ignore_index=-100,shift_labels=None,**kwargs):
 if shift_labels is None:shift_labels=F.pad(labels,(0,1),value=ignore_index)[...,1:].contiguous()
 x=logits.reshape(-1,vocab_size);y=shift_labels.reshape(-1).to(x.device)
 den=(y!=ignore_index).sum() if num_items_in_batch is None else num_items_in_batch
 den=torch.as_tensor(den,device=x.device,dtype=torch.float32)
 return ChunkedCE.apply(x,y,den,ignore_index)

def install():
 import transformers.loss.loss_utils as u
 u.ForCausalLMLoss=chunked_causal_loss
 u.LOSS_MAPPING['ForCausalLM']=chunked_causal_loss
