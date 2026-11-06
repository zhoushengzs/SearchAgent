import torch,json
from transformers.loss.loss_utils import ForCausalLMLoss
from chunked_sft_loss import chunked_causal_loss
checks=[]
for dtype in [torch.float32,torch.bfloat16]:
 for shifted in [False,True]:
  for normalized in [False,True]:
   torch.manual_seed(42);x=torch.randn(2,519,71,dtype=dtype,requires_grad=True);z=x.detach().clone().requires_grad_(True)
   y=torch.randint(0,71,(2,519));y[:,::7]=-100
   kwargs={'shift_labels':y} if shifted else {}
   if normalized:kwargs['num_items_in_batch']=torch.tensor(1500)
   a=ForCausalLMLoss(x,y,71,**kwargs);b=chunked_causal_loss(z,y,71,**kwargs)
   (a*1.7).backward();(b*1.7).backward()
   torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)
   torch.testing.assert_close(x.grad,z.grad,rtol=0.01 if dtype==torch.bfloat16 else 1e-5,atol=2e-5 if dtype==torch.bfloat16 else 1e-7)
   checks.append({'dtype':str(dtype),'shifted':shifted,'normalized':normalized,'loss_abs_diff':float((a-b).abs()),'grad_max_abs_diff':float((x.grad-z.grad).abs().max())})
print(json.dumps({'passed':True,'checks':checks}))
