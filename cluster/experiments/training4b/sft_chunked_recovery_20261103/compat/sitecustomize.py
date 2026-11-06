import os
if os.environ.get("SFT_CHUNKED_LOSS")=="1":
 from chunked_sft_loss import install
 install()
