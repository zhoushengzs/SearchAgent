# Candidate environment only. The tag/digest exists; this build has not been run.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6
RUN /opt/conda/bin/conda create -y -n internvl3-sft python=3.11.13
ENV PATH=/opt/conda/envs/internvl3-sft/bin:/opt/conda/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/starter
COPY training_examples/internvl3_14b/ /opt/starter/
COPY train/OpenSearch-VL-Training/SFT/ /opt/SFT/
RUN python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
RUN DS_BUILD_OPS=0 python -m pip install -c constraints.txt -e /opt/SFT deepspeed==0.18.8
RUN python -m pip check && python -m pip freeze > /opt/starter/installed-freeze.txt
ENV SFT_WORK=/work
CMD ["bash"]
