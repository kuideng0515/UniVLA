## Install
```bash
conda create -n lams python==3.10 -y
onda activate lams
pip install torch
pip install -r requirements.txt
# test data loading
cd ./latent_action_model && python -m dataloader.lam_datamodule && cd ..
```

## Train
```bash
bash train_lam.sh 
```
