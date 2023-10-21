Pretraining Scripts
- `run_cavmae_pretrain_scale++.sh`: Pretrain CAV-MAE Scale++ with 256 batch size. Requires largest GPUs (4x48GB).
- `run_cavmae_pretrain_scale+.sh`: Pretrain CAV-MAE Scale+ with 120 batch size. Requires larger GPUs (4x24GB).
- `run_cavmae_pretrain_base.sh`: Pretrain CAV-MAE with 48 batch size. Requires smaller GPUs (4x12GB).

Finetuning Scrips
- `run_cavmae_ft_bal.sh`: Finetune CAV-MAE Scale++ on balanced AS-20K.
- `run_cavmae_ft_bal_audioonly.sh`: Finetune CAV-MAE Scale++ on balanced AS-20K with audio only.
- `run_cavmae_ft_bal_videoonly.sh`: Finetune CAV-MAE Scale++ on balanced AS-20K with visual data only.
- `run_cavmae_ft_full.sh`: Finetune CAV-MAE Scale++ on full AS-2M. 

#SBATCH -p a5
#SBATCH --qos regular
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[1,2],sls-sm-[2,5,6,7,12]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="as-bal-ft"
#SBATCH --output=../log/%j_as_ft.txt