
from scripts.diffusion import Diffusion_model
import os
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = REPO_ROOT / 'fgsm_collection_dataset'
DEFAULT_BENIGN_CSV = DEFAULT_DATASET_DIR / 'fgsm_collection_fgsm015_benign_obs_data.csv'
DEFAULT_ADVERSARIAL_CSV = DEFAULT_DATASET_DIR / 'fgsm_collection_fgsm015_adversarial_obs_data.csv'

def main(args):
   
    experiment_name = args.experiment_name
    #Create experiment directory
    Path('./Experiments/'+experiment_name).mkdir(parents=True, exist_ok=True)
    #Use debug to stop after running 10 samples, for debugging purposes
    diff_model = Diffusion_model(
        experiment_name,
        debug=args.debug,
        inference_mode=args.mode,
        inference_steps=args.inference_steps,
        renoise_strength=args.renoise_strength,
        benign_csv=args.benign_csv,
        adversarial_csv=args.adversarial_csv,
    )
    #Check experiment progress and load
    print('##############################')
    if os.path.exists(diff_model.checkpoint_path):
        print("Previous checkpoint from experiment "+experiment_name+" found in: "+str(diff_model.checkpoint_path))
        print("Loading model!")
        diff_model.load_params()
        print("Loading complete!")
    else:
        print("No checkpoint found for experiment, beginning fresh training...")
    print('##############################')

    # Start and run the training loop
    if(args.pretrain_ddpm):
        diff_model.run()
    if(args.sample):
        diff_model.origin_sampling(999,starting_t=10,eval=True,subset_factor=1000, samples=1)


#
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training and Testing for DDPM diffusion model')
    parser.add_argument('--pretrain_ddpm', action='store_true', default=True,
                        help='Performs DDPM pretraining')
    parser.add_argument('--experiment_name', type=str, default='diffusion_defense_2', #diffusion_defense_1
                        help='Name of experiment to load/save to.')
    parser.add_argument('--sample',action='store_true', default=False,
                        help='Performs origin sampling on model')
    parser.add_argument('--debug',action='store_true', default=False,
                        help='runs model in debug mode (each epoch will run only 1 batch to verify process)')
    parser.add_argument('--mode', type=str, default='stochastic_light',
                        choices=['deterministic', 'stochastic_light', 'stochastic_heavy'],
                        help='Runtime inference mode to store with the DDPM defense.')
    parser.add_argument('--inference_steps', type=int, default=3,
                        help='Number of denoising steps to use at inference time.')
    parser.add_argument('--renoise_strength', type=float, default=1.0,
                        help='Strength of re-noising during stochastic inference.')
    parser.add_argument('--benign_csv', type=str, default=str(DEFAULT_BENIGN_CSV),
                        help='Benign CSV path.')
    parser.add_argument('--adversarial_csv', type=str, default=str(DEFAULT_ADVERSARIAL_CSV),
                        help='Adversarial CSV path.')
    args = parser.parse_args()

    main(args)
