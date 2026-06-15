
from scripts.diffusion import Diffusion_model
import os
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = REPO_ROOT / 'fgsm_collection_dataset'
DEFAULT_BENIGN_CSV = DEFAULT_DATASET_DIR / 'fgsm_collection_fgsm015_benign_obs_data.csv'
DEFAULT_ADVERSARIAL_CSV = DEFAULT_DATASET_DIR / 'fgsm_collection_fgsm015_adversarial_obs_data.csv'

def print_inference_speed(diff_model, args):
    metrics = diff_model.benchmark_inference_speed(
        batch_size=args.speed_test_batch_size,
        warmup=args.speed_test_warmup,
        iterations=args.speed_test_iterations,
    )
    print("DDPM inference speed test:")
    print(f"  mode: {diff_model.inference_mode}")
    print(f"  configured inference steps: {diff_model.inference_steps}")
    print(f"  timed inference steps: {metrics['inference_steps']}")
    print(f"  starting t: {metrics['starting_t']}")
    print(f"  uses origin_q_sample: {metrics['uses_origin_q_sample']}")
    print(f"  renoise strength: {diff_model.renoise_strength}")
    print(f"  device: {diff_model.device}")
    print(f"  batch size: {metrics['batch_size']}")
    print(f"  warmup iterations: {metrics['warmup_iterations']}")
    print(f"  timed iterations: {metrics['timed_iterations']}")
    print(f"  ms per batch: {metrics['ms_per_batch']:.3f}")
    print(f"  ms per sample: {metrics['ms_per_sample']:.3f}")
    print(f"  samples per second: {metrics['samples_per_second']:.2f}")


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
    checkpoint_loaded = False
    if os.path.exists(diff_model.checkpoint_path):
        print("Previous checkpoint from experiment "+experiment_name+" found in: "+str(diff_model.checkpoint_path))
        print("Loading model!")
        diff_model.load_params()
        checkpoint_loaded = True
        print("Loading complete!")
    else:
        print("No checkpoint found for experiment, beginning fresh training...")
    print('##############################')

    if args.inference_speed_only:
        if checkpoint_loaded:
            print("Running loaded-model inference speed check only.")
        else:
            print("Running inference speed check only on a freshly initialized model; no checkpoint was found.")
        print_inference_speed(diff_model, args)
        return

    # Start and run the training loop
    if(args.pretrain_ddpm):
        print("Running pre-training inference speed check...")
        print_inference_speed(diff_model, args)
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
    parser.add_argument('--inference_speed_only', action='store_true', default=False,
                        help='Load the configured DDPM experiment, print inference speed, and exit without training.')
    parser.add_argument('--speed_test_batch_size', type=int, default=1,
                        help='Batch size for DDPM inference speed checks.')
    parser.add_argument('--speed_test_warmup', type=int, default=5,
                        help='Warmup iterations for DDPM inference speed checks.')
    parser.add_argument('--speed_test_iterations', type=int, default=50,
                        help='Timed iterations for DDPM inference speed checks.')
    parser.add_argument('--benign_csv', type=str, default=str(DEFAULT_BENIGN_CSV),
                        help='Benign CSV path.')
    parser.add_argument('--adversarial_csv', type=str, default=str(DEFAULT_ADVERSARIAL_CSV),
                        help='Adversarial CSV path.')
    args = parser.parse_args()

    main(args)
