"""
Compare robustness of wm model with wm models trained with contrastive learning
"""
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm

from data.lj_dataset import LjAudioDataset
from distortions.attacks import (delete_samples, mp3_compression,
                                 pcm_bit_depth_conversion, resample)
from model.utils import (accuracy, get_model_average_pesq_dataset,
                         get_model_average_stoi_dataset, load_from_ckpt,
                         truncate_or_pad_np)

warnings.filterwarnings(
    "ignore", message=".*TorchCodec's decoder.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def evaluate(embedder, decoder, dataset, attack_fn, attack_param, process_config, train_config):
    """
    Evaluate wm model performance on attack_fn with attack_param
    """
    embedder.eval()
    decoder.eval()

    total_acc = 0
    total_bits = 0

    with torch.no_grad():
        tbar = tqdm(dataset, total=len(dataset), desc=f"Attack {attack_fn.__name__}, param: {attack_param}")
        for item in tbar:
            wav = item["wav"].unsqueeze(0).to(device)
            curr_bs = wav.shape[0]
            msg = (
                torch.randint(
                    0, 2, (curr_bs, 1, train_config["watermark"]["length"]), device=device
                ).float()
                * 2
                - 1
            ).to(device)

            # --- Step 2: embed watermark ---
            embedded, _ = embedder(wav, msg)  # embedded: [1, T]

            # --- Step 3: attack on embedded signal ---
            embedded_np = embedded.squeeze(0).cpu().numpy().flatten()

            if attack_fn.__name__ == "delete_samples":
                attacked_np = attack_fn(embedded_np, percentage=attack_param)
            elif attack_fn.__name__ == "resample":
                attacked_np = attack_fn(embedded_np, downsample_sr=attack_param, sr=process_config["audio"]["sample_rate"])
            elif attack_fn.__name__ == "mp3_compression":
                attacked_np = attack_fn(embedded_np, sr=process_config["audio"]["sample_rate"], quality=attack_param)
            elif attack_fn.__name__ == "pcm_bit_depth_conversion":
                attacked_np = attack_fn(embedded_np, sr=process_config["audio"]["sample_rate"], pcm=attack_param)
            else:
                raise ValueError(f"Unsupported attack: {attack_fn.__name__}")

            attacked_np = truncate_or_pad_np(attacked_np, process_config["audio"]["max_len"])
            attacked_embedded = torch.from_numpy(attacked_np).unsqueeze(0).float().to(device)

            # --- Step 4: decode attacked embedded ---
            decoded = decoder(attacked_embedded.unsqueeze(0))

            # --- Step 5: bitwise accuracy ---
            total_acc += accuracy(decoded=decoded, msg=msg)
            total_bits += msg.numel()

            avg_acc = total_acc / total_bits
            tbar.set_postfix(acc=f"{avg_acc:.4f}")

    return {"acc": total_acc / total_bits if total_bits > 0 else 0.0}


def main(args):
    # Load configs
    with open("config/train_part_cl.yaml", "r") as f:
        train_config = yaml.safe_load(f)
    with open("config/process_lj.yaml", "r") as f:
        process_config = yaml.safe_load(f)
    with open("config/model.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    # --- Load dataset ---
    test_ds = LjAudioDataset(process_config=process_config, split="test")

    # config
    config = [
        {
            "model_ckpt": "model_ckpts/lj/wm_model_lj_pesq_3.61_acc_0.99_dist_acc_0.87_epoch_7_gs.pt",
            "model_desc": "wm model",
        },
        {
            "model_ckpt": "model_ckpts/lj/cl/wm_model_lj_pesq_3.65_acc_1.00_dist_acc_0.84_epoch_2_gs.pt",
            "model_desc": "wm cl model",
        },
    ]

    # Map attack types
    attack_map = {
        "delete": delete_samples,
        "resample": resample,
        "mp3": mp3_compression,
        "pcm": pcm_bit_depth_conversion,
    }
    if args.attack_type not in attack_map:
        raise ValueError(f"Unknown attack type: {args.attack_type}")
    attack_fn = attack_map[args.attack_type]

    # Different parameter sweeps depending on attack type
    if args.attack_type == "delete":
        sweep = [0.05 * x for x in range(1, 20)]  # delete percentages
    elif args.attack_type == "resample":
        sweep = [16000, 12000, 8000, 4000]  # downsample target rates
    elif args.attack_type == "mp3":
        sweep = [0, 2, 5, 9]  # ffmpeg quality settings
    elif args.attack_type == "pcm":
        sweep = [8, 16, 24]  # bit depths

    # Load models
    for model in config:
        embedder, decoder = load_from_ckpt(
            ckpt=model["model_ckpt"],
            model_config=model_config,
            train_config=train_config,
            process_config=process_config,
            device=device,
        )
        avg_pesq = get_model_average_pesq_dataset(
            embedder=embedder,
            dataset=test_ds,
            msg_length=train_config["watermark"]["length"],
            device=device,
            sr=process_config["audio"]["sample_rate"],
        )
        avg_stoi = get_model_average_stoi_dataset(
            embedder=embedder,
            dataset=test_ds,
            msg_length=train_config["watermark"]["length"],
            device=device,
            sr=process_config["audio"]["sample_rate"],
        )
        print(
            f"Loaded model from: {model['model_ckpt']} with avg pesq: {avg_pesq} and avg stoi: {avg_stoi}"
        )
        model.update(
            {
                "embedder": embedder,
                "decoder": decoder,
                "avg_pesq": avg_pesq,
                "avg_stoi": avg_stoi,
            }
        )

    # Run evaluations
    results = {}
    for model in config:
        embedder = model["embedder"]
        decoder = model["decoder"]
        model_desc = model["model_desc"]

        results[model_desc] = {}
        for param in sweep:
            print(f"Evaluating {model_desc} with {args.attack_type}, param={param}")
            metrics = evaluate(embedder, decoder, test_ds, attack_fn, param, process_config, train_config)
            results[model_desc][param] = metrics

    # Plot results
    plt.figure(figsize=(10, 7))
    for model in config:
        model_desc = model["model_desc"]
        avg_pesq = model["avg_pesq"]
        avg_stoi = model["avg_stoi"]

        acc_values = [results[model_desc][p]["acc"] for p in sweep]
        labels = [str(round(p, 2)) for p in sweep]

        plt.plot(
            labels,
            acc_values,
            marker="o",
            label=f"{model_desc} (PESQ={avg_pesq:.5f}, STOI={avg_stoi:.5f})",
        )

    plt.xlabel(f"{args.attack_type} parameter")
    plt.ylabel("Bitwise Accuracy")
    plt.title(f"Watermark Accuracy vs {args.attack_type}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    output_dir = "results_images"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"accuracy_vs_{args.attack_type}.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_type", type=str, required=True,
                        choices=["delete", "resample", "mp3", "pcm"],
                        help="Type of attack to evaluate")
    args = parser.parse_args()
    main(args)
