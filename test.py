import sys
import yaml
import torch
import json
import pandas as pd
from torchvision import transforms
from utils.parse_import import run_imports
from utils.build_vocab import clean_txt
from metrics.BLEU import compute_bleu
from metrics.CIDEr import compute_cider
from utils.visualize import visualize_predictions
from datasets.collate_functions import pad_captions

def run_eval(df, dataset, EXPERIMENT_PATH):
    df = df.copy()
    sampled_df = df.sample(n = 6)
    metadata = [json.loads(sampled_df.loc[i]["metadata"].replace("'", '"')) for i in sampled_df.index] 
    visualize_predictions(
        savepath=f"{EXPERIMENT_PATH}/viz.png",
        images=dataset.get_images_from_metadata(metadata),
        true_captions=[sampled_df.loc[i]['true_caption'] for i in sampled_df.index],
        pred_captions=[sampled_df.loc[i]['generated_caption'] for i in sampled_df.index],
        n_rows=2,
        n_cols=3
    )
    metrics = {}
    df["true_caption"] = df["true_caption"].apply(clean_txt)
    df["generated_caption"] = df["generated_caption"].apply(clean_txt)
    
    all_preds = df["generated_caption"].unique()
    all_refs = [df[df["generated_caption"]==pred]["true_caption"].to_list() for pred in all_preds]
    
    metrics["BLUE@4"] = compute_bleu(all_refs, all_preds, n=4)
    metrics["CIDEr"] = compute_cider(all_preds, all_refs, max_n=5)
    
    with open(f"{EXPERIMENT_PATH}/metrics.json", "w") as f:
        json.dump(metrics, f)
    return

def run_test(config):
    EXPERIMENT_NAME = config.get("EXPERIMENT_NAME")
    EXPERIMENT_PATH = f"saved_models/{EXPERIMENT_NAME}"

    training_classes = run_imports(config)
    DatasetClass = training_classes["DatasetClass"]
    TokenizerClass = training_classes["TokenizerClass"]
    EncoderClass = training_classes["EncoderClass"]
    DecoderClass = training_classes["DecoderClass"]
    ModelClass = training_classes["ModelClass"]

    test_config = config.get("TESTING", {})

    BATCH_SIZE = test_config.get("BATCH_SIZE", 32)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = test_config.get("NUM_WORKERS", 1)

    tokenizer = TokenizerClass()
    tokenizer.load(f"{EXPERIMENT_PATH}/tokenizer.json")
    pad_token_id = tokenizer.pad_token_id

    encoder = EncoderClass(**config["ENCODER"].get("PARAMS", {}))
    decoder = DecoderClass(vocab_size=tokenizer.vocab_size, **config["DECODER"].get("PARAMS", {}))
    model = ModelClass(encoder, decoder)
    state_dict = torch.load(f"{EXPERIMENT_PATH}/best_model.pth", weights_only=True, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    dataset = DatasetClass(
        tokenizer=tokenizer, img_transform=encoder.transforms,
        rank=0,world_size=1,
        **config["DATASET"]["TEST"].get("PARAMS", {})
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        prefetch_factor=2, pin_memory=False,
        collate_fn=lambda x: pad_captions(x, pad_token_id),
        shuffle=(True if not isinstance(dataset, torch.utils.data.IterableDataset) else None)
    )

    outputs = model.generate_caption(dataloader, tokenizer, DEVICE, beam_size=5)
    test_df = pd.DataFrame(outputs)
    test_df.to_csv(f"{EXPERIMENT_PATH}/test_results.csv", index=False)
    print(f"Results saved at: {EXPERIMENT_PATH}/test_results.csv")
    test_df = pd.read_csv(f"{EXPERIMENT_PATH}/test_results.csv")
    run_eval(test_df, dataset, EXPERIMENT_PATH)
    return


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python test.py <config.yaml>")
        return
    config_path = args[0]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    run_test(config)
    return

if __name__ == "__main__":
    main()