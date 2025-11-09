import sys
import yaml
import pandas as pd
import os
import json
from PIL import Image
from utils.visualize import visualize_predictions
from utils.build_vocab import clean_txt
from metrics.BLEU import compute_bleu
from metrics.CIDEr import compute_cider

def run_eval(config):
    EXPERIMENT_NAME = config.get("EXPERIMENT_NAME")
    EXPERIMENT_PATH = f"saved_models/{EXPERIMENT_NAME}"

    if not os.path.exists(f"{EXPERIMENT_PATH}/test_results.csv"):
        raise Exception("no test results found for the passed experiment")
    df  = pd.read_csv(f"{EXPERIMENT_PATH}/test_results.csv")
    image_dir = config["DATASET"]["TEST"]["IMAGE_DIR"]

    sampled_df = df.sample(n = 6)
    visualize_predictions(
        savepath=f"{EXPERIMENT_PATH}/viz.png",
        images=[
            Image.open(f"{image_dir}/{sampled_df.loc[i]['image']}").convert("RGB").resize((224, 224)) 
            for i in sampled_df.index
        ],
        true_captions=[sampled_df.loc[i]['caption'] for i in sampled_df.index],
        pred_captions=[sampled_df.loc[i]['generated_caption'] for i in sampled_df.index],
        n_rows=2,
        n_cols=3
    )

    metrics = {}
    df["caption"] = df["caption"].apply(clean_txt)
    df["generated_caption"] = df["generated_caption"].apply(clean_txt)
    df = df.set_index("image")
    
    all_ims = list(df.index.unique())
    all_refs = [df.loc[im]["caption"].to_list() for im in all_ims]
    all_preds = [df.loc[im]["generated_caption"].unique().tolist()[0] for im in all_ims]
    
    metrics["BLUE@4"] = compute_bleu(all_refs, all_preds, n=4)
    metrics["CIDEr"] = compute_cider(all_preds, all_refs, max_n=5)
    
    with open(f"{EXPERIMENT_PATH}/metrics.json", "w") as f:
        json.dump(metrics, f)

    return


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python eval.py <config.yaml>")
        return
    config_path = args[0]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    run_eval(config)
    return

if __name__ == "__main__":
    main()