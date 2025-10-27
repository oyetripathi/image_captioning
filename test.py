import sys
import yaml
import torch
import pandas as pd
from torchvision import transforms
from utils.parse_import import run_imports
from datasets.collate_functions import pad_captions


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

    test_df = pd.read_csv(config["DATASET"]["TEST"]["CSV_PATH"])

    tokenizer = TokenizerClass()
    tokenizer.load(f"{EXPERIMENT_PATH}/tokenizer.json")
    pad_token_id = tokenizer.pad_token_id

    im_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = DatasetClass(
        test_df,
        img_dir=config["DATASET"]["TEST"]["IMAGE_DIR"],
        tokenizer=tokenizer, img_transform=im_transforms,
        **config["DATASET"]["TEST"].get("PARAMS", {})
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: pad_captions(x, pad_token_id))

    encoder = EncoderClass(**config["ENCODER"].get("PARAMS", {}))
    decoder = DecoderClass(vocab_size=tokenizer.vocab_size, **config["DECODER"].get("PARAMS", {}))
    model = ModelClass(encoder, decoder)
    model.load_state_dict(torch.load(f"{EXPERIMENT_PATH}/model.pth", weights_only=True))

    outputs = model.generate_caption(dataloader, tokenizer, DEVICE, max_len=20, beam_size=5)
    out_df = pd.DataFrame(outputs).set_index("id")

    test_df = pd.merge(test_df, out_df, left_index=True, right_index=True, how="left")
    test_df.to_csv(f"{EXPERIMENT_PATH}/test_results.csv", index=False)
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