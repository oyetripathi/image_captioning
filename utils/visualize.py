import matplotlib.pyplot as plt
import textwrap

def visualize_predictions(savepath, images, true_captions, pred_captions, n_rows=2, n_cols=2):
    num_samples = n_rows * n_cols
    assert (num_samples == len(images) == len(true_captions))
    
    plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    
    for i in range(num_samples):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        
        true_cap = "\n".join(textwrap.wrap(f"True: {true_captions[i]}", width=70))
        pred_cap = "\n".join(textwrap.wrap(f"Pred: {pred_captions[i]}", width=70))
        
        title = f"{true_cap}\n{pred_cap}"
        plt.title(title, fontsize=10, wrap=True, loc='left')
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)