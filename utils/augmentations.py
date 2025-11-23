import albumentations as A

def get_augmentation_transforms(augmentation_level):
    if abs(augmentation_level - 0) < 1e-6:
        return None
    level = max(0.0, min(1.0, augmentation_level))
    color_p = 0.8 * level
    noise_p = 0.3 * level
    secondary_p = 0.3 * level
    jpeg_p = 0.5 * level

    transform = A.Compose(
        [
            A.RandomScale(scale_limit=0.75, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=secondary_p),
            
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=1.0),
                A.FancyPCA(alpha=0.1, p=0.4),
            ], p=color_p), 

            A.OneOf([
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5), 
                A.GaussNoise(std_range=(0.1, 0.2), mean_range=(0.0, 0.0), p=0.5),                      
            ], p=noise_p),

            A.ImageCompression(quality_range=(70, 95), p=jpeg_p),

            A.Sharpen(alpha=(0.0, 0.2), p=secondary_p), 
            A.GaussianBlur(blur_limit=(3, 5), p=secondary_p)
        ]
    )
    return transform