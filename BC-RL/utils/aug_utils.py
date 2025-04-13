from torchvision import transforms

def get_rgb_transform(size=224):
    """
    Creates a composition of image transforms for RGB images.
    
    Args:
        size (int): Target size for resize and crop operations
        
    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize(size),  
        transforms.RandomResizedCrop(  
            size=size,
            scale=(0.8, 1.0),  
            ratio=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=(0.9, 1.1),
            contrast=(0.9, 1.1),
            saturation=(0.9, 1.1),
            hue=(-0.1, 0.1)
        ),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4850, 0.4560, 0.4060],
            std=[0.2290, 0.2240, 0.2250]
        )
    ])

def get_center_transform(size=128):
    """
    Creates a composition of image transforms for center view, with color augmentation
    and normalization but no resizing.
    
    Args:
        size (int): Target size for center crop operation
        
    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    return transforms.Compose([
        transforms.ColorJitter(
            brightness=(0.9, 1.1),
            contrast=(0.9, 1.1),
            saturation=(0.9, 1.1),
            hue=(-0.1, 0.1)
        ),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4850, 0.4560, 0.4060],
            std=[0.2290, 0.2240, 0.2250]
        )
    ])

def get_multi_view_transform(size=128):
    """
    Creates a composition of image transforms for side cameras, including resize,
    random resize, color augmentation, and normalization.
    
    Args:
        size (int): Target size for resize and crop operations
        
    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize(size),  
        transforms.RandomResizedCrop(  
            size=size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=(0.9, 1.1),
            contrast=(0.9, 1.1),
            saturation=(0.9, 1.1),
            hue=(-0.1, 0.1)
        ),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4850, 0.4560, 0.4060],
            std=[0.2290, 0.2240, 0.2250]
        )
    ])