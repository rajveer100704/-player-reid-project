import random
from PIL import Image
from torchvision.transforms import transforms

class RandomErasing(object):
    """Randomly selects a rectangle region in an image and erases its pixels.
    Reference: Zhong et al. Random Erasing Data Augmentation. arXiv:1708.04896.
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(torch.sqrt(torch.tensor(target_area * aspect_ratio))))
            w = int(round(torch.sqrt(torch.tensor(target_area / aspect_ratio))))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

def build_transforms(height=256, width=128, is_train=True):
    """Builds standard high-performance ReID augmentations."""
    res = []
    res.append(transforms.Resize((height, width), interpolation=3))
    
    if is_train:
        res.append(transforms.RandomHorizontalFlip(p=0.5))
        res.append(transforms.Pad(10))
        res.append(transforms.RandomCrop((height, width)))
        res.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        
    res.append(transforms.ToTensor())
    res.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    if is_train:
        res.append(RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0]))
        
    return transforms.Compose(res)
