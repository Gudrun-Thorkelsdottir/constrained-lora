from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader



dataset="camelyon17"
dataset = get_dataset(dataset=args.dataset, download=True)
