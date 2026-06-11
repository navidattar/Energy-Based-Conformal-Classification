import argparse
import pickle
import csv
import os
import torch
import torchvision
import torchvision.transforms as trn
from transformers import set_seed
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
from collections import defaultdict
from tqdm.auto import tqdm
from torchvision.datasets import Places365
import json

# Assuming examples.utils is available in the PYTHONPATH or current directory structure
try:
    from examples.utils import get_dataset_dir
except ImportError:
    print("Warning: 'examples.utils.get_dataset_dir' not found. CIFAR dataset path may need to be specified via --cifar_data_root.")
    def get_dataset_dir():
        return "./data" # Default CIFAR download location if root is "."


from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS, RAPS, SAPS, EnergyAPS, EnergyRAPS, EnergySAPS, LAC, EnergyLAC
from torchvision.models import resnet50

#######################################
# Model Definitions
#######################################

CIFAR10_MODEL_NAMES = [
    "resnet20", "resnet32", "resnet44", "resnet56",
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
    "mobilenetv2_x0_5", "mobilenetv2_x0_75", "mobilenetv2_x1_0", "mobilenetv2_x1_4",
    "shufflenetv2_x0_5", "shufflenetv2_x1_0", "shufflenetv2_x1_5", "shufflenetv2_x2_0",
    "repvgg_a0", "repvgg_a1", "repvgg_a2"
]

CIFAR100_MODEL_NAMES = [ # Same architecture names, but trained on CIFAR-100
    "resnet20", "resnet32", "resnet44", "resnet56",
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
    "mobilenetv2_x0_5", "mobilenetv2_x0_75", "mobilenetv2_x1_0", "mobilenetv2_x1_4",
    "shufflenetv2_x0_5", "shufflenetv2_x1_0", "shufflenetv2_x1_5", "shufflenetv2_x2_0",
    "repvgg_a0", "repvgg_a1", "repvgg_a2"
]

IMBALANCED_CIFAR100_MODEL_NAMES = [
    "resnet50_a0.1", "resnet50_a0.2", "resnet50_a0.3"
]
# Places365 model names from https://github.com/CSAILVision/places365
PLACES365_MODEL_NAMES = [
    # PyTorch models
    "alexnet", "resnet18", "resnet50", "densenet161",
    # Model names as mentioned in the repository
    "vgg16", "googlenet", "resnet152"
]

IMAGENET_TORCHVISION_MODEL_NAMES = sorted(list(set([
    # ResNets
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnext50_32x4d", "resnext101_32x8d", #"resnext101_64x4d", # Not in TorchVision 1.13
    "wide_resnet50_2", "wide_resnet101_2",
    # VGG
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
    # DenseNet
    "densenet121", "densenet161", "densenet169", "densenet201",
    # MobileNet
    "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
    # EfficientNet
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
    # Vision Transformer
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", #"vit_h_14", # Might require newer torchvision
    # Swin Transformer
    "swin_t", "swin_s", "swin_b", #"swin_v2_t", "swin_v2_s", "swin_v2_b", # Might require newer torchvision
    # SqueezeNet
    "squeezenet1_0", "squeezenet1_1",
    # ShuffleNet V2
    "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
    # MNASNet
    "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", #"mnasnet1_3", # Not in TorchVision 1.13
    # RegNet
    "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf", "regnet_y_8gf",
    "regnet_y_16gf", "regnet_y_32gf", #"regnet_y_128gf",
    "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_3_2gf", "regnet_x_8gf",
    "regnet_x_16gf", "regnet_x_32gf",
])))

# For argparse choices and error messages
ALL_AVAILABLE_MODEL_NAMES = sorted(list(set(CIFAR10_MODEL_NAMES + CIFAR100_MODEL_NAMES + IMAGENET_TORCHVISION_MODEL_NAMES + PLACES365_MODEL_NAMES)))

DEFAULT_MODEL_PER_DATASET = {
    "cifar10": "resnet56",
    "cifar100": "resnet56",
    "imagenet-val": "resnet50",
    "places365-val": "resnet50",  # Default model for Places365
    "imbalanced_cifar100": "resnet50_a0.1"
}

#######################################
# Helper Functions
#######################################

def setup_environment(seed_value):
    """Sets the random seed for reproducibility."""
    set_seed(seed_value)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}") # Keep print outside the loop
    return device

def create_dataloaders(dataset_name, dataset_instance, cal_split_size, test_split_size=None, batch_size=64, num_workers=4, seed=None):
    """Creates calibration and test dataloaders from a dataset instance."""
    # print(f"Preparing dataloaders for {dataset_name}: Cal size={cal_split_size}, Test size={test_split_size if test_split_size else 'remainder'}") # Keep print outside the loop
    if test_split_size is None:
        if cal_split_size >= len(dataset_instance):
            raise ValueError(f"Calibration size ({cal_split_size}) must be less than {dataset_name} size ({len(dataset_instance)}) to leave data for testing.")
        split_lengths = [cal_split_size, len(dataset_instance) - cal_split_size]
        if split_lengths[1] == 0:
            raise ValueError(f"Test split for {dataset_name} has 0 samples. Adjust calibration size.")
    else:
        if cal_split_size + test_split_size > len(dataset_instance):
            raise ValueError(f"Sum of calibration ({cal_split_size}) and test ({test_split_size}) sizes cannot exceed {dataset_instance.__class__.__name__} size ({len(dataset_instance)}).")
        split_lengths = [cal_split_size, test_split_size]
        if len(dataset_instance) > sum(split_lengths):
             # print(f"Warning: {len(dataset_instance) - sum(split_lengths)} samples from {dataset_instance.__class__.__name__} test set will be unused.") # Keep print outside the loop
             split_lengths = split_lengths + [len(dataset_instance) - sum(split_lengths)]
             cal_dataset, test_dataset, unused_dataset = torch.utils.data.random_split(dataset_instance, split_lengths, generator=torch.Generator().manual_seed(seed))
        else:
            cal_dataset, test_dataset = torch.utils.data.random_split(dataset_instance, split_lengths, generator=torch.Generator().manual_seed(seed))
    cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # print(f"Calibration loader size: {len(cal_dataloader.dataset)}, Test loader size: {len(test_dataloader.dataset)}") # Keep print outside the loop
    return cal_dataloader, test_dataloader

def prepare_imagenet_data(val_dir, cal_size, test_size, batch_size, num_workers, imagenet_mean, imagenet_std, seed):
    """Prepares ImageNet validation data."""
    print(f"Using ImageNet validation set from: {val_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"ImageNet validation directory not found: {val_dir}. Please check --imagenet_val_dir.")
    transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean=imagenet_mean, std=imagenet_std)])
    dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No images found in ImageNet validation directory: {val_dir}.")
    return create_dataloaders("ImageNet-val", dataset, cal_size, test_size, batch_size, num_workers, seed)

def _prepare_cifar_data(dataset_cls, dataset_name_str, root_dir, cal_size, test_size, batch_size, num_workers, mean, std, seed):
    """Helper function to prepare CIFAR datasets."""
    print(f"Using {dataset_name_str} dataset from: {root_dir}")
    transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean=mean, std=std)])

    try:
        dataset = dataset_cls(root=root_dir, train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Error loading {dataset_name_str} dataset from {root_dir}.")
        raise e
    if len(dataset) == 0:
        raise ValueError(f"{dataset_name_str} dataset is empty.")
    return create_dataloaders(dataset_name_str, dataset, cal_size, test_size, batch_size, num_workers, seed)

def prepare_cifar10_data(root_dir, cal_size, test_size, batch_size, num_workers, cifar10_mean, cifar10_std, seed):
    """Prepares CIFAR10 data."""
    return _prepare_cifar_data(torchvision.datasets.CIFAR10, "CIFAR10", root_dir, cal_size, test_size, batch_size, num_workers, cifar10_mean, cifar10_std, seed)

def prepare_cifar100_data(root_dir, cal_size, test_size, batch_size, num_workers, cifar100_mean, cifar100_std, seed):
    """Prepares CIFAR100 data."""
    return _prepare_cifar_data(torchvision.datasets.CIFAR100, "CIFAR100", root_dir, cal_size, test_size, batch_size, num_workers, cifar100_mean, cifar100_std, seed)

def prepare_places365_data(root_dir, cal_size, test_size, batch_size, num_workers, places365_mean, places365_std, download=False, seed=None):
    """Prepares Places365 validation data using torchvision's Places365 dataset class."""
    print(f"Using Places365 dataset from: {root_dir}")
    if not os.path.isdir(root_dir):
        if download:
            print(f"Creating Places365 root directory: {root_dir}")
            os.makedirs(root_dir, exist_ok=True)
        else:
            raise FileNotFoundError(f"Places365 root directory not found: {root_dir}. Please check --places365_root or use --download_places365.")
    
    # Places365 dataset uses 224x224 images by default when small=True
    transform = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=places365_mean, std=places365_std)
    ])
    
    try:
        # Use the validation split of Places365
        dataset = Places365(
            root=root_dir,
            split='val',
            small=True,  # Use 256x256 images for faster processing
            transform=transform,
            download=download  # Download if requested
        )
    except Exception as e:
        print(f"Error loading Places365 dataset from {root_dir}: {e}")
        raise
        
    if len(dataset) == 0:
        raise ValueError(f"Places365 dataset is empty.")
    
    return create_dataloaders("Places365-val", dataset, cal_size, test_size, batch_size, num_workers, seed)

def prepare_places365_ood_data(root_dir, test_size, batch_size, num_workers, cifar_mean, cifar_std, download=False, seed=None):
    """Prepares Places365 data for OOD evaluation with CIFAR preprocessing."""
    print(f"Using Places365 dataset for OOD evaluation from: {root_dir}")
    if not os.path.isdir(root_dir):
        if download:
            print(f"Creating Places365 root directory: {root_dir}")
            os.makedirs(root_dir, exist_ok=True)
        else:
            raise FileNotFoundError(f"Places365 root directory not found: {root_dir}. Please check --places365_root or use --download_places365.")
    
    # Use CIFAR preprocessing for OOD evaluation
    transform = trn.Compose([
        trn.Resize(32),  # Resize to CIFAR size
        trn.ToTensor(),
        trn.Normalize(mean=cifar_mean, std=cifar_std)  # Use CIFAR normalization
    ])
    
    try:
        # Use the validation split of Places365
        dataset = Places365(
            root=root_dir,
            split='val',
            small=True,
            transform=transform,
            download=download
        )
    except Exception as e:
        print(f"Error loading Places365 dataset from {root_dir}: {e}")
        raise
        
    if len(dataset) == 0:
        raise ValueError(f"Places365 dataset is empty.")
    
    # For OOD, we only need test data (no calibration split)
    if test_size > len(dataset):
        test_size = len(dataset)
        print(f"Warning: Requested test size ({test_size}) exceeds dataset size. Using full dataset.")
    
    # Create a subset for testing
    test_dataset, _ = torch.utils.data.random_split(
        dataset, 
        [test_size, len(dataset) - test_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    print(f"OOD test loader size: {len(test_dataloader.dataset)}")
    return test_dataloader

def load_model(dataset_name, model_name_str, device):
    """Loads a pretrained model for the specified dataset."""
    print(f"Loading pretrained model '{model_name_str}' for dataset '{dataset_name}'...")
    model = None
    if dataset_name == "imbalanced_cifar100":
        if model_name_str not in IMBALANCED_CIFAR100_MODEL_NAMES:
            raise ValueError(f"Model '{model_name_str}' not in predefined list for Imbalanced CIFAR-100. Available: {IMBALANCED_CIFAR100_MODEL_NAMES}")
        # Load checkpoint
        print(f'Loading checkpoint from ./CIFAR100-Imb-ckpts/best_model_{model_name_str}.pth')
        checkpoint = torch.load(f'./CIFAR100-Imb-ckpts/best_model_{model_name_str}.pth')

        # Create model
        model = resnet50()
        # Apply same modifications as in training
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        model.fc = torch.nn.Linear(model.fc.in_features, 100)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
    elif dataset_name == "imagenet-val":
        if model_name_str not in IMAGENET_TORCHVISION_MODEL_NAMES:
            raise ValueError(f"Model '{model_name_str}' not in predefined list for ImageNet. Available: {IMAGENET_TORCHVISION_MODEL_NAMES}")
        try:
            # Use weights='DEFAULT' for the best available pretrained weights
            model_fn = getattr(models, model_name_str)
            model = model_fn(weights='DEFAULT')
        except AttributeError:
            raise ValueError(f"Torchvision does not have model: {model_name_str}")
        except Exception as e:
            raise RuntimeError(f"Error loading torchvision model {model_name_str}: {e}")

    elif dataset_name == "places365-val":
        if model_name_str not in PLACES365_MODEL_NAMES:
            raise ValueError(f"Model '{model_name_str}' not in predefined list for Places365. Available: {PLACES365_MODEL_NAMES}")
        
        # Define helper function to load Places365 models
        def load_places365_model(model_fn, model_name):
            try:
                # Initialize model with empty weights
                model = model_fn(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, 365)
                
                # Load weights from Places365 repository
                state_dict = torch.hub.load_state_dict_from_url(
                    f'http://places2.csail.mit.edu/models_places365/{model_name}_places365.pth.tar',
                    progress=True, check_hash=False
                )
                
                # The state dict keys may need adjustment for the modern model
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Handle key format differences
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        k = k[7:]  # Remove 'module.' prefix
                    new_state_dict[k] = v
                    
                model.load_state_dict(new_state_dict, strict=False)
                return model
            except Exception as e:
                raise RuntimeError(f"Error loading Places365 {model_name} model: {e}")
        
        # Map model names to their constructor functions
        model_constructors = {
            "alexnet": models.alexnet,
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
            "densenet161": models.densenet161
        }
        
        if model_name_str in model_constructors:
            model = load_places365_model(model_constructors[model_name_str], model_name_str)
        else:
            # Other models mentioned in repository require direct file paths
            raise NotImplementedError(
                f"The Places365 {model_name_str} model requires a local path to the model file. "
                f"Please use one of the PyTorch models: {', '.join(model_constructors.keys())}"
            )

    elif dataset_name == "cifar100":
        if model_name_str not in CIFAR100_MODEL_NAMES:
            raise ValueError(f"Model '{model_name_str}' not in predefined list for CIFAR-100. Available: {CIFAR100_MODEL_NAMES}")
        hub_model_name = f"cifar100_{model_name_str}"
        try:
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_model_name, pretrained=True, trust_repo=True)
        except Exception as e:
            raise RuntimeError(f"Error loading CIFAR-100 model {hub_model_name} from torch.hub: {e}")

    elif dataset_name == "cifar10":
        if model_name_str not in CIFAR10_MODEL_NAMES:
            raise ValueError(f"Model '{model_name_str}' not in predefined list for CIFAR-10. Available: {CIFAR10_MODEL_NAMES}")
        hub_model_name = f"cifar10_{model_name_str}"
        try:
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_model_name, pretrained=True, trust_repo=True)
        except Exception as e:
            raise RuntimeError(f"Error loading CIFAR-10 model {hub_model_name} from torch.hub: {e}")
    else:
        raise ValueError(f"Unsupported dataset for model loading: {dataset_name}")

    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def evaluate_conformal_method(model, cal_loader, test_loader, alpha, score_fn_instance, temp_cal=1 , device=None, class_conditional=False):
    """Evaluates a conformal prediction method."""
    predictor = SplitPredictor(score_function=score_fn_instance, model=model, alpha=alpha, temperature=temp_cal, class_conditional=class_conditional)
    predictor.calibrate(cal_loader)
    eval_results = predictor.evaluate(test_loader)
    return eval_results['coverage_rate'], eval_results['average_size'], eval_results['CovGap']

def evaluate_conformal_method_with_logits(cal_logits, cal_labels, test_logits, test_labels, alpha, score_fn_instance, device=None, class_conditional=False, diff_violation=False):
    """
    Evaluates a conformal prediction method using pre-computed logits.
    
    Args:
        cal_logits (torch.Tensor): Pre-computed calibration logits.
        cal_labels (torch.Tensor): Calibration labels.
        test_logits (torch.Tensor): Pre-computed test logits.
        test_labels (torch.Tensor): Test labels.
        alpha (float): Significance level.
        score_fn_instance: Score function instance.
        device (torch.device): Device to use for computation.
        class_conditional (bool): Whether to use class-conditional conformal prediction.    
    Returns:
        tuple: (coverage_rate, average_size, covgap)
    """
    
    # Create predictor without model (we'll use pre-computed logits)
    predictor = SplitPredictor(score_function=score_fn_instance, model=None, alpha=alpha, class_conditional=class_conditional)
    
    # Set device manually since we don't have a model
    if device is not None:
        predictor._device = device
    
    # Calibrate using pre-computed logits
    predictor.calibrate_with_logits(cal_logits, cal_labels, alpha)
    
    # Evaluate using pre-computed logits
    eval_results = predictor.evaluate_with_logits(test_logits, test_labels, diff_violation=diff_violation)
    
    if diff_violation:
        return eval_results['coverage_rate'], eval_results['average_size'], eval_results['CovGap'],  eval_results['SSCV'], eval_results['VioClasses'], eval_results['EmptySetsPercentage'] , eval_results['DiffViolation']
    else:
        return eval_results['coverage_rate'], eval_results['average_size'], eval_results['CovGap'], eval_results['SSCV'], eval_results['VioClasses'], eval_results['EmptySetsPercentage']

def evaluate_conformal_method_ood(cal_logits, cal_labels, ood_logits, alpha, score_fn_instance, device=None, class_conditional=False):
    """
    Evaluates a conformal prediction method for OOD scenarios using pre-computed logits.
    Only computes set size and empty set percentage (no coverage metrics since no ground truth).
    
    Args:
        cal_logits (torch.Tensor): Pre-computed calibration logits.
        cal_labels (torch.Tensor): Calibration labels.
        ood_logits (torch.Tensor): Pre-computed OOD test logits.
        alpha (float): Significance level.
        score_fn_instance: Score function instance.
        device (torch.device): Device to use for computation.
        class_conditional (bool): Whether to use class-conditional conformal prediction.    
    Returns:
        tuple: (average_size, empty_sets_percentage)
    """
    
    # Create predictor without model (we'll use pre-computed logits)
    predictor = SplitPredictor(score_function=score_fn_instance, model=None, alpha=alpha, class_conditional=class_conditional)
    
    # Set device manually since we don't have a model
    if device is not None:
        predictor._device = device
    
    # Calibrate using pre-computed logits
    predictor.calibrate_with_logits(cal_logits, cal_labels, alpha)
    
    # Generate prediction sets for OOD data
    prediction_sets = predictor.predict_with_logits(ood_logits)
    
    # Calculate metrics
    set_sizes = prediction_sets.sum(axis=1).cpu().numpy()
    average_size = np.mean(set_sizes)
    empty_sets_count = np.sum(set_sizes == 0)
    average_size_without_empty = np.sum(set_sizes) / (len(set_sizes) - empty_sets_count)
    empty_sets_percentage = (empty_sets_count / len(set_sizes)) * 100.0
    
    return average_size, average_size_without_empty, empty_sets_percentage

def extract_logits_from_dataloader(model, dataloader, device=None):
    """
    Extract logits and labels from a dataloader using the model.
    
    Args:
        model (torch.nn.Module): The model to use for inference.
        dataloader (torch.utils.data.DataLoader): The dataloader to process.
        temperature (float): Temperature scaling parameter.
        device (torch.device): Device to use for computation.
    
    Returns:
        tuple: (logits, labels) where both are concatenated tensors.
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting logits"):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            # Get raw logits from model
            raw_logits = model(inputs)
            # Apply temperature scaling
            logits_list.append(raw_logits.detach())
            labels_list.append(labels)
    
    logits = torch.cat(logits_list, dim=0).float()
    labels = torch.cat(labels_list, dim=0)
    
    return logits, labels

def generate_performance_plot(dataset_name_str, plot_data, method_family_name,
                              alpha_value, target_coverage,
                              output_dir, bar_color, line_style_and_color,
                              conditional_mode_str):
    """Generates a plot comparing methods with different temperatures_energy."""
    plt.figure(figsize=(12, 7))
    x_labels = [item[0] for item in plot_data]
    coverages = [item[1] for item in plot_data]
    avg_sizes = [item[2] for item in plot_data]

    ax1 = plt.subplot(111)
    ax1.bar(x_labels, coverages, color=bar_color, alpha=0.5, label='Coverage Rate')
    min_cov_display = min(0.6, (min(coverages) - 0.05 if coverages else 0.6), target_coverage - 0.05)
    max_cov_display = max(1.0, (max(coverages) + 0.05 if coverages else 1.0), target_coverage + 0.05)
    ax1.set_ylim([min_cov_display, max_cov_display])
    ax1.set_ylabel('Coverage Rate')
    ax1.axhline(y=target_coverage, linestyle='--', color=bar_color, label=f'Target Coverage (1-{alpha_value:.2f})')
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")
    ax1.set_xlabel('Temperature_energy (or "Without Energy")')

    ax2 = ax1.twinx()
    # Plot average set sizes, highlight the minimum in green, and annotate all values
    min_idx = None
    if avg_sizes:
        min_val = min(avg_sizes)
        min_idx = avg_sizes.index(min_val)
    for i, val in enumerate(avg_sizes):
        color = 'green' if (min_idx is not None and i == min_idx) else 'black'
        ax2.plot(i, val, marker='o', color=color, markersize=10 if color == 'green' else 5)
        ax2.annotate(f"{val:.2f}", (i, val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=10, color=color if color else 'black')
    # Also plot the line connecting the points (use the original style)
    ax2.plot(range(len(x_labels)), avg_sizes, line_style_and_color, label='Average Set Size', zorder=1)
    ax2.set_ylabel('Average Set Size')
    if avg_sizes:
        ax2.set_ylim([0, max(avg_sizes) * 1.1 + 1])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.title(f'Effect of Temperature_energy on {method_family_name} ({dataset_name_str}) Performance\n({conditional_mode_str}, α = {alpha_value:.2f})')
    plt.tight_layout()

    plot_filename = f"{method_family_name.lower()}_energy_{conditional_mode_str.lower().replace(' ', '_')}_alpha_{alpha_value:.2f}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plot_path}")

#######################################
# Main Execution
#######################################

def main(args):
    device = setup_environment(args.seed)

    # Handle OOD mode: override dataset to cifar100 for model loading and calibration
    if args.ood:
        print("OOD mode enabled: Calibrating on CIFAR-100, testing on Places365 with CIFAR preprocessing")
        calibration_dataset = "cifar100"
        # Force model to be CIFAR-100 compatible if not specified
        if args.model_name is None:
            model_name_to_load = DEFAULT_MODEL_PER_DATASET["cifar100"]
            print(f"OOD mode: Using default CIFAR-100 model: {model_name_to_load}")
        else:
            # Validate that the specified model is compatible with CIFAR-100
            if args.model_name not in CIFAR100_MODEL_NAMES:
                raise ValueError(f"OOD mode requires a CIFAR-100 compatible model. '{args.model_name}' is not in CIFAR-100 model list: {CIFAR100_MODEL_NAMES}")
            model_name_to_load = args.model_name
            print(f"OOD mode: Using specified CIFAR-100 model: {model_name_to_load}")
    else:
        calibration_dataset = args.dataset
        # Determine and validate model name for normal mode
        model_name_to_load = args.model_name
        if model_name_to_load is None:
            model_name_to_load = DEFAULT_MODEL_PER_DATASET[args.dataset]
            print(f"--model_name not specified, using default for {args.dataset}: {model_name_to_load}")
        else:
            # Validate model_name against the chosen dataset
            valid_models_for_this_dataset = []
            if args.dataset == "cifar10":
                valid_models_for_this_dataset = CIFAR10_MODEL_NAMES
            elif args.dataset == "cifar100":
                valid_models_for_this_dataset = CIFAR100_MODEL_NAMES
            elif args.dataset == "imbalanced_cifar100":
                valid_models_for_this_dataset = IMBALANCED_CIFAR100_MODEL_NAMES
            elif args.dataset == "imagenet-val":
                valid_models_for_this_dataset = IMAGENET_TORCHVISION_MODEL_NAMES
            elif args.dataset == "places365-val":
                valid_models_for_this_dataset = PLACES365_MODEL_NAMES

            if model_name_to_load not in valid_models_for_this_dataset:
                parser_error_message = (
                    f"Model '{model_name_to_load}' is not valid for dataset '{args.dataset}'.\n"
                    f"Available models for {args.dataset}:\n{', '.join(valid_models_for_this_dataset)}"
                )
                # We can't call parser.error() here, so raise ValueError
                raise ValueError(parser_error_message)
            print(f"Using specified model for {args.dataset}: {model_name_to_load}")

    # Load the model once before the trial loop
    model = load_model(calibration_dataset, model_name_to_load, device)

    all_trials_results = [] # List to store results from all trials

    for trial_idx in range(args.num_trials):
        current_trial_seed = args.seed + trial_idx
        print(f"\n--- Starting Trial {trial_idx + 1}/{args.num_trials} (Seed: {current_trial_seed}) ---")
        setup_environment(current_trial_seed) # Set seed for the current trial

        # Prepare data inside the trial loop to ensure random_split uses the trial-specific seed
        if args.ood:
            # OOD mode: calibrate on CIFAR-100, test on Places365 with CIFAR preprocessing
            cal_dataloader, _ = prepare_cifar100_data(
                args.cifar_data_root, args.cal_size_cifar100, args.test_size_cifar100, args.batch_size, args.num_workers,
                args.cifar100_mean, args.cifar100_std, current_trial_seed
            )
            # Prepare OOD test data (Places365 with CIFAR preprocessing)
            ood_test_dataloader = prepare_places365_ood_data(
                args.places365_root, args.ood_test_size, args.batch_size, args.num_workers,
                args.cifar100_mean, args.cifar100_std, args.download_places365, current_trial_seed
            )
            test_dataloader = None  # No regular test data in OOD mode
        elif args.dataset == "imagenet-val":
            cal_dataloader, test_dataloader = prepare_imagenet_data(
                args.imagenet_val_dir, args.cal_size_imagenet, args.test_size_imagenet, args.batch_size, args.num_workers,
                args.imagenet_mean, args.imagenet_std, current_trial_seed
            )
        elif "cifar100" in args.dataset:
            cal_dataloader, test_dataloader = prepare_cifar100_data(
                args.cifar_data_root, args.cal_size_cifar100, args.test_size_cifar100, args.batch_size, args.num_workers,
                args.cifar100_mean, args.cifar100_std, current_trial_seed
            )
        elif args.dataset == "cifar10":
            cal_dataloader, test_dataloader = prepare_cifar10_data(
                args.cifar_data_root, args.cal_size_cifar10, args.test_size_cifar10, args.batch_size, args.num_workers,
                args.cifar10_mean, args.cifar10_std, current_trial_seed
            )
        elif args.dataset == "places365-val":
            cal_dataloader, test_dataloader = prepare_places365_data(
                args.places365_root, args.cal_size_places365, args.test_size_places365, args.batch_size, args.num_workers,
                args.places365_mean, args.places365_std, args.download_places365, current_trial_seed
            )
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}") # Should be caught by argparse choices

        print("-" * 140)
        # Update header to include Trial index and Class Conditional mode
        if args.ood:
            print(f"{'Trial':<6} {'Method':<14} {'Mode':<20} {'Alpha':<10} {'Temperature_energy':<20} {'Temperature_calibration':<24} {'Softplus_beta':<18} {'Average Set Size':<20} {'EmptySetsPercentage':<18}")
        else:
            print(f"{'Trial':<6} {'Method':<14} {'Mode':<20} {'Alpha':<10} {'Temperature_energy':<20} {'Temperature_calibration':<24} {'Softplus_beta':<18} {'Coverage Rate':<18} {'Average Set Size':<20} {'CovGap':<18} {'SSCV':<18} {'VioClasses':<18} {'EmptySetsPercentage':<18}")
        print("-" * 140)

        # Only run Class-Conditional if the flag is set
        conditional_modes_to_run = [True] if args.class_conditional else [False]

        cal_logits, cal_labels = extract_logits_from_dataloader(model, cal_dataloader, device=device)
        if args.ood:
            ood_logits, _ = extract_logits_from_dataloader(model, ood_test_dataloader, device=device)
            test_logits, test_labels = None, None
        else:
            test_logits, test_labels = extract_logits_from_dataloader(model, test_dataloader, device=device)
            ood_logits = None
        
        for is_class_conditional in conditional_modes_to_run:
            conditional_mode_str = "Class-Conditional" if is_class_conditional else "Standard"

            for alpha in args.alphas:
                for temp_cal in args.temperatures_calibration:
                    # Standard (or Energy-less) Methods
                    methods_config_standard = [
                        ('LAC', LAC, {}),
                        ('APS', APS, {}),
                        ('RAPS', RAPS, {'penalty': args.raps_penalty, 'kreg': args.raps_kreg}),
                        ('SAPS', SAPS, {'weight': args.saps_weight})
                    ]
                    if args.ent:
                        methods_config_standard += [
                            ('EntropyLAC', EnergyLAC, {'ent': True}),
                            ('EntropyAPS', EnergyAPS, {'ent': True}),
                            ('EntropyRAPS', EnergyRAPS, {'penalty': args.raps_penalty, 'kreg': args.raps_kreg, 'ent': True}),
                            ('EntropySAPS', EnergySAPS, {'weight': args.saps_weight, 'ent': True})
                        ]
                    for method_name, score_class, base_params in methods_config_standard:
                        score_fn_instance = score_class(**base_params)
                        cal_logits_temp = cal_logits / temp_cal
                        
                        if args.ood:
                            # OOD evaluation: only set size and empty sets
                            ood_logits_temp = ood_logits / temp_cal
                            size, size_without_empty, empty_sets = evaluate_conformal_method_ood(cal_logits_temp, cal_labels, ood_logits_temp, alpha, score_fn_instance, device=device, class_conditional=is_class_conditional)
                            print(f"{trial_idx + 1:<6} {method_name:<14} {conditional_mode_str:<20} {alpha:<10.3f} {'N/A':<20} {temp_cal:<24.2f} {'N/A':<18} {size:<20.4f} {empty_sets:<18.4f}")
                            all_trials_results.append({
                                'Trial': trial_idx + 1,
                                'Method': method_name,
                                'Alpha': alpha,
                                'Temperature_energy': 'N/A',
                                'Temperature_calibration': temp_cal,
                                'Softplus_beta': 'N/A',
                                'ClassConditional': is_class_conditional,
                                'Coverage': None,  # No coverage in OOD
                                'Size': size,
                                'SizeWithoutEmpty': size_without_empty,
                                'CovGap': None,  # No coverage gap in OOD
                                'SSCV': None,  # No SSCV in OOD
                                'VioClasses': None,  # No violation classes in OOD
                                'EmptySetsPercentage': empty_sets,
                                'DiffViolation': None,  # No diff violation in OOD
                                'OOD': True
                            })
                        else:
                            # Normal evaluation
                            test_logits_temp = test_logits / temp_cal
                            if args.diff_violation:
                                cov, size, covgap, sscv, vio_classes, empty_sets, diff_violation = evaluate_conformal_method_with_logits(cal_logits_temp, cal_labels, test_logits_temp, test_labels, alpha, score_fn_instance, device=device, class_conditional=is_class_conditional, diff_violation=True)
                            else:
                                cov, size, covgap, sscv, vio_classes, empty_sets = evaluate_conformal_method_with_logits(cal_logits_temp, cal_labels, test_logits_temp, test_labels, alpha, score_fn_instance, device=device, class_conditional=is_class_conditional, diff_violation=False)
                            print(f"{trial_idx + 1:<6} {method_name:<14} {conditional_mode_str:<20} {alpha:<10.3f} {'N/A':<20} {temp_cal:<24.2f} {'N/A':<18} {cov:<18.4f} {size:<20.4f} {covgap:<18.4f} {sscv:<18.4f} {vio_classes:<18.4f} {empty_sets:<18.4f}")
                            all_trials_results.append({
                                'Trial': trial_idx + 1,
                                'Method': method_name,
                                'Alpha': alpha,
                                'Temperature_energy': 'N/A',
                                'Temperature_calibration': temp_cal,
                                'Softplus_beta': 'N/A',
                                'ClassConditional': is_class_conditional,
                                'Coverage': cov,
                                'Size': size,
                                'CovGap': covgap,
                                'SSCV': sscv,
                                'VioClasses': vio_classes,
                                'EmptySetsPercentage': empty_sets,
                                'DiffViolation': diff_violation if args.diff_violation else None,
                                'OOD': False
                            })

                    # Energy-augmented Methods
                    energy_methods_config = [
                        ('EnergyLAC', EnergyLAC, {}),
                        ('EnergyAPS', EnergyAPS, {}),
                        ('EnergyRAPS', EnergyRAPS, {'penalty': args.raps_penalty, 'kreg': args.raps_kreg}),
                        ('EnergySAPS', EnergySAPS, {'weight': args.saps_weight})
                    ]

                    for method_name, score_class, base_params in energy_methods_config:

                        for temp_e in args.temperatures_energy:
                            for softplus_beta in args.softplus_beta:
                                score_fn_params = {**base_params, 'score_type': args.energy_score_type, 'temp_e': temp_e, 'temp_cal': temp_cal, 'softplus_beta': softplus_beta}
                                score_fn_instance = score_class(**score_fn_params)
                                cal_logits_temp = cal_logits / temp_cal
                                
                                if args.ood:
                                    # OOD evaluation: only set size and empty sets
                                    ood_logits_temp = ood_logits / temp_cal
                                    size, size_without_empty, empty_sets = evaluate_conformal_method_ood(cal_logits_temp, cal_labels, ood_logits_temp, alpha, score_fn_instance, device=device, class_conditional=is_class_conditional)
                                    print(f"{trial_idx + 1:<6} {method_name:<14} {conditional_mode_str:<20} {alpha:<10.3f} {temp_e:<20.2f} {temp_cal:<24.2f} {softplus_beta:<18.4f} {size:<20.4f} {empty_sets:<18.4f}")
                                    all_trials_results.append({
                                        'Trial': trial_idx + 1,
                                        'Method': method_name,
                                        'Alpha': alpha,
                                        'Temperature_energy': temp_e,
                                        'Temperature_calibration': temp_cal,
                                        'Softplus_beta': softplus_beta,
                                        'ClassConditional': is_class_conditional,
                                        'Coverage': None,  # No coverage in OOD
                                        'Size': size,
                                        'SizeWithoutEmpty': size_without_empty,
                                        'CovGap': None,  # No coverage gap in OOD
                                        'SSCV': None,  # No SSCV in OOD
                                        'VioClasses': None,  # No violation classes in OOD
                                        'EmptySetsPercentage': empty_sets,
                                        'DiffViolation': None,  # No diff violation in OOD
                                        'OOD': True
                                    })
                                else:
                                    # Normal evaluation
                                    test_logits_temp = test_logits / temp_cal
                                    if args.diff_violation:
                                        cov, size, covgap, sscv, vio_classes, empty_sets, diff_violation = evaluate_conformal_method_with_logits(cal_logits_temp, cal_labels, test_logits_temp, test_labels, alpha, score_fn_instance, device=device, class_conditional=is_class_conditional, diff_violation=True)
                                    else:                              
                                        cov, size, covgap, sscv, vio_classes, empty_sets = evaluate_conformal_method_with_logits(cal_logits_temp, cal_labels, test_logits_temp, test_labels, alpha, score_fn_instance, device=device, class_conditional=is_class_conditional, diff_violation=False)

                                    # Use trial index for consistent printing within a trial block
                                    print(f"{trial_idx + 1:<6} {method_name:<14} {conditional_mode_str:<20} {alpha:<10.3f} {temp_e:<20.2f} {temp_cal:<24.2f} {softplus_beta:<18.4f} {cov:<18.4f} {size:<20.4f} {covgap:<18.4f} {sscv:<18.4f} {vio_classes:<18.4f} {empty_sets:<18.4f}")
                                    all_trials_results.append({
                                        'Trial': trial_idx + 1,
                                        'Method': method_name,
                                        'Alpha': alpha,
                                        'Temperature_energy': temp_e,
                                        'Temperature_calibration': temp_cal,
                                        'Softplus_beta': softplus_beta,
                                        'ClassConditional': is_class_conditional,
                                        'Coverage': cov,
                                        'Size': size,
                                        'CovGap': covgap,
                                        'SSCV': sscv,
                                        'VioClasses': vio_classes,
                                        'EmptySetsPercentage': empty_sets,
                                        'DiffViolation': diff_violation if args.diff_violation else None,
                                        'OOD': False
                                    })
                            
                print("-" * 140) # Separator for each conditional mode within a trial

    print("\n--- All Trials Complete ---")

    # Calculate and Print Summary Statistics
    print("\n--- Summary Statistics Across Trials ---")
    # Group results by configuration
    grouped_results = defaultdict(lambda: defaultdict(list))
    for result in all_trials_results:
        config_key = (
            result['Method'],
            result['Alpha'],
            result['Temperature_energy'],
            result['Temperature_calibration'],
            result.get('Softplus_beta', 'N/A'),
            result['ClassConditional']
        )
        # Only append non-None values
        if result['Coverage'] is not None:
            grouped_results[config_key]['Coverage'].append(result['Coverage'])
        if result['Size'] is not None:
            grouped_results[config_key]['Size'].append(result['Size'])
        if result['CovGap'] is not None:
            grouped_results[config_key]['CovGap'].append(result['CovGap'])
        if result['SSCV'] is not None:
            grouped_results[config_key]['SSCV'].append(result['SSCV'])
        if result['VioClasses'] is not None:
            grouped_results[config_key]['VioClasses'].append(result['VioClasses'])
        if result['EmptySetsPercentage'] is not None:
            grouped_results[config_key]['EmptySetsPercentage'].append(result['EmptySetsPercentage'])
        if args.diff_violation and result['DiffViolation'] is not None:
            grouped_results[config_key]['DiffViolation'].append(result['DiffViolation'])

    print("-" * 200)
    if args.ood:
        print(f"{'Method':<14} {'Mode':<20} {'Alpha':<10} {'Temperature_energy':<20} {'Temperature_calibration':<24} {'Softplus_beta':<18} {'Average Set Size (Mean ± Std)':<32} {'Average Set Size Without Empty (Mean ± Std)':<32} {'EmptySetsPercentage (Mean ± Std)':<32}")
    else:
        print(f"{'Method':<14} {'Mode':<20} {'Alpha':<10} {'Temperature_energy':<20} {'Temperature_calibration':<24} {'Softplus_beta':<18} {'Coverage Rate (Mean ± Std)':<32} {'Average Set Size (Mean ± Std)':<32} {'CovGap (Mean ± Std)':<30} {'SSCV (Mean ± Std)':<30} {'VioClasses (Mean ± Std)':<30}")
    print("-" * 200)

    for config_key in grouped_results.keys():
        method, alpha, temp_e, temp_cal, softplus_beta, is_conditional = config_key
        results = grouped_results[config_key]

        mode_str = "Class-Conditional" if is_conditional else "Standard"
        temp_e_str = f"{temp_e:.2f}" if temp_e != 'N/A' else 'N/A'
        softplus_beta_str = f"{softplus_beta:.4f}" if softplus_beta != 'N/A' else 'N/A'

        if args.ood:
            # OOD mode: only show set size and empty sets percentage
            if 'Size' in results and results['Size']:
                mean_size = np.mean(results['Size'])
                std_size = np.std(results['Size'])
            else:
                mean_size, std_size = 0.0, 0.0

            if 'SizeWithoutEmpty' in results and results['SizeWithoutEmpty']:
                mean_size_without_empty = np.mean(results['SizeWithoutEmpty'])
                std_size_without_empty = np.std(results['SizeWithoutEmpty'])
            else:
                mean_size_without_empty, std_size_without_empty = 0.0, 0.0
                
            if 'EmptySetsPercentage' in results and results['EmptySetsPercentage']:
                mean_empty_sets = np.mean(results['EmptySetsPercentage'])
                std_empty_sets = np.std(results['EmptySetsPercentage'])
            else:
                mean_empty_sets, std_empty_sets = 0.0, 0.0

            print(
                f"{method:<14} {mode_str:<20} {alpha:<10.2f} {temp_e_str:<20} {temp_cal:<24.2f} {softplus_beta_str:<18} "
                f"{(f'{mean_size:.4f} ± {std_size:.4f}'): <32} "
                f"{(f'{mean_size_without_empty:.4f} ± {std_size_without_empty:.4f}'): <32} "
                f"{(f'{mean_empty_sets:.4f} ± {std_empty_sets:.4f}'): <32}"
            )
        else:
            # Normal mode: show all metrics
            mean_cov = np.mean(results['Coverage']) if results['Coverage'] else 0.0
            std_cov = np.std(results['Coverage']) if results['Coverage'] else 0.0
            mean_size = np.mean(results['Size']) if results['Size'] else 0.0
            std_size = np.std(results['Size']) if results['Size'] else 0.0
            mean_covgap = np.mean(results['CovGap']) if results['CovGap'] else 0.0
            std_covgap = np.std(results['CovGap']) if results['CovGap'] else 0.0
            mean_sscv = np.mean(results['SSCV']) if results['SSCV'] else 0.0
            std_sscv = np.std(results['SSCV']) if results['SSCV'] else 0.0
            mean_vio_classes = np.mean(results['VioClasses']) if results['VioClasses'] else 0.0
            std_vio_classes = np.std(results['VioClasses']) if results['VioClasses'] else 0.0
            mean_empty_sets = np.mean(results['EmptySetsPercentage']) if results['EmptySetsPercentage'] else 0.0
            std_empty_sets = np.std(results['EmptySetsPercentage']) if results['EmptySetsPercentage'] else 0.0

            print(
                f"{method:<14} {mode_str:<20} {alpha:<10.2f} {temp_e_str:<20} {temp_cal:<24.2f} {softplus_beta_str:<18} "
                f"{(f'{mean_cov:.4f} ± {std_cov:.4f}'): <32} "
                f"{(f'{mean_size:.4f} ± {std_size:.4f}'): <32} "
                f"{(f'{mean_covgap:.4f} ± {std_covgap:.4f}'): <30} "
                f"{(f'{mean_sscv:.4f} ± {std_sscv:.4f}'): <30} "
                f"{(f'{mean_vio_classes:.4f} ± {std_vio_classes:.4f}'): <30} "
                f"{(f'{mean_empty_sets:.4f} ± {std_empty_sets:.4f}'): <30}"
            )
    print("-" * 200)

    # Save all results to a CSV and pickle file
    # Include conditional mode and number of trials in filename
    ent_suffix = "_ent" if args.ent else ""
    mode_suffix = "_class_conditional" if args.class_conditional else "_standard"
    ood_suffix = "_ood" if args.ood else ""
    dataset_name_for_filename = "cifar100_to_places365" if args.ood else args.dataset
    csv_filename = f"results/csv/{dataset_name_for_filename}_{model_name_to_load}_energy_alpha0.1_results_trials_{args.num_trials}{mode_suffix}{ent_suffix}{ood_suffix}_rectified_softplus_beta.csv"
    pickle_filename = f"results/pkl/{dataset_name_for_filename}_{model_name_to_load}_energy_results_trials_{args.num_trials}{mode_suffix}{ent_suffix}{ood_suffix}_rectified_softplus_beta.pkl"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)

    # Save the list of dictionaries
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_trials_results[0].keys())
        writer.writeheader()
        writer.writerows(all_trials_results)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(all_trials_results, f)
    print(f"Saved results to {csv_filename} and {pickle_filename}")

    # Plotting
    if args.enable_plotting:
        if args.num_trials > 1:
            print(f"\nSkipping plotting because num_trials > 1. Plots are typically generated for single runs.")
        else:
            print("\nGenerating plots...")
            os.makedirs(args.plot_output_dir, exist_ok=True)
            plot_configs = {
                'LAC': {'methods': ['LAC', 'EnergyLAC'], 'bar_color': 'r', 'line_style': 'b-'},
                'APS': {'methods': ['APS', 'EnergyAPS'], 'bar_color': 'b', 'line_style': 'r-'},
                'RAPS': {'methods': ['RAPS', 'EnergyRAPS'], 'bar_color': 'g', 'line_style': 'm-'},
                'SAPS': {'methods': ['SAPS', 'EnergySAPS'], 'bar_color': 'y', 'line_style': 'c-'}
            }

            # Filter results for the single trial (trial_idx = 0)
            single_trial_results = [r for r in all_trials_results if r['Trial'] == 1]

            # Only plot for the mode that was run
            for is_class_conditional in conditional_modes_to_run:
                conditional_mode_str = "Class-Conditional" if is_class_conditional else "Standard"

                for family_name, config in plot_configs.items():
                    # Filter results for this family AND this conditional mode AND the single trial
                    family_results = [r for r in single_trial_results if r['Method'] in config['methods'] and r['ClassConditional'] == is_class_conditional]

                    for alpha_val in args.alphas:
                        # Filter results for this alpha
                        alpha_specific_results = [r for r in family_results if r['Alpha'] == alpha_val]

                        plot_data_for_alpha = []
                        # The dict structure is {'Trial', 'Method', 'Alpha', 'Temperature_energy', 'ClassConditional', 'Coverage', 'Size', 'CovGap', 'DiffViolation', 'SSCV', 'VioClasses'}
                        for res_dict in alpha_specific_results:
                            x_label = "Without Energy" if res_dict['Temperature_energy'] == 'N/A' else str(res_dict['Temperature_energy'])
                            plot_data_for_alpha.append((x_label, res_dict['Coverage'], res_dict['Size'], res_dict['CovGap'], res_dict['DiffViolation'], res_dict['SSCV'], res_dict['VioClasses'], res_dict['EmptySetsPercentage']))

                        # Sort 'Without Energy' first, then by temperature_energy
                        plot_data_for_alpha.sort(key=lambda x: (x[0] != "Without Energy", float(x[0]) if x[0] != "Without Energy" else -float('inf')))

                        if plot_data_for_alpha: # Only generate plot if there's data
                             generate_performance_plot(
                                dataset_name_str=args.dataset, plot_data=plot_data_for_alpha,
                                method_family_name=family_name, alpha_value=alpha_val,
                                target_coverage=(1 - alpha_val), output_dir=args.plot_output_dir,
                                bar_color=config['bar_color'], line_style_and_color=config['line_style'],
                                conditional_mode_str=conditional_mode_str
                            )
    elif args.num_trials > 1 and not args.enable_plotting:
         print(f"\nPlotting is disabled (--enable_plotting is False) and num_trials > 1. No plots will be generated.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Conformal Prediction Method Comparison", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run for each configuration.")
    parser.add_argument("--diff_violation", action="store_true", default=False, help="Calculate DiffViolation metric.")
    parser.add_argument("--ent", action="store_true", default=False, help="Calculate entropy-based methods.")
    parser.add_argument("--ood", action="store_true", default=False, help="Enable OOD evaluation: calibrate on CIFAR-100, test on Places365 with CIFAR preprocessing.")
    
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar10", "cifar100", "imagenet-val", "places365-val", "imbalanced_cifar100"],
                        help="Dataset to use.")

    dataset_group.add_argument("--model_name", type=str, default=None,
                        choices=ALL_AVAILABLE_MODEL_NAMES,
                        help="Model architecture. If None, a default for the dataset is used. "
                             "Ensure the model is compatible with the chosen dataset (validation done at runtime).")
    dataset_group.add_argument("--imagenet_val_dir", type=str, default="/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/val",
                        help="Path to ImageNet validation folder.")
    dataset_group.add_argument("--cifar_data_root", type=str, default=get_dataset_dir(),
                        help="Root directory for CIFAR datasets.")
    dataset_group.add_argument("--places365_root", type=str, default="./data/places365",
                        help="Root directory for Places365 dataset (will contain downloaded files if download=True).")
    dataset_group.add_argument("--download_places365", action="store_true", default=False,
                        help="Download Places365 dataset if not available locally.")

    loader_group = parser.add_argument_group('Dataloader Configuration')
    loader_group.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    loader_group.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")

    split_group = parser.add_argument_group('Dataset Split Sizes')
    split_group.add_argument("--cal_size_cifar10", type=int, default=5000, help="Calibration size for CIFAR10 (from 10k test images).") # Reduced default for faster testing
    split_group.add_argument("--test_size_cifar10", type=int, default=5000, help="Test size for CIFAR10 (from 10k test images).") # Reduced default
    split_group.add_argument("--cal_size_cifar100", type=int, default=5000, help="Calibration size for CIFAR100 (from 10k test images).") # Reduced default
    split_group.add_argument("--test_size_cifar100", type=int, default=5000, help="Test size for CIFAR100 (from 10k test images).") # Reduced default
    split_group.add_argument("--cal_size_imagenet", type=int, default=20000, help="Calibration size for ImageNet-val (test is remainder).")
    split_group.add_argument("--test_size_imagenet", type=int, default=20000, help="Test size for ImageNet-val (test is remainder).")
    split_group.add_argument("--cal_size_places365", type=int, default=18000, help="Calibration size for Places365-val (test is remainder).")
    split_group.add_argument("--test_size_places365", type=int, default=18500, help="Test size for Places365-val (test is remainder).")
    split_group.add_argument("--ood_test_size", type=int, default=10000, help="Test size for OOD evaluation on Places365.")

    exp_group = parser.add_argument_group('Experiment Hyperparameters')
    exp_group.add_argument("--alphas", type=float, nargs='+', default=[0.1, 0.05, 0.025, 0.01],  
                        help="Significance levels (alpha).")
    exp_group.add_argument("--softplus_beta", type=float, nargs='+', default=[1000, 750, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50, 25, 10, 5, 2, 1, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025, 0.00001, 0.000005], help="Beta for softplus function.")                    
    exp_group.add_argument("--temperatures_energy", type=float, nargs='+',
                        default = [0.0002, 0.00026, 0.00034, 0.00043, 0.00055, 0.00071, 0.00091, 0.00117, 0.0015, 0.00193, 0.00248, 0.00318, 0.00409, 0.00525, 0.00674, 0.00865, 0.01111, 0.01426, 0.01832, 0.02352, 0.0302, 0.03877, 0.04979, 0.06393, 0.08208, 0.1054, 0.13534, 0.17377, 0.22313, 0.2865, 0.36788, 0.47237, 0.60653, 0.7788, 1.0, 1.28403],
                        help="Temperatures of logit scaling for energy methods.")
    exp_group.add_argument("--temperatures_calibration", type=float, nargs='+',
                        default=[25, 20, 17.5, 15, 12.5, 10, 7.5, 5, 2.5, 2.0, 1.5, 1.3, 1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.01], 
                        help="Temperatures of logit scaling for model calibration.")
    exp_group.add_argument("--raps_penalty", type=float, default=0.2, help="Penalty for RAPS.")
    exp_group.add_argument("--raps_kreg", type=int, default=2, help="K_reg for RAPS.")
    exp_group.add_argument("--saps_weight", type=float, default=0.2, help="Weight for SAPS.")
    exp_group.add_argument("--energy_score_type", type=str, default="identity", choices=["identity", "softmax"],
                        help="Score type for EnergyXXX methods.")
    exp_group.add_argument("--class_conditional", action="store_true",
                             help="Enable class-conditional conformal prediction (if set, ONLY class-conditional is run).")
    parser.set_defaults(class_conditional=False) # Explicitly set default to False


    norm_group = parser.add_argument_group('Normalization Constants')
    norm_group.add_argument("--imagenet_mean", type=float, nargs=3, default=[0.485, 0.456, 0.406], help="ImageNet mean.")
    norm_group.add_argument("--imagenet_std", type=float, nargs=3, default=[0.229, 0.224, 0.225], help="ImageNet std.")
    norm_group.add_argument("--cifar10_mean", type=float, nargs=3, default=[0.4914, 0.4822, 0.4465], help="CIFAR10 mean.")
    norm_group.add_argument("--cifar10_std", type=float, nargs=3, default=[0.2023, 0.1994, 0.2010], help="CIFAR10 std.")
    norm_group.add_argument("--cifar100_mean", type=float, nargs=3, default=[0.5071, 0.4867, 0.4408], help="CIFAR100 mean.")
    norm_group.add_argument("--cifar100_std", type=float, nargs=3, default=[0.2675, 0.2565, 0.2761], help="CIFAR100 std.")
    norm_group.add_argument("--places365_mean", type=float, nargs=3, default=[0.485, 0.456, 0.406], help="Places365 mean.")
    norm_group.add_argument("--places365_std", type=float, nargs=3, default=[0.229, 0.224, 0.225], help="Places365 std.")

    plot_group = parser.add_argument_group('Plotting')
    plot_group.add_argument("--enable_plotting", default=False, action="store_true", help="Generate performance plots.")
    plot_group.add_argument("--plot_output_dir", type=str, default="energy_results", help="Directory for plots.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    model_name = args.model_name if args.model_name else DEFAULT_MODEL_PER_DATASET[args.dataset]
    # Ensure plot output directory is specific to dataset and model
    if args.ood:
        args.plot_output_dir = f"figures/cifar100_to_places365/{model_name}"
    else:
        args.plot_output_dir = f"figures/{args.dataset}/{model_name}"

    try:
        main(args)
    except ValueError as e: # Catch validation errors from main() related to model/dataset compatibility
        print(f"Configuration Error: {e}")
        # Consider exiting with an error code if preferred:
        # import sys
        # sys.exit(1)