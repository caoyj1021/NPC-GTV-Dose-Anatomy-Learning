from pathlib import Path
import os
from datetime import datetime
import gc, hashlib, json, math, random, shutil, sys, time, traceback, warnings, platform, contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc, log_loss
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from scipy.special import logit
import matplotlib.pyplot as plt

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
EXPECTED_SHAPE_ZYX=(64,112,80)
BASE_CHANNELS,DROPOUT=16,0.20
BATCH_SIZE,NUM_WORKERS=4,0
MAX_EPOCHS=120
MIN_CHECKPOINT_EPOCH,MIN_EARLY_STOPPING_EPOCH=10,40
EARLY_STOPPING_PATIENCE,EARLY_STOPPING_MIN_DELTA=25,1e-4
SELECTION_SMOOTHING_WINDOW=3
BASE_LR,MIN_LR,WARMUP_EPOCHS=1e-3,5e-5,5
WEIGHT_DECAY,MAX_GRAD_NORM=1e-4,5.0
USE_AMP,AMP_PRECISION_MODE=True,'auto'
ENABLE_AUGMENTATION=False
DESCRIPTIVE_THRESHOLD=0.5
BOOTSTRAP_N,BOOTSTRAP_SEED=2000,20261025
ALLOWED_IMAGE_CHANNELS={'ct','dose','oral','gtv','dose_oral','dose_gtv'}
REPEAT_NUMBER=FOLD_NUMBER=GLOBAL_SEED=INNER_SPLIT_SEED=None

def safe_pr_auc(y_true, probabilities):
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    return float(auc(recall, precision))

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def save_json(payload, output_path: Path) -> None:

    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value
    output_path.write_text(json.dumps(convert(payload), ensure_ascii=False, indent=2), encoding='utf-8')

def sha256_file(file_path: Path, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(file_path, 'rb') as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def safe_auc(y_true, probabilities) -> float:
    try:
        return float(roc_auc_score(y_true, probabilities))
    except Exception:
        return np.nan

def calculate_binary_metrics(y_true, probabilities, threshold: float=0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    sensitivity = tp / (tp + fn) if tp + fn > 0 else np.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
    return {'n': int(len(y_true)), 'prevalence': float(y_true.mean()), 'roc_auc': safe_auc(y_true, probabilities), 'pr_auc': safe_pr_auc(y_true, probabilities), 'brier': float(brier_score_loss(y_true, probabilities)), 'threshold_descriptive_only': float(threshold), 'accuracy': float(accuracy_score(y_true, predictions)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, predictions)), 'sensitivity': float(sensitivity), 'specificity': float(specificity), 'precision': float(precision_score(y_true, predictions, zero_division=0)), 'f1': float(f1_score(y_true, predictions, zero_division=0)), 'probability_min': float(probabilities.min()), 'probability_max': float(probabilities.max()), 'probability_mean': float(probabilities.mean()), 'probability_std': float(probabilities.std(ddof=1)), 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad)))

def learning_rate_for_epoch(epoch: int) -> float:
    """
    固定的绝对epoch学习率计划。

    Stage A与Stage B均使用相同函数。
    Stage B即使只训练selected_epoch，也不会将余弦周期压缩。
    """
    if epoch <= WARMUP_EPOCHS:
        if WARMUP_EPOCHS == 1:
            warmup_fraction = 1.0
        else:
            warmup_fraction = (epoch - 1) / (WARMUP_EPOCHS - 1)
        start_ratio = 0.2
        ratio = start_ratio + (1.0 - start_ratio) * warmup_fraction
        return float(BASE_LR * ratio)
    progress = (epoch - WARMUP_EPOCHS) / (MAX_EPOCHS - WARMUP_EPOCHS)
    progress = min(max(progress, 0.0), 1.0)
    cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(MIN_LR + (BASE_LR - MIN_LR) * cosine_value)

def set_optimizer_lr(optimizer, learning_rate: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group['lr'] = float(learning_rate)

def resolve_amp_settings(device: torch.device) -> dict:
    """
    自动选择训练精度。

    优先级：
    1. CUDA + BF16支持：BF16 autocast，不使用GradScaler；
    2. CUDA但不支持BF16：FP16 autocast，使用GradScaler；
    3. USE_AMP=False：全FP32。
    """
    if not USE_AMP or device.type != 'cuda':
        return {'enabled': False, 'dtype': torch.float32, 'mode': 'FP32', 'scaler_enabled': False}
    bf16_supported = False
    if hasattr(torch.cuda, 'is_bf16_supported'):
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False
    if AMP_PRECISION_MODE in {'auto', 'bf16'} and bf16_supported:
        return {'enabled': True, 'dtype': torch.bfloat16, 'mode': 'BF16', 'scaler_enabled': False}
    return {'enabled': True, 'dtype': torch.float16, 'mode': 'FP16', 'scaler_enabled': True}

class NPCNPZDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, channels, training: bool):
        self.dataframe = dataframe.copy().reset_index(drop=True)
        self.channels = list(channels)
        self.training = bool(training)
        unknown_channels = set(self.channels) - ALLOWED_IMAGE_CHANNELS
        if unknown_channels:
            raise ValueError(f'未知图像通道：{sorted(unknown_channels)}')
        if 'dose_oral' in self.channels and (not {'dose', 'oral'}.issubset(set(self.channels))):
            raise ValueError('使用dose_oral时，模型输入必须同时包含dose和oral')
        if 'dose_gtv' in self.channels and (not {'dose', 'gtv'}.issubset(set(self.channels))):
            raise ValueError('使用dose_gtv时，模型输入必须同时包含dose和gtv')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        patient_id = int(row['patient_id'])
        label = int(row['severe_mucositis'])
        npz_path = Path(row['npz_path'])
        required_base_channels = {channel for channel in self.channels if channel in {'ct', 'dose', 'oral', 'gtv'}}
        if 'dose_oral' in self.channels:
            required_base_channels.update({'dose', 'oral'})
        if 'dose_gtv' in self.channels:
            required_base_channels.update({'dose', 'gtv'})
        arrays = {}
        with np.load(npz_path, allow_pickle=False) as data:
            for channel in sorted(required_base_channels):
                if channel not in data:
                    raise KeyError(f'病例{patient_id}缺少基础通道{channel}')
                array = np.asarray(data[channel], dtype=np.float32)
                if tuple(array.shape) != EXPECTED_SHAPE_ZYX:
                    raise RuntimeError(f'病例{patient_id}通道{channel}形状错误：{array.shape}')
                if not np.isfinite(array).all():
                    raise RuntimeError(f'病例{patient_id}通道{channel}存在NaN/Inf')
                arrays[channel] = array
            if 'patient_id' not in data.files:
                raise KeyError(f'病例{patient_id}的NPZ缺少patient_id')
            internal_id = np.asarray(data['patient_id'])
            if internal_id.size != 1 or not np.isfinite(internal_id).all() or float(internal_id.item()) != int(internal_id.item()):
                raise RuntimeError('NPZ patient_id must be one finite integer')
            internal_patient_id = int(internal_id.item())
        if internal_patient_id != patient_id:
            raise RuntimeError(f'NPZ内部ID与病例不一致: {internal_patient_id} != {patient_id}')
        if 'oral' in arrays:
            oral_mask = (arrays['oral'] > 0.5).astype(np.float32)
            if int(oral_mask.sum()) <= 0:
                raise RuntimeError(f'病例{patient_id}的oral为空')
            arrays['oral'] = oral_mask
        if 'gtv' in arrays:
            gtv_mask = (arrays['gtv'] > 0.5).astype(np.float32)
            if int(gtv_mask.sum()) <= 0:
                raise RuntimeError(f'病例{patient_id}的gtv为空')
            arrays['gtv'] = gtv_mask
        if 'dose_oral' in self.channels:
            arrays['dose_oral'] = (arrays['dose'] * arrays['oral']).astype(np.float32, copy=False)
            if not np.isfinite(arrays['dose_oral']).all():
                raise RuntimeError(f'病例{patient_id}的dose_oral存在NaN/Inf')
        if 'dose_gtv' in self.channels:
            arrays['dose_gtv'] = (arrays['dose'] * arrays['gtv']).astype(np.float32, copy=False)
            if not np.isfinite(arrays['dose_gtv']).all():
                raise RuntimeError(f'病例{patient_id}的dose_gtv存在NaN/Inf')
        image = np.stack([arrays[channel] for channel in self.channels], axis=0).astype(np.float32, copy=False)
        expected_image_shape = (len(self.channels), *EXPECTED_SHAPE_ZYX)
        if tuple(image.shape) != expected_image_shape:
            raise RuntimeError(f'病例{patient_id}最终输入shape错误：{image.shape}，预期{expected_image_shape}')
        if self.training and ENABLE_AUGMENTATION:
            pass
        return (torch.from_numpy(np.ascontiguousarray(image)), torch.tensor(float(label), dtype=torch.float32), torch.tensor(patient_id, dtype=torch.int64))

def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = 8
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(num_groups=groups, num_channels=channels)

class BasicResidualBlock3D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = make_group_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = make_group_norm(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), make_group_norm(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + identity
        x = self.relu(x)
        return x

class LightweightResNet10_3D(nn.Module):

    def __init__(self, in_channels: int, base_channels: int=16, dropout: float=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False), make_group_norm(base_channels), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer1 = BasicResidualBlock3D(base_channels, base_channels, stride=1)
        self.layer2 = BasicResidualBlock3D(base_channels, base_channels * 2, stride=2)
        self.layer3 = BasicResidualBlock3D(base_channels * 2, base_channels * 4, stride=2)
        self.layer4 = BasicResidualBlock3D(base_channels * 4, base_channels * 8, stride=2)
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(base_channels * 8, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x).squeeze(1)

def make_loader(dataframe: pd.DataFrame, channels, training: bool, seed: int) -> DataLoader:
    dataset = NPCNPZDataset(dataframe=dataframe, channels=channels, training=training)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=training, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=False, persistent_workers=False, generator=generator)

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, scaler, device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype) -> dict:
    model.train()
    total_loss = 0.0
    total_samples = 0
    skipped_nonfinite_batches = 0
    y_true_all = []
    probabilities_all = []
    for images, labels, _patient_ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            skipped_nonfinite_batches += 1
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        if not torch.isfinite(grad_norm):
            skipped_nonfinite_batches += 1
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        probabilities = torch.sigmoid(logits.detach().float())
        y_true_all.append(labels.detach().cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())
    if total_samples <= 0:
        raise RuntimeError('当前epoch没有有效训练样本')
    y_true = np.concatenate(y_true_all).astype(int)
    probabilities = np.concatenate(probabilities_all).astype(float)
    return {'loss': float(total_loss / total_samples), 'roc_auc': safe_auc(y_true, probabilities), 'pr_auc': safe_pr_auc(y_true, probabilities), 'skipped_nonfinite_batches': int(skipped_nonfinite_batches)}

@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, criterion, device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    patient_ids_all = []
    y_true_all = []
    probabilities_all = []
    for images, labels, patient_ids in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels_device)
        probabilities = torch.sigmoid(logits.float())
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().float().item()) * batch_size
        total_samples += batch_size
        patient_ids_all.append(patient_ids.numpy())
        y_true_all.append(labels.numpy())
        probabilities_all.append(probabilities.cpu().numpy())
    if total_samples <= 0:
        raise RuntimeError('评价DataLoader为空')
    return {'loss': float(total_loss / total_samples), 'patient_ids': np.concatenate(patient_ids_all).astype(int), 'y_true': np.concatenate(y_true_all).astype(int), 'probabilities': np.concatenate(probabilities_all).astype(float)}

def make_prediction_dataframe(output: dict, source_df: pd.DataFrame, model_name: str, role_name: str) -> pd.DataFrame:
    dataframe = pd.DataFrame({'patient_id': output['patient_ids'], 'y_true': output['y_true'], 'probability': output['probabilities'], 'prediction_at_0_5_descriptive_only': (output['probabilities'] >= DESCRIPTIVE_THRESHOLD).astype(int)})
    metadata = source_df[['patient_id', 'model_center', 'stratum']].copy()
    dataframe = dataframe.merge(metadata, on='patient_id', how='left', validate='one_to_one')
    dataframe['model_name'] = model_name
    dataframe['dataset_role'] = role_name
    dataframe['repeat'] = REPEAT_NUMBER
    dataframe['fold'] = FOLD_NUMBER
    return dataframe.sort_values('patient_id').reset_index(drop=True)

def run_stage_a(model_name: str, channels, inner_train_df: pd.DataFrame, inner_validation_df: pd.DataFrame, device: torch.device, model_dir: Path):
    seed_everything(GLOBAL_SEED)
    model = LightweightResNet10_3D(in_channels=len(channels), base_channels=BASE_CHANNELS, dropout=DROPOUT).to(device)
    parameter_count = count_trainable_parameters(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=WEIGHT_DECAY)
    amp_settings = resolve_amp_settings(device)
    amp_enabled = bool(amp_settings['enabled'])
    amp_dtype = amp_settings['dtype']
    amp_mode = str(amp_settings['mode'])
    scaler = torch.amp.GradScaler(device.type, enabled=bool(amp_settings['scaler_enabled']))
    print(f'[{model_name}] AMP mode：{amp_mode}')
    train_loader = make_loader(inner_train_df, channels, training=True, seed=GLOBAL_SEED)
    validation_loader = make_loader(inner_validation_df, channels, training=False, seed=GLOBAL_SEED)
    checkpoint_path = model_dir / 'stage_A_best_model.pt'
    history_records = []
    recent_validation_losses = []
    best_selection_score = math.inf
    best_epoch = None
    best_inner_loss_raw = None
    no_improvement_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        current_lr = learning_rate_for_epoch(epoch)
        set_optimizer_lr(optimizer, current_lr)
        train_output = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, scaler=scaler, device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        validation_output = evaluate_loader(model=model, loader=validation_loader, criterion=criterion, device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        validation_metrics = calculate_binary_metrics(validation_output['y_true'], validation_output['probabilities'], threshold=DESCRIPTIVE_THRESHOLD)
        recent_validation_losses.append(float(validation_output['loss']))
        if len(recent_validation_losses) > SELECTION_SMOOTHING_WINDOW:
            recent_validation_losses.pop(0)
        if len(recent_validation_losses) == SELECTION_SMOOTHING_WINDOW:
            selection_score = float(np.mean(recent_validation_losses))
        else:
            selection_score = np.nan
        eligible_for_checkpoint = epoch >= MIN_CHECKPOINT_EPOCH and np.isfinite(selection_score)
        improved = eligible_for_checkpoint and selection_score < best_selection_score - EARLY_STOPPING_MIN_DELTA
        if improved:
            best_selection_score = float(selection_score)
            best_epoch = int(epoch)
            best_inner_loss_raw = float(validation_output['loss'])
            no_improvement_epochs = 0
            torch.save({'model_name': model_name, 'channels': list(channels), 'model_state': model.state_dict(), 'best_epoch': best_epoch, 'best_selection_score_smoothed_inner_loss': best_selection_score, 'best_inner_loss_raw_at_checkpoint': best_inner_loss_raw, 'parameter_count': parameter_count, 'training_seed': GLOBAL_SEED, 'inner_split_seed': INNER_SPLIT_SEED, 'amp_mode': amp_mode}, checkpoint_path)
        elif eligible_for_checkpoint:
            no_improvement_epochs += 1
        history_records.append({'epoch': epoch, 'learning_rate': current_lr, 'amp_mode': amp_mode, 'train_loss': train_output['loss'], 'train_roc_auc_training_mode': train_output['roc_auc'], 'train_pr_auc_training_mode': train_output['pr_auc'], 'inner_validation_loss': validation_output['loss'], 'inner_validation_loss_smoothed': selection_score, 'inner_validation_roc_auc': validation_metrics['roc_auc'], 'inner_validation_pr_auc': validation_metrics['pr_auc'], 'inner_validation_brier': validation_metrics['brier'], 'inner_probability_min': validation_metrics['probability_min'], 'inner_probability_max': validation_metrics['probability_max'], 'inner_probability_std': validation_metrics['probability_std'], 'checkpoint_eligible': eligible_for_checkpoint, 'checkpoint_improved': improved, 'current_best_epoch': best_epoch, 'no_improvement_epochs': no_improvement_epochs, 'skipped_nonfinite_batches': train_output['skipped_nonfinite_batches'], 'epoch_seconds': float(time.time() - epoch_start)})
        print(f"[{model_name}] Stage A | Epoch {epoch:03d} | LR={current_lr:.6f} | TrainLoss={train_output['loss']:.4f} | InnerLoss={validation_output['loss']:.4f} | InnerAUC={validation_metrics['roc_auc']:.4f} | InnerPR={validation_metrics['pr_auc']:.4f} | BestEpoch={best_epoch} | Bad={no_improvement_epochs}/{EARLY_STOPPING_PATIENCE} | SkippedInf={train_output['skipped_nonfinite_batches']}")
        should_stop = epoch >= MIN_EARLY_STOPPING_EPOCH and best_epoch is not None and (no_improvement_epochs >= EARLY_STOPPING_PATIENCE)
        if should_stop:
            print(f'[{model_name}] Stage A在epoch {epoch}触发early stopping。')
            break
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(model_dir / 'stage_A_history.csv', index=False, encoding='utf-8-sig')
    if best_epoch is None or not checkpoint_path.exists():
        raise RuntimeError(f'{model_name}未生成有效Stage A checkpoint')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    best_inner_output = evaluate_loader(model=model, loader=validation_loader, criterion=criterion, device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
    best_inner_metrics = calculate_binary_metrics(best_inner_output['y_true'], best_inner_output['probabilities'], threshold=DESCRIPTIVE_THRESHOLD)
    prediction_df = make_prediction_dataframe(output=best_inner_output, source_df=inner_validation_df, model_name=model_name, role_name='inner_validation_20pct')
    prediction_df.to_csv(model_dir / 'inner_validation_predictions_at_best_checkpoint.csv', index=False, encoding='utf-8-sig')
    return {'best_epoch': int(best_epoch), 'best_selection_score': float(best_selection_score), 'best_inner_loss_raw': float(best_inner_loss_raw), 'best_inner_metrics': best_inner_metrics, 'parameter_count': parameter_count, 'history_df': history_df, 'checkpoint_path': checkpoint_path, 'total_skipped_nonfinite_batches': int(history_df['skipped_nonfinite_batches'].sum())}

def run_stage_b(model_name: str, channels, selected_epoch: int, outer_train_df: pd.DataFrame, outer_validation_df: pd.DataFrame, device: torch.device, model_dir: Path):
    seed_everything(GLOBAL_SEED)
    model = LightweightResNet10_3D(in_channels=len(channels), base_channels=BASE_CHANNELS, dropout=DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_for_epoch(1), weight_decay=WEIGHT_DECAY)
    amp_settings = resolve_amp_settings(device)
    amp_enabled = bool(amp_settings['enabled'])
    amp_dtype = amp_settings['dtype']
    amp_mode = str(amp_settings['mode'])
    scaler = torch.amp.GradScaler(device.type, enabled=bool(amp_settings['scaler_enabled']))
    print(f'[{model_name}] AMP mode：{amp_mode}')
    train_loader = make_loader(outer_train_df, channels, training=True, seed=GLOBAL_SEED)
    train_evaluation_loader = make_loader(outer_train_df, channels, training=False, seed=GLOBAL_SEED)
    outer_validation_loader = make_loader(outer_validation_df, channels, training=False, seed=GLOBAL_SEED)
    history_records = []
    for epoch in range(1, selected_epoch + 1):
        epoch_start = time.time()
        current_lr = learning_rate_for_epoch(epoch)
        set_optimizer_lr(optimizer, current_lr)
        train_output = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, scaler=scaler, device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        history_records.append({'epoch': epoch, 'learning_rate': current_lr, 'amp_mode': amp_mode, 'train_loss': train_output['loss'], 'train_roc_auc_training_mode': train_output['roc_auc'], 'train_pr_auc_training_mode': train_output['pr_auc'], 'skipped_nonfinite_batches': train_output['skipped_nonfinite_batches'], 'epoch_seconds': float(time.time() - epoch_start)})
        print(f"[{model_name}] Stage B | Epoch {epoch:03d}/{selected_epoch:03d} | LR={current_lr:.6f} | TrainLoss={train_output['loss']:.4f} | TrainAUC={train_output['roc_auc']:.4f} | TrainPR={train_output['pr_auc']:.4f} | SkippedInf={train_output['skipped_nonfinite_batches']}")
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(model_dir / 'stage_B_history.csv', index=False, encoding='utf-8-sig')
    checkpoint_path = model_dir / 'stage_B_outer_train_final_model.pt'
    torch.save({'model_name': model_name, 'channels': list(channels), 'model_state': model.state_dict(), 'selected_epoch': int(selected_epoch), 'training_seed': int(GLOBAL_SEED), 'learning_rate_schedule': '5-epoch warmup + fixed 120-epoch cosine schedule', 'external_validation_used': False, 'amp_mode': amp_mode}, checkpoint_path)
    outer_train_output = evaluate_loader(model=model, loader=train_evaluation_loader, criterion=criterion, device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
    outer_validation_output = evaluate_loader(model=model, loader=outer_validation_loader, criterion=criterion, device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
    outer_train_metrics = calculate_binary_metrics(outer_train_output['y_true'], outer_train_output['probabilities'], threshold=DESCRIPTIVE_THRESHOLD)
    outer_validation_metrics = calculate_binary_metrics(outer_validation_output['y_true'], outer_validation_output['probabilities'], threshold=DESCRIPTIVE_THRESHOLD)
    train_predictions_df = make_prediction_dataframe(output=outer_train_output, source_df=outer_train_df, model_name=model_name, role_name='outer_train_final_eval')
    validation_predictions_df = make_prediction_dataframe(output=outer_validation_output, source_df=outer_validation_df, model_name=model_name, role_name='outer_validation')
    train_predictions_df.to_csv(model_dir / 'outer_train_final_predictions.csv', index=False, encoding='utf-8-sig')
    validation_predictions_df.to_csv(model_dir / 'outer_validation_predictions.csv', index=False, encoding='utf-8-sig')
    center_metrics = {}
    for center in ['B', 'C']:
        center_df = validation_predictions_df.loc[validation_predictions_df['model_center'] == center]
        if len(center_df) == 0:
            continue
        center_metrics[center] = calculate_binary_metrics(center_df['y_true'].to_numpy(), center_df['probability'].to_numpy(), threshold=DESCRIPTIVE_THRESHOLD)
    return {'outer_train_metrics': outer_train_metrics, 'outer_validation_metrics': outer_validation_metrics, 'outer_validation_center_metrics': center_metrics, 'history_df': history_df, 'checkpoint_path': checkpoint_path, 'amp_mode': amp_mode, 'total_skipped_nonfinite_batches': int(history_df['skipped_nonfinite_batches'].sum())}

def save_training_plot(model_name: str, stage_a_history: pd.DataFrame, stage_b_history: pd.DataFrame, output_path: Path) -> None:
    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(stage_a_history['epoch'], stage_a_history['train_loss'], label='Stage A train loss')
    axis.plot(stage_a_history['epoch'], stage_a_history['inner_validation_loss'], label='Stage A inner validation loss')
    axis.plot(stage_b_history['epoch'], stage_b_history['train_loss'], linestyle='--', label='Stage B outer-train loss')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('BCE loss')
    axis.set_title(model_name)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(figure)

def amp_context_from_checkpoint(checkpoint, device):
    amp_mode = str(checkpoint.get('amp_mode', 'FP32')).upper()
    if device.type != 'cuda' or amp_mode == 'FP32':
        return {'enabled': False, 'dtype': torch.float32, 'mode': 'FP32'}
    if amp_mode == 'BF16':
        return {'enabled': True, 'dtype': torch.bfloat16, 'mode': 'BF16'}
    if amp_mode == 'FP16':
        return {'enabled': True, 'dtype': torch.float16, 'mode': 'FP16'}
    return {'enabled': False, 'dtype': torch.float32, 'mode': 'FP32'}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def csv_write(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig')

def checked_ids(series):
    values = pd.to_numeric(series, errors='raise').to_numpy(dtype=float)
    require(np.isfinite(values).all() and np.equal(values, np.floor(values)).all(), '病例ID必须为非空整数')
    return values.astype(np.int64)

def training_settings():
    return {'base_channels': BASE_CHANNELS, 'dropout': DROPOUT, 'batch_size': BATCH_SIZE, 'max_epochs': MAX_EPOCHS, 'min_checkpoint_epoch': MIN_CHECKPOINT_EPOCH, 'min_early_stopping_epoch': MIN_EARLY_STOPPING_EPOCH, 'early_stopping_patience': EARLY_STOPPING_PATIENCE, 'early_stopping_min_delta': EARLY_STOPPING_MIN_DELTA, 'selection_smoothing_window': SELECTION_SMOOTHING_WINDOW, 'base_lr': BASE_LR, 'min_lr': MIN_LR, 'warmup_epochs': WARMUP_EPOCHS, 'weight_decay': WEIGHT_DECAY, 'max_grad_norm': MAX_GRAD_NORM, 'use_amp': USE_AMP, 'amp_precision_mode': AMP_PRECISION_MODE, 'enable_augmentation': ENABLE_AUGMENTATION, 'descriptive_threshold': DESCRIPTIVE_THRESHOLD, 'shape_zyx': list(EXPECTED_SHAPE_ZYX), 'num_workers': NUM_WORKERS}

def environment_record():
    return {'recorded_at': datetime.now().astimezone().isoformat(), 'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'torch': torch.__version__, 'cuda_build': torch.version.cuda, 'cudnn_version_integer': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'statsmodels': statsmodels.__version__, 'scope': 'This revision run, not a reconstruction of original cuDNN', 'determinism': 'Seeded; deterministic algorithms warn_only=True; not bitwise reproducibility guaranteed'}

def initialize_run(audit):
    if RESUME_RUN_DIR is None:
        folder = ROOT / f'CNN_{MODEL_NAME}_{datetime.now():%Y%m%d_%H%M%S_%f}'
        folder.mkdir(exist_ok=False)
        (folder / 'input_snapshot').mkdir()
        for rel, src in audit['input_paths'].items():
            target = folder / 'input_snapshot' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        save_json({'signature': audit['signature'], 'payload': audit['payload'], 'created_at': datetime.now().astimezone().isoformat()}, folder / 'run_config.json')
        csv_write(audit['included'][['patient_id', 'model_center', 'severe_mucositis', 'npz_path']], folder / 'included_489.csv')
        csv_write(pd.DataFrame(audit['payload']['npz_hashes']), folder / 'npz_hashes.csv')
    else:
        folder = Path(RESUME_RUN_DIR).resolve()
        require(folder.is_dir() and folder.parent == ROOT.resolve(), 'Resume folder must be an independent run under configured result_root')
        cfg = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
        require(cfg['signature'] == audit['signature'], '续跑签名不符：输入、参数或模型已变化，请创建新运行')
        for row in audit['payload']['inputs']:
            require(sha256_file(folder / 'input_snapshot' / row['key']) == row['sha256'], '运行快照已改变')
    return folder

def task_record(folder, key, signature):
    marker = folder / f'repeat_{key[0]:02d}' / f'fold_{key[1]:02d}' / 'TASK_COMPLETED.json'
    if not marker.exists():
        return None
    record = json.loads(marker.read_text(encoding='utf-8'))
    require(record['run_signature'] == signature, '已完成任务签名不符')
    require(record['repeat'] == key[0] and record['fold'] == key[1], '任务编号不符')
    for rel, digest in record['file_hashes'].items():
        p = (folder / rel).resolve()
        require(p.is_relative_to(folder.resolve()), '非法任务路径')
        require(p.is_file() and sha256_file(p) == digest, f'已完成任务文件缺失或损坏: {p}')
    return record

def train_tasks(folder, audit, device):
    global REPEAT_NUMBER, FOLD_NUMBER, GLOBAL_SEED, INNER_SPLIT_SEED
    records = []
    for key in sorted(audit['frames']):
        done = task_record(folder, key, audit['signature'])
        if done is not None:
            print('已校验并跳过完成任务', key, flush=True)
            records.append(done)
            continue
        REPEAT_NUMBER, FOLD_NUMBER = key
        cfg = audit['configurations'][key]
        GLOBAL_SEED, INNER_SPLIT_SEED = (int(cfg['training_seed']), int(cfg['inner_split_seed']))
        fold_dir = folder / f'repeat_{key[0]:02d}' / f'fold_{key[1]:02d}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        attempt = fold_dir / f'attempt_{datetime.now():%Y%m%d_%H%M%S_%f}'
        attempt.mkdir(exist_ok=False)
        parts = audit['frames'][key]
        print(f'\n{MODEL_NAME} repeat {key[0]}/4 fold {key[1]}/5 | seed={GLOBAL_SEED}', flush=True)
        with (attempt / 'training_log.txt').open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                sa = run_stage_a(MODEL_NAME, CHANNELS, parts['inner_train'], parts['inner_validation'], device, attempt)
                require(sa['total_skipped_nonfinite_batches'] == 0, 'Stage A出现非有限批次；停止并检查，不自动接受结果')
                sb = run_stage_b(MODEL_NAME, CHANNELS, sa['best_epoch'], parts['outer_train'], parts['outer_validation'], device, attempt)
                require(sb['total_skipped_nonfinite_batches'] == 0, 'Stage B出现非有限批次；停止并检查')
        prediction = pd.read_csv(attempt / 'outer_validation_predictions.csv')
        validate_predictions(prediction, parts['outer_validation'])
        save_training_plot(MODEL_NAME, sa['history_df'], sb['history_df'], attempt / 'training_history.png')
        files = [p for p in attempt.iterdir() if p.is_file()]
        record = {'repeat': key[0], 'fold': key[1], 'run_signature': audit['signature'], 'selected_epoch': sa['best_epoch'], 'training_seed': GLOBAL_SEED, 'amp_mode': sb['amp_mode'], 'parameter_count': sa['parameter_count'], 'checkpoint': str(sb['checkpoint_path'].relative_to(folder)), 'oof_csv': str((attempt / 'outer_validation_predictions.csv').relative_to(folder)), 'file_hashes': {str(p.relative_to(folder)): sha256_file(p) for p in files}, 'completed_at': datetime.now().astimezone().isoformat()}
        save_json(record, fold_dir / 'TASK_COMPLETED.json')
        records.append(record)
        csv_write(pd.DataFrame([{k: r[k] for k in ['repeat', 'fold', 'selected_epoch', 'training_seed', 'amp_mode']} for r in records]), folder / 'task_progress.csv')
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    require(len(records) == 20, '未完成20个折模型，不得评价外部性能')
    return records

class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

def validate_predictions(df, expected):
    require(not df.patient_id.duplicated().any(), '预测中出现重复ID')
    ids = checked_ids(df.patient_id)
    require(set(ids) == set(expected.patient_id), '预测病例集合不符')
    labels = expected.set_index('patient_id').loc[ids, 'severe_mucositis'].to_numpy()
    require(np.array_equal(df.y_true.to_numpy(), labels), '预测结局标签不符')
    prob = df.probability.to_numpy(dtype=float)
    require(np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all(), '预测概率非法')

def aggregate_development(folder, audit, records):
    long = pd.concat([pd.read_csv(folder / r['oof_csv']) for r in records], ignore_index=True)
    require(len(long) == 1236 and (not long.duplicated(['repeat', 'patient_id']).any()), 'OOF必须每名患者每repeat一次')
    require(long.groupby('patient_id').size().eq(4).all(), '每例必须4个OOF预测')
    for _, sub in long.groupby('repeat'):
        validate_predictions(sub, audit['dev'])
    avg = long.groupby('patient_id', as_index=False).agg(y_true=('y_true', 'first'), probability=('probability', 'mean'), prediction_count=('probability', 'size'), model_center=('model_center', 'first'))
    validate_predictions(avg, audit['dev'])
    out = folder / 'aggregate'
    out.mkdir(exist_ok=True)
    csv_write(long, out / 'Development_OOF_long_1236.csv')
    csv_write(avg, out / 'Development_patient_predictions_309.csv')
    return avg

def infer_external(folder, audit, records, device):
    freeze = {'run_signature': audit['signature'], 'models': [{'repeat': r['repeat'], 'fold': r['fold'], 'checkpoint': r['checkpoint'], 'sha256': sha256_file(folder / r['checkpoint'])} for r in records]}
    frozen_path = folder / 'MODELS_FROZEN_BEFORE_EXTERNAL.json'
    if frozen_path.exists():
        require(json.loads(frozen_path.read_text(encoding='utf-8')) == freeze, '外部验证前模型冻结记录已变化')
    else:
        save_json(freeze, frozen_path)
    ext = audit['ext']
    all_predictions = []
    for record in records:
        checkpoint = torch.load(folder / record['checkpoint'], map_location='cpu', weights_only=False)
        require(checkpoint['model_name'] == MODEL_NAME and checkpoint['channels'] == CHANNELS, 'checkpoint模型或通道错误')
        model = LightweightResNet10_3D(len(CHANNELS), BASE_CHANNELS, DROPOUT).to(device)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        loader = make_loader(ext, CHANNELS, False, record['training_seed'])
        amp = amp_context_from_checkpoint(checkpoint, device)
        output = evaluate_loader(model, loader, nn.BCEWithLogitsLoss(), device, amp['enabled'], amp['dtype'])
        df = pd.DataFrame({'patient_id': output['patient_ids'], 'y_true': output['y_true'], 'probability': output['probabilities']})
        validate_predictions(df, ext)
        df['repeat'], df['fold'] = (record['repeat'], record['fold'])
        all_predictions.append(df)
        print('外部推理完成', record['repeat'], record['fold'], flush=True)
        del model, checkpoint, loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    long = pd.concat(all_predictions, ignore_index=True)
    require(len(long) == 3600 and long.groupby('patient_id').size().eq(20).all(), '外部20模型预测不完整')
    avg = long.groupby('patient_id', as_index=False).agg(y_true=('y_true', 'first'), probability=('probability', 'mean'), prediction_count=('probability', 'size'))
    avg['model_center'] = 'A'
    validate_predictions(avg, ext)
    csv_write(long, folder / 'aggregate/External_20model_long_3600.csv')
    csv_write(avg, folder / 'aggregate/External_patient_predictions_180.csv')
    return avg

def metric_vector(y, p):
    return np.array([roc_auc_score(y, p), safe_pr_auc(y, p), brier_score_loss(y, p), log_loss(y, p, labels=[0, 1])])

def summarize_results(folder, audit, dev_pred, ext_pred):
    metrics, calibration, bins, dca = ([], [], [], [])
    metric_names = ['ROC_AUC', 'PR_AUC_trapezoidal', 'Brier', 'Log_loss']
    for cohort, data in [('Development', dev_pred), ('External', ext_pred)]:
        data = data.sort_values('patient_id').reset_index(drop=True)
        y, p = (data.y_true.to_numpy(int), data.probability.to_numpy(float))
        estimate = metric_vector(y, p)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = []
        while len(draws) < BOOTSTRAP_N:
            ix = rng.integers(0, len(y), len(y))
            if len(np.unique(y[ix])) < 2:
                continue
            draws.append(metric_vector(y[ix], p[ix]))
        low, high = np.percentile(draws, [2.5, 97.5], axis=0)
        metrics += [{'cohort': cohort, 'model': MODEL_NAME, 'metric': k, 'estimate': float(v), 'ci_low': float(l), 'ci_high': float(h)} for k, v, l, h in zip(metric_names, estimate, low, high)]
        z = logit(np.clip(p, 1e-06, 1 - 1e-06))
        try:
            citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z).fit()
            joint = sm.GLM(y, sm.add_constant(z, has_constant='add'), family=sm.families.Binomial()).fit()
            require(citl.converged and joint.converged, '校准拟合未收敛')
            calibration.append({'cohort': cohort, 'CITL': float(citl.params[0]), 'intercept_joint': float(joint.params[0]), 'slope': float(joint.params[1]), 'status': 'point_estimates_only'})
        except Exception as exc:
            calibration.append({'cohort': cohort, 'status': 'FAILED: ' + str(exc)})
        for group, ix in enumerate(np.array_split(np.argsort(p, kind='stable'), 10), 1):
            n, events = (len(ix), int(y[ix].sum()))
            lo, hi = proportion_confint(events, n, method='wilson')
            bins.append({'cohort': cohort, 'group': group, 'n': n, 'events': events, 'mean_probability': float(p[ix].mean()), 'observed_rate': events / n, 'wilson_low': float(lo), 'wilson_high': float(hi)})
        for threshold in np.arange(1, 100) / 100:
            selected = p >= threshold
            tp, fp = (int(((y == 1) & selected).sum()), int(((y == 0) & selected).sum()))
            dca.append({'cohort': cohort, 'threshold': threshold, 'net_benefit': tp / len(y) - fp / len(y) * threshold / (1 - threshold), 'treat_all': y.mean() - (1 - y.mean()) * threshold / (1 - threshold), 'treat_none': 0.0})
    frames = {'metrics_95CI': pd.DataFrame(metrics), 'calibration': pd.DataFrame(calibration), 'calibration_bins': pd.DataFrame(bins), 'DCA': pd.DataFrame(dca)}
    for name, frame in frames.items():
        csv_write(frame, folder / 'aggregate' / f'{name}.csv')
    with pd.ExcelWriter(folder / 'aggregate/CNN_summary.xlsx', engine='openpyxl') as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return frames
from sklearn.metrics import roc_curve
def development_youden_threshold(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    finite = np.isfinite(thresholds)
    fpr = fpr[finite]
    tpr = tpr[finite]
    thresholds = thresholds[finite]
    youden = tpr - fpr
    best_value = np.max(youden)
    candidate_indices = np.where(np.isclose(youden, best_value, atol=1e-12, rtol=0))[0]
    if len(candidate_indices) > 1:
        chosen_index = candidate_indices[np.argmin(np.abs(thresholds[candidate_indices] - 0.5))]
    else:
        chosen_index = int(candidate_indices[0])
    return {'threshold': float(thresholds[chosen_index]), 'development_sensitivity': float(tpr[chosen_index]), 'development_specificity': float(1.0 - fpr[chosen_index]), 'development_youden': float(youden[chosen_index])}