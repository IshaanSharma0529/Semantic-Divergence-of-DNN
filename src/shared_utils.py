"""
Shared utilities for CrossVision-Attacks experiments.

Extracts common data-loading, model-loading, preprocessing, and attack
functions from cnn-attacks.ipynb so that all experiment scripts share
identical logic.  DO NOT modify — any change here affects every experiment.
"""

import os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────
# GPU MEMORY MANAGEMENT  (avoid OOM on ≤4 GB GPUs)
# ────────────────────────────────────────────────────────────────
try:
    _gpus = tf.config.list_physical_devices('GPU')
    for _gpu in _gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
except RuntimeError:
    pass  # must be set before GPUs are initialised

# ────────────────────────────────────────────────────────────────
# CONSTANTS  (mirror cnn-attacks.ipynb exactly)
# ────────────────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 8          # reduced from 32 for 4 GB VRAM (GradientTape doubles memory)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
NUM_CLASSES = 101

MODEL_NAMES = ['VGG19', 'ResNet50', 'DenseNet121']
EPSILONS = [0.0, 0.005, 0.01, 0.02, 0.04]
ADV_EPSILONS = [0.005, 0.01, 0.02, 0.04]
PGD_EPSILONS = [0.005, 0.01, 0.02, 0.04]
PGD_STEPS = 10

PREPROCESS_FNS = {
    'VGG19': vgg_preprocess,
    'ResNet50': resnet_preprocess,
    'DenseNet121': densenet_preprocess,
}
PREPROCESS_MODE = {
    'VGG19': 'caffe',
    'ResNet50': 'caffe',
    'DenseNet121': 'torch',
}

# ────────────────────────────────────────────────────────────────
# PATH RESOLUTION
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # d:\Research-paper
MODEL_TRAINING_DIR = PROJECT_ROOT / 'Model Training'
CHECKPOINT_DIR = MODEL_TRAINING_DIR / 'checkpoints'
DATA_DIR = MODEL_TRAINING_DIR / 'caltech101_data'
SPLIT_FILE = MODEL_TRAINING_DIR / 'frozen_split_indices.json'
BASELINES_FILE = MODEL_TRAINING_DIR / 'clean_baselines' / 'clean_baselines.json'
RESULTS_DIR = MODEL_TRAINING_DIR / 'fgsm_results'
CLASS_NAMES_FILE = MODEL_TRAINING_DIR / 'final_models' / 'class_names.txt'


def get_class_names() -> List[str]:
    """Load class names from file or derive from dataset directory."""
    if CLASS_NAMES_FILE.exists():
        with open(CLASS_NAMES_FILE, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        return names
    # Fallback: derive from image root
    image_root = find_image_root(DATA_DIR)
    exclude = {'__MACOSX', '.DS_Store', 'BACKGROUND_Google', '__pycache__'}
    return sorted([
        d.name for d in image_root.iterdir()
        if d.is_dir() and d.name not in exclude and not d.name.startswith('.')
    ])


# ────────────────────────────────────────────────────────────────
# DATA LOADING  (exact copy from cnn-attacks.ipynb)
# ────────────────────────────────────────────────────────────────

def find_image_root(base_path: Path) -> Path:
    """Find 101_ObjectCategories directory."""
    for obj_cat in base_path.rglob('101_ObjectCategories'):
        if obj_cat.is_dir():
            subdirs = [d for d in obj_cat.iterdir()
                       if d.is_dir() and d.name != '__MACOSX']
            if len(subdirs) > 10:
                return obj_cat
    raise FileNotFoundError(f"101_ObjectCategories not found under {base_path}")


def load_test_split() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the frozen test split paths and labels.

    Returns:
        test_paths, test_labels, class_names
    """
    image_root = find_image_root(DATA_DIR)
    exclude = {'__MACOSX', '.DS_Store', 'BACKGROUND_Google', '__pycache__'}
    class_names = sorted([
        d.name for d in image_root.iterdir()
        if d.is_dir() and d.name not in exclude and not d.name.startswith('.')
    ])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_paths, all_labels = [], []
    for class_name in class_names:
        class_dir = image_root / class_name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                all_paths.append(str(img_path))
                all_labels.append(class_to_idx[class_name])
    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    with open(SPLIT_FILE, 'r') as f:
        saved = json.load(f)
    indices = np.array(saved['indices'])
    assert saved['seed'] == SEED

    all_paths = all_paths[indices]
    all_labels = all_labels[indices]

    train_size = int(TRAIN_SPLIT * len(all_paths))
    val_size = int(VAL_SPLIT * len(all_paths))

    test_paths = all_paths[train_size + val_size:]
    test_labels = all_labels[train_size + val_size:]
    return test_paths, test_labels, class_names


def build_raw_test_dataset(paths: np.ndarray, labels: np.ndarray,
                           img_size: Tuple[int, int] = IMG_SIZE,
                           batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Build deterministic test dataset yielding RAW [0,1] images + labels."""
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ────────────────────────────────────────────────────────────────
# PREPROCESSING  (exact copy from cnn-attacks.ipynb)
# ────────────────────────────────────────────────────────────────

@tf.function
def preprocess_for_model(images_01: tf.Tensor, model_name: str) -> tf.Tensor:
    """Convert [0,1] images to model-specific input format (differentiable)."""
    if model_name == 'VGG19' or model_name == 'ResNet50':
        x = images_01 * 255.0
        x = x[..., ::-1]
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        x = x - mean
    else:  # DenseNet121
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        x = (images_01 - mean) / std
    return x


# ────────────────────────────────────────────────────────────────
# MODEL LOADING
# ────────────────────────────────────────────────────────────────

def load_models() -> Dict[str, keras.Model]:
    """Load all 3 CNN models from checkpoints."""
    models = {}
    for name in MODEL_NAMES:
        ckpt = CHECKPOINT_DIR / f"{name}_best.h5"
        assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
        models[name] = keras.models.load_model(ckpt)
        print(f"  Loaded {name}: {models[name].count_params():,} params")
    return models


def build_logits_model(model: keras.Model) -> keras.Model:
    """Create model outputting raw logits (strip softmax)."""
    last_layer = model.layers[-1]

    if isinstance(last_layer, (tf.keras.layers.Softmax,
                               tf.keras.layers.Activation)):
        return tf.keras.Model(inputs=model.input,
                              outputs=last_layer.input)

    if isinstance(last_layer, tf.keras.layers.Dense):
        act_name = getattr(last_layer.activation, '__name__', '')
        if act_name == 'softmax':
            new_dense = tf.keras.layers.Dense(
                units=last_layer.units, activation='linear',
                use_bias=last_layer.use_bias, dtype='float32',
                name=last_layer.name + '_logits',
            )
            logits_output = new_dense(last_layer.input)
            logits_model = tf.keras.Model(inputs=model.input,
                                          outputs=logits_output)
            new_dense.set_weights(last_layer.get_weights())
            return logits_model

    return tf.keras.Model(inputs=model.input, outputs=model.output)


def build_all_logits_models(models: Dict[str, keras.Model]) -> Dict[str, keras.Model]:
    """Build logits models for all architectures."""
    logits_models = {}
    for name in MODEL_NAMES:
        logits_models[name] = build_logits_model(models[name])
    return logits_models


# ────────────────────────────────────────────────────────────────
# ATTACK FUNCTIONS  (exact copy from cnn-attacks.ipynb)
# ────────────────────────────────────────────────────────────────

def fgsm_attack(model: keras.Model, images: tf.Tensor,
                labels: tf.Tensor, epsilon: float,
                model_name: str) -> tf.Tensor:
    """FGSM attack: x_adv = clip(x + eps*sign(grad L), 0, 1)."""
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preprocessed = preprocess_for_model(images, model_name)
        predictions = model(preprocessed, training=False)
        predictions = tf.cast(predictions, tf.float32)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

    gradients = tape.gradient(loss, images)
    signed_grad = tf.sign(gradients)
    adv_images = images + epsilon * signed_grad
    adv_images = tf.clip_by_value(adv_images, 0.0, 1.0)
    return adv_images


# Pre-build PGD loss
_pgd_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def pgd_attack(logits_model: keras.Model, images: tf.Tensor,
               labels: tf.Tensor, epsilon: float,
               steps: int, step_size: float,
               model_name: str) -> tf.Tensor:
    """PGD attack (Madry et al.) on logits."""
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int32)

    noise = tf.random.uniform(tf.shape(images), minval=-epsilon, maxval=epsilon)
    adv_images = images + noise
    adv_images = tf.clip_by_value(adv_images, 0.0, 1.0)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            preprocessed = preprocess_for_model(adv_images, model_name)
            logits = logits_model(preprocessed, training=False)
            logits = tf.cast(logits, tf.float32)
            loss = _pgd_loss_fn(labels, logits)

        gradients = tape.gradient(loss, adv_images)
        adv_images = adv_images + step_size * tf.sign(gradients)
        perturbation = tf.clip_by_value(adv_images - images, -epsilon, epsilon)
        adv_images = images + perturbation
        adv_images = tf.clip_by_value(adv_images, 0.0, 1.0)

    return adv_images


def deepfool_attack(logits_model: keras.Model, image: tf.Tensor,
                    label: int, model_name: str,
                    max_iter: int = 50, num_candidates: int = 10,
                    overshoot: float = 0.02) -> Tuple[bool, float, int]:
    """DeepFool L2 attack on raw logits."""
    x_adv = tf.identity(image)
    num_classes = logits_model.output_shape[-1]
    iterations = 0
    fooled = False

    for _ in range(max_iter):
        x_curr = tf.identity(x_adv)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_curr)
            preprocessed = preprocess_for_model(
                tf.expand_dims(x_curr, 0), model_name)
            logits = logits_model(preprocessed, training=False)
            logits_vec = tf.squeeze(tf.cast(logits, tf.float32), 0)

        y = int(tf.argmax(logits_vec).numpy())
        if y != label:
            fooled = True
            del tape
            break

        logit_vals = logits_vec.numpy()
        mask_y = tf.one_hot(y, num_classes, dtype=tf.float32)
        grad_y = tape.gradient(logits_vec, x_curr, output_gradients=mask_y)
        if grad_y is None:
            del tape
            break

        diffs = [(k, abs(logit_vals[k] - logit_vals[y]))
                 for k in range(num_classes) if k != y]
        diffs.sort(key=lambda pair: pair[1])
        candidates = [k for k, _ in diffs[:num_candidates]]

        best_dist = np.inf
        best_w = None
        best_w_norm = 0.0

        for k in candidates:
            mask_k = tf.one_hot(k, num_classes, dtype=tf.float32)
            grad_k = tape.gradient(logits_vec, x_curr, output_gradients=mask_k)
            if grad_k is None:
                continue

            w_k = grad_k - grad_y
            f_k = float(logit_vals[k] - logit_vals[y])
            w_k_flat = tf.reshape(w_k, [-1])
            w_k_norm = float(tf.norm(w_k_flat).numpy())
            if w_k_norm < 1e-12:
                continue

            dist_k = abs(f_k) / w_k_norm
            if dist_k < best_dist:
                best_dist = dist_k
                best_w = w_k
                best_w_norm = w_k_norm

        del tape

        if best_w is None:
            break

        r_i = (1.0 + overshoot) * (best_dist + 1e-4) * best_w / best_w_norm
        x_adv = tf.clip_by_value(x_adv + r_i, 0.0, 1.0)
        iterations += 1

    if not fooled:
        final_pre = preprocess_for_model(tf.expand_dims(x_adv, 0), model_name)
        final_logits = logits_model(final_pre, training=False)
        final_pred = int(tf.argmax(tf.cast(final_logits, tf.float32)[0]).numpy())
        fooled = (final_pred != label)

    l2_norm = float(tf.norm(tf.reshape(x_adv - image, [-1])).numpy())
    return fooled, l2_norm, iterations


# ────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ────────────────────────────────────────────────────────────────

def evaluate_under_attack(model: keras.Model, dataset: tf.data.Dataset,
                          epsilon: float, model_name: str) -> dict:
    """Evaluate model under FGSM attack. Returns {accuracy, fooling_rate, clean_conf, adv_conf}."""
    total = 0
    correct_adv = 0
    fooled = 0
    clean_confs, adv_confs = [], []

    for images, labels in dataset:
        batch_size = tf.shape(images)[0].numpy()
        labels_int = tf.cast(labels, tf.int32)

        clean_preprocessed = preprocess_for_model(images, model_name)
        clean_preds = tf.cast(model(clean_preprocessed, training=False), tf.float32)
        clean_classes = tf.argmax(clean_preds, axis=1, output_type=tf.int32)
        clean_confidence = tf.reduce_max(clean_preds, axis=1)

        if epsilon > 0:
            adv_images = fgsm_attack(model, images, labels, epsilon, model_name)
        else:
            adv_images = images

        adv_preprocessed = preprocess_for_model(adv_images, model_name)
        adv_preds = tf.cast(model(adv_preprocessed, training=False), tf.float32)
        adv_classes = tf.argmax(adv_preds, axis=1, output_type=tf.int32)
        adv_confidence = tf.reduce_max(adv_preds, axis=1)

        correct_adv += tf.reduce_sum(tf.cast(tf.equal(adv_classes, labels_int), tf.int32)).numpy()
        fooled += tf.reduce_sum(tf.cast(tf.not_equal(adv_classes, clean_classes), tf.int32)).numpy()
        clean_confs.append(clean_confidence.numpy())
        adv_confs.append(adv_confidence.numpy())
        total += batch_size

    return {
        'accuracy': correct_adv / total,
        'fooling_rate': fooled / total,
        'clean_conf': float(np.mean(np.concatenate(clean_confs))),
        'adv_conf': float(np.mean(np.concatenate(adv_confs))),
    }


def evaluate_pgd_attack(logits_model, model, dataset, epsilon,
                         steps, step_size, model_name) -> dict:
    """Evaluate model under PGD attack. Returns same format as FGSM eval."""
    total = 0
    correct_adv = 0
    fooled = 0
    clean_confs, adv_confs = [], []

    for images, labels in dataset:
        batch_size = tf.shape(images)[0].numpy()
        labels_int = tf.cast(labels, tf.int32)

        clean_preprocessed = preprocess_for_model(images, model_name)
        clean_preds = tf.cast(model(clean_preprocessed, training=False), tf.float32)
        clean_classes = tf.argmax(clean_preds, axis=1, output_type=tf.int32)
        clean_confidence = tf.reduce_max(clean_preds, axis=1)

        adv_images = pgd_attack(logits_model, images, labels, epsilon,
                                steps, step_size, model_name)

        adv_preprocessed = preprocess_for_model(adv_images, model_name)
        adv_preds = tf.cast(model(adv_preprocessed, training=False), tf.float32)
        adv_classes = tf.argmax(adv_preds, axis=1, output_type=tf.int32)
        adv_confidence = tf.reduce_max(adv_preds, axis=1)

        correct_adv += tf.reduce_sum(tf.cast(tf.equal(adv_classes, labels_int), tf.int32)).numpy()
        fooled += tf.reduce_sum(tf.cast(tf.not_equal(adv_classes, clean_classes), tf.int32)).numpy()
        clean_confs.append(clean_confidence.numpy())
        adv_confs.append(adv_confidence.numpy())
        total += batch_size

    return {
        'accuracy': correct_adv / total,
        'fooling_rate': fooled / total,
        'clean_conf': float(np.mean(np.concatenate(clean_confs))),
        'adv_conf': float(np.mean(np.concatenate(adv_confs))),
    }


def seed_everything(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
