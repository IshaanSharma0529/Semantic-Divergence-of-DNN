"""
Task 2 — Build Adversarial Bank
================================
Generate adversarial examples for all (model × attack × epsilon)
combinations and store full metadata including adv_pred (which the
original cnn-attacks.ipynb never recorded).

The bank is essential for:
  - Task 3  (transfer matrix: source adversarials evaluated on target models)
  - Task 4  (confusion direction: needs adv_pred to build confusion matrices)

Writes to:
    Model Training/adversarial_bank/{model}/{attack}_eps{eps}/metadata.json
    Model Training/adversarial_bank/{model}/{attack}_eps{eps}/images/  (optional)

Run from project root:
    python experiments/build_adversarial_bank.py [--save-images]
"""

import os, sys, time, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.shared_utils import (
    load_test_split, build_raw_test_dataset, load_models,
    build_all_logits_models, preprocess_for_model,
    fgsm_attack, pgd_attack, deepfool_attack,
    MODEL_NAMES, ADV_EPSILONS, PGD_STEPS, SEED,
    seed_everything, MODEL_TRAINING_DIR,
)
from src.attacks.adversarial_bank import AdversarialBank


BANK_DIR = MODEL_TRAINING_DIR / 'adversarial_bank'


def build_fgsm_bank(models, bank, raw_test_ds, class_names,
                     test_labels_flat, save_images):
    """Generate FGSM adversarial bank for all models and epsilons."""
    print("\n--- FGSM Bank ---")
    for name in MODEL_NAMES:
        model = models[name]
        for eps in ADV_EPSILONS:
            print(f"  {name} ε={eps} ... ", end='', flush=True)
            t0 = time.time()
            idx_offset = 0
            for images, labels in raw_test_ds:
                labels_int = tf.cast(labels, tf.int32)
                bs = tf.shape(images)[0].numpy()

                # clean predictions
                clean_pre = preprocess_for_model(images, name)
                clean_preds_prob = tf.cast(model(clean_pre, training=False), tf.float32)
                clean_classes = tf.argmax(clean_preds_prob, axis=1).numpy()
                clean_confs = tf.reduce_max(clean_preds_prob, axis=1).numpy()

                # adversarial
                adv_images = fgsm_attack(model, images, labels, eps, name)
                adv_pre = preprocess_for_model(adv_images, name)
                adv_preds_prob = tf.cast(model(adv_pre, training=False), tf.float32)
                adv_classes = tf.argmax(adv_preds_prob, axis=1).numpy()
                adv_confs = tf.reduce_max(adv_preds_prob, axis=1).numpy()

                for j in range(bs):
                    true_lab = int(labels_int[j].numpy())
                    bank.add_record(
                        model_name=name, attack='fgsm', epsilon=eps,
                        sample_idx=idx_offset + j,
                        true_label=true_lab,
                        true_class=class_names[true_lab],
                        clean_pred=int(clean_classes[j]),
                        clean_class=class_names[int(clean_classes[j])],
                        clean_confidence=float(clean_confs[j]),
                        adv_pred=int(adv_classes[j]),
                        adv_class=class_names[int(adv_classes[j])],
                        adv_confidence=float(adv_confs[j]),
                        fooled=bool(adv_classes[j] != clean_classes[j]),
                        linf_norm=float(tf.reduce_max(
                            tf.abs(adv_images[j] - images[j])).numpy()),
                        l2_norm=float(tf.norm(
                            tf.reshape(adv_images[j] - images[j], [-1])).numpy()),
                        adv_image=adv_images[j].numpy() if save_images else None,
                    )
                idx_offset += bs
            print(f"done ({time.time()-t0:.1f}s)")


def build_pgd_bank(models, logits_models, bank, raw_test_ds,
                    class_names, save_images):
    """Generate PGD adversarial bank for all models and epsilons."""
    print("\n--- PGD Bank ---")
    for name in MODEL_NAMES:
        model = models[name]
        logits_model = logits_models[name]
        for eps in ADV_EPSILONS:
            step_size = eps / 4.0
            print(f"  {name} ε={eps} ... ", end='', flush=True)
            t0 = time.time()
            idx_offset = 0
            for images, labels in raw_test_ds:
                labels_int = tf.cast(labels, tf.int32)
                bs = tf.shape(images)[0].numpy()

                clean_pre = preprocess_for_model(images, name)
                clean_preds_prob = tf.cast(model(clean_pre, training=False), tf.float32)
                clean_classes = tf.argmax(clean_preds_prob, axis=1).numpy()
                clean_confs = tf.reduce_max(clean_preds_prob, axis=1).numpy()

                adv_images = pgd_attack(logits_model, images, labels, eps,
                                        PGD_STEPS, step_size, name)
                adv_pre = preprocess_for_model(adv_images, name)
                adv_preds_prob = tf.cast(model(adv_pre, training=False), tf.float32)
                adv_classes = tf.argmax(adv_preds_prob, axis=1).numpy()
                adv_confs = tf.reduce_max(adv_preds_prob, axis=1).numpy()

                for j in range(bs):
                    true_lab = int(labels_int[j].numpy())
                    bank.add_record(
                        model_name=name, attack='pgd', epsilon=eps,
                        sample_idx=idx_offset + j,
                        true_label=true_lab,
                        true_class=class_names[true_lab],
                        clean_pred=int(clean_classes[j]),
                        clean_class=class_names[int(clean_classes[j])],
                        clean_confidence=float(clean_confs[j]),
                        adv_pred=int(adv_classes[j]),
                        adv_class=class_names[int(adv_classes[j])],
                        adv_confidence=float(adv_confs[j]),
                        fooled=bool(adv_classes[j] != clean_classes[j]),
                        linf_norm=float(tf.reduce_max(
                            tf.abs(adv_images[j] - images[j])).numpy()),
                        l2_norm=float(tf.norm(
                            tf.reshape(adv_images[j] - images[j], [-1])).numpy()),
                        adv_image=adv_images[j].numpy() if save_images else None,
                    )
                idx_offset += bs
            print(f"done ({time.time()-t0:.1f}s)")


def build_deepfool_bank(models, logits_models, bank, raw_test_ds,
                        class_names, save_images):
    """Generate DeepFool adversarial bank (single-image attack)."""
    print("\n--- DeepFool Bank ---")
    for name in MODEL_NAMES:
        model = models[name]
        logits_model = logits_models[name]
        print(f"  {name} ... ", flush=True)
        t0 = time.time()
        idx_offset = 0
        for images, labels in raw_test_ds:
            labels_int = tf.cast(labels, tf.int32)
            bs = tf.shape(images)[0].numpy()

            clean_pre = preprocess_for_model(images, name)
            clean_preds_prob = tf.cast(model(clean_pre, training=False), tf.float32)
            clean_classes = tf.argmax(clean_preds_prob, axis=1).numpy()
            clean_confs = tf.reduce_max(clean_preds_prob, axis=1).numpy()

            for j in range(bs):
                img = images[j]
                true_lab = int(labels_int[j].numpy())

                fooled, l2_norm, iters = deepfool_attack(
                    logits_model, img, true_lab, name)

                # get adversarial prediction
                # (reconstruct adv image from the attack would be ideal
                #  but deepfool_attack returns only metrics; re-run minimally)
                # Instead, we check what the model predicts on the image
                # post-perturbation -- but we don't have the adv image.
                # For DeepFool, we set eps=0 sentinel in metadata.
                bank.add_record(
                    model_name=name, attack='deepfool', epsilon=0.0,
                    sample_idx=idx_offset + j,
                    true_label=true_lab,
                    true_class=class_names[true_lab],
                    clean_pred=int(clean_classes[j]),
                    clean_class=class_names[int(clean_classes[j])],
                    clean_confidence=float(clean_confs[j]),
                    adv_pred=-1,  # DeepFool doesn't return adv image in shared_utils
                    adv_class='unknown',
                    adv_confidence=-1.0,
                    fooled=bool(fooled),
                    l2_norm=l2_norm,
                )
                if (idx_offset + j) % 100 == 0:
                    print(f"    [{idx_offset + j}] fooled={fooled} l2={l2_norm:.4f}")
            idx_offset += bs
        print(f"  {name} done ({time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-images', action='store_true',
                        help='Save adversarial images as .npy (requires ~10 GB)')
    parser.add_argument('--skip-deepfool', action='store_true',
                        help='Skip DeepFool (slow single-image attack)')
    args = parser.parse_args()

    print("=" * 60)
    print("TASK 2: Build Adversarial Bank")
    print("=" * 60)
    seed_everything(SEED)

    models = load_models()
    logits_models = build_all_logits_models(models)
    test_paths, test_labels, class_names = load_test_split()
    raw_test_ds = build_raw_test_dataset(test_paths, test_labels)
    print(f"  Test samples: {len(test_labels)}, Classes: {len(class_names)}")

    bank = AdversarialBank(BANK_DIR, save_images=args.save_images)

    build_fgsm_bank(models, bank, raw_test_ds, class_names,
                    test_labels, args.save_images)
    build_pgd_bank(models, logits_models, bank, raw_test_ds,
                   class_names, args.save_images)
    if not args.skip_deepfool:
        build_deepfool_bank(models, logits_models, bank, raw_test_ds,
                            class_names, args.save_images)

    bank.flush_all()

    # ── summary ──
    summary = bank.summary()
    summary_file = BANK_DIR / 'bank_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Bank summary saved to {summary_file}")
    for key, stats in summary.items():
        print(f"  {key}: {stats['total']} samples, "
              f"fooling={stats['fooling_rate']:.2%}")

    print("\n" + "=" * 60)
    print("  Adversarial bank created successfully.")
    print("=" * 60)


if __name__ == '__main__':
    main()
