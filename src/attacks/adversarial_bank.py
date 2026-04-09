"""
Adversarial Bank — persistent storage & metadata for all adversarial examples.

Stores per-sample records:
  - sample_idx, true_label, true_class
  - clean_pred, clean_confidence
  - adv_pred, adv_class, adv_confidence   (MISSING in original code)
  - attack_type, epsilon, fooled
  - l2_norm, linf_norm
  - (optional) adversarial image path (saved as .npy)

The bank is serialised as a JSON file per (model, attack, epsilon) and
optionally as a single HDF5 chunked dataset.
"""

import os, json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class AdversarialBank:
    """Manages adversarial example metadata and optional image storage.

    Parameters
    ----------
    out_dir : Path or str
        Directory for the bank.  Structure created:
          {out_dir}/{model}/{attack}_eps{eps}/metadata.json
          {out_dir}/{model}/{attack}_eps{eps}/images/  (optional)
    save_images : bool
        If True, adversarial images (numpy float32) are saved as .npy.
    """

    def __init__(self, out_dir, save_images: bool = False):
        self.out_dir = Path(out_dir)
        self.save_images = save_images
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, list] = {}       # key → list[dict]

    def _key(self, model_name: str, attack: str, epsilon: float) -> str:
        return f"{model_name}/{attack}_eps{epsilon}"

    def add_record(self, model_name: str, attack: str, epsilon: float,
                   sample_idx: int, true_label: int, true_class: str,
                   clean_pred: int, clean_class: str, clean_confidence: float,
                   adv_pred: int, adv_class: str, adv_confidence: float,
                   fooled: bool,
                   l2_norm: Optional[float] = None,
                   linf_norm: Optional[float] = None,
                   adv_image: Optional[np.ndarray] = None):
        key = self._key(model_name, attack, epsilon)
        if key not in self._records:
            self._records[key] = []

        rec = {
            'sample_idx': int(sample_idx),
            'true_label': int(true_label),
            'true_class': true_class,
            'clean_pred': int(clean_pred),
            'clean_class': clean_class,
            'clean_confidence': round(float(clean_confidence), 6),
            'adv_pred': int(adv_pred),
            'adv_class': adv_class,
            'adv_confidence': round(float(adv_confidence), 6),
            'fooled': bool(fooled),
        }
        if l2_norm is not None:
            rec['l2_norm'] = round(float(l2_norm), 8)
        if linf_norm is not None:
            rec['linf_norm'] = round(float(linf_norm), 8)

        self._records[key].append(rec)

        if self.save_images and adv_image is not None:
            img_dir = self.out_dir / key / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)
            np.save(img_dir / f'{sample_idx}.npy', adv_image.astype(np.float32))

    def flush_all(self):
        """Write all accumulated metadata to disk."""
        for key, records in self._records.items():
            out_path = self.out_dir / key
            out_path.mkdir(parents=True, exist_ok=True)
            meta_file = out_path / 'metadata.json'
            with open(meta_file, 'w') as f:
                json.dump({
                    'count': len(records),
                    'fooled_count': sum(1 for r in records if r['fooled']),
                    'records': records,
                }, f, indent=2)
        print(f"  Flushed {len(self._records)} bank slices to {self.out_dir}")

    def summary(self) -> Dict[str, dict]:
        """Return per-slice summary stats."""
        out = {}
        for key, records in self._records.items():
            total = len(records)
            fooled = sum(1 for r in records if r['fooled'])
            out[key] = {
                'total': total,
                'fooled': fooled,
                'fooling_rate': round(fooled / total, 4) if total > 0 else 0.0,
                'mean_adv_confidence': round(
                    np.mean([r['adv_confidence'] for r in records]), 4),
                'mean_clean_confidence': round(
                    np.mean([r['clean_confidence'] for r in records]), 4),
            }
        return out
