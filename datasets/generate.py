# Dataset generator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datasets.bars_stripes import BarsAndStripes


def generate_datasets(grid_sizes, test_fraction=0.2, output_dir='data', seed=42):
    """Generate train/test splits for Bars & Stripes datasets."""
    Path(output_dir).mkdir(exist_ok=True)
    
    for N in grid_sizes:
        dataset = BarsAndStripes(N)
        train, test = dataset.split_train_test(test_fraction, random_state=seed)
        
        # Save as flattened voltage arrays
        train_voltages = dataset.to_voltages(train.reshape(len(train), -1))
        test_voltages = dataset.to_voltages(test.reshape(len(test), -1))
        
        np.save(f'{output_dir}/train_N{N}.npy', train_voltages)
        np.save(f'{output_dir}/test_N{N}.npy', test_voltages)
        
        print(f"N={N}: {len(train)} train, {len(test)} test → {output_dir}/")


if __name__ == "__main__":
    generate_datasets(
        grid_sizes=[3, 4, 5],
        test_fraction=0.2,
        output_dir='data',
        seed=42
    )
