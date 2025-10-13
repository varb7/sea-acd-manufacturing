# RFCI Integration Guide

This guide explains how to use your Tetrad RFCI algorithm with the SEA pipeline.

## What Was Implemented

1. **RFCI Module** (`src/rfci_module.py`)
   - Complete Tetrad RFCI implementation using PyTetrad
   - Robust JVM bootstrap and data type handling
   - `run_rfci()` function with clean API

2. **RFCI Wrapper Function** (`src/data/utils.py`)
   - `run_rfci()` function that wraps the RFCI module
   - Handles data format conversion (numpy array → RFCI input)
   - Returns PAG format adjacency matrix compatible with SEA pipeline

3. **Edge Mapping** (`src/data/utils.py`)
   - `edge_map_rfci_pag` maps PAG format to SEA's token space
   - `(2,0)` → forward edge, `(0,2)` → backward edge, `(1,1)` → undirected, `(4,4)` → ambiguous

4. **Algorithm Registration** (`src/data/dataset.py`)
   - Added RFCI to `get_run_alg()` function
   - RFCI is treated as observational algorithm (like FCI/GES)

5. **CLI Support** (`src/args.py`)
   - Added `"rfci"` to algorithm choices
   - Can be selected via `--algorithm rfci`

6. **Configuration** (`config/aggregator_tf_rfci.yaml`)
   - Ready-to-use config file for RFCI experiments
   - Based on FCI config with appropriate settings

## Prerequisites

1. **Install Tetrad dependencies**:
   ```bash
   pip install jpype pandas
   # Install py-tetrad or set TETRAD_JAR environment variable
   ```

2. **Ensure RFCI module is accessible**:
   - The `rfci_module.py` file is located in `src/` directory
   - No additional path configuration needed

## Usage

### 1. Training with RFCI
```bash
cd sea-reproduce
python src/train.py \
  --config_file config/aggregator_tf_rfci.yaml \
  --save_path checkpoints/my_rfci_run \
  --gpu 0
```

### 2. Inference with RFCI
```bash
python src/inference.py \
  --config_file config/aggregator_tf_rfci.yaml \
  --run_name aggregator_tf_rfci \
  --gpu 0 \
  --checkpoint_path checkpoints/my_rfci_run/model_best.ckpt
```

### 3. Using CLI directly
```bash
python src/inference.py \
  --algorithm rfci \
  --data_file data/intervention_8160.csv \
  --gpu 0
```

## Configuration Options

The RFCI wrapper accepts these parameters (modify in `src/data/utils.py`):

- `alpha`: significance level for independence tests (default: 0.05)
- `depth`: max conditioning set size (default: -1 for unlimited)
- `include_undirected`: include undirected edges (default: True)
- `count_partial`: count partial orientations (default: False)

## Testing

Run the integration test:
```bash
cd sea-reproduce
python test_rfci_integration.py
```

This will verify:
- RFCI wrapper function works
- Algorithm selector recognizes RFCI
- Edge mapping produces correct tokens

## Troubleshooting

1. **Import Error**: If `rfci_module` not found:
   - Ensure `rfci_module.py` is in `src/` directory
   - Check that Python can import the module

2. **JVM Issues**: If Tetrad JVM fails to start:
   - Install `py-tetrad` package
   - Or set `TETRAD_JAR` environment variable
   - Or place Tetrad jar in `./resources/` folder

3. **Memory Issues**: If JVM runs out of memory:
   - Reduce `fci_batch_size` in config
   - Set `num_workers: 0` to avoid multiple JVM instances

## File Structure

```
sea-reproduce/
├── src/rfci_module.py         # RFCI implementation
├── src/data/utils.py          # RFCI wrapper + edge mapping
├── src/data/dataset.py        # Algorithm selector
├── src/args.py                # CLI arguments
├── config/aggregator_tf_rfci.yaml  # RFCI config
├── test_rfci_integration.py   # Integration tests
└── RFCI_INTEGRATION_GUIDE.md  # This guide
```

## Next Steps

1. Test with your data using the provided config
2. Adjust RFCI parameters in the wrapper function
3. Train the aggregator model with RFCI
4. Compare results with FCI baseline
