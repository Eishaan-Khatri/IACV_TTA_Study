# TODO

## Phase 0 - Setup

- [ ] Create virtual environment
- [ ] Install PyTorch / torchvision / numpy / pandas / matplotlib / tqdm / pyyaml
- [ ] Download CIFAR-10 through torchvision
- [ ] Download CIFAR-10-C `.npy` files
- [ ] Confirm whether GPU is available

## Phase 1 - Source Model

- [ ] Implement compact CIFAR model
- [ ] Train on clean CIFAR-10
- [ ] Save checkpoint
- [ ] Record clean test accuracy

## Phase 2 - Evaluation Loader

- [ ] Implement CIFAR-10-C loader
- [ ] Verify labels align with corruption arrays
- [ ] Evaluate source-only on first-pass corruptions

## Phase 3 - Baselines

- [ ] Implement BN Adapt
- [ ] Implement Tent-style entropy minimization
- [ ] Run baselines on 5 corruptions x 5 severities

## Phase 4 - UG-Tent

- [ ] Implement confidence/entropy gating
- [ ] Log update ratio
- [ ] Compare against Tent

## Phase 5 - Reporting

- [ ] Generate results CSV
- [ ] Generate summary tables
- [ ] Generate plots
- [ ] Write README results section
- [ ] Add limitations
- [ ] Decide if CV bullet is justified

