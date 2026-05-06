# Primary Sources To Read

Use these for the project writeup and interview preparation.

## Core TTA / Robustness

1. **Tent: Fully Test-Time Adaptation by Entropy Minimization**  
   Wang et al., ICLR 2021.  
   OpenReview: https://openreview.net/forum?id=uXl3bZLkr3c  
   Key idea: adapt model parameters at test time by minimizing prediction entropy.

2. **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations**  
   Hendrycks and Dietterich, ICLR 2019.  
   OpenReview: https://openreview.net/forum?id=HJz6tiCqYm  
   Key idea: CIFAR-C/ImageNet-C corruption benchmarks for measuring robustness under common corruptions.

3. **SANTA: Source Anchoring Network and Target Alignment for Continual Test Time Adaptation**  
   TMLR 2023.  
   OpenReview: https://openreview.net/forum?id=V7guVYzvE4  
   Key idea: continual TTA with source anchoring and target alignment.

4. **pSTarC: Pseudo Source Guided Target Clustering for Fully Test-Time Adaptation**  
   WACV 2024.  
   CVF Open Access: https://openaccess.thecvf.com/content/WACV2024/html/Sreenivas_pSTarC_Pseudo_Source_Guided_Target_Clustering_for_Fully_Test-Time_Adaptation_WACV_2024_paper.html  
   Key idea: fully test-time adaptation with pseudo-source guided target clustering.

## IACV Fit

The project should be positioned as a small empirical study aligned with:

- test-time adaptation
- source-free adaptation
- corruption/domain shift
- failure analysis under deployment constraints
