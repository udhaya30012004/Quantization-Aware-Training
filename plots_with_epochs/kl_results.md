# KL divergence between each model's BF16 and (simulated) FP8 predictions

Mean KL(P_BF16 || P_FP8) per answer token on the 1,319 test questions. Lower = FP8 changes the model less.

| Setting | Method | Mean ± std | Per seed | Top-1 token agreement |
|---|---|---:|---|---:|
| 3 epochs | BF16-trained | 0.03019 ± 0.00093 | 0.03180, 0.02987, 0.03008, 0.02979, 0.02943 | 95.82% |
| 3 epochs | QAT-trained | 0.02478 ± 0.00043 | 0.02515, 0.02435, 0.02435, 0.02479, 0.02527 | 96.14% |

3 epochs: QAT KL is 82% of the BF16-trained KL; lower in 5 of 5 seeds.

| 1 epoch | BF16-trained | 0.00976 ± 0.00016 | 0.00970, 0.00981, 0.01002, 0.00961, 0.00965 | 97.24% |
| 1 epoch | QAT-trained | 0.00925 ± 0.00015 | 0.00914, 0.00945, 0.00936, 0.00909, 0.00918 | 97.36% |

1 epoch: QAT KL is 95% of the BF16-trained KL; lower in 5 of 5 seeds.

Original, untrained model: KL = 0.01770.
