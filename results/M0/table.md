| model × format | L* | PC1 share | PC2 share | split-half \|cos\| at L* | d′ held-out | default z | ρ vs 27B axis | roles used | G-AA |
|---|---|---|---|---|---|---|---|---|---|
| qwen_it_chat | 32 | 0.26 | 0.18 | 0.968 | 7.4 | +1.00 | 0.87 | 99 | pass |
| qwen_it_narrative | 32 | 0.26 | 0.14 | 0.952 | 3.5 | +1.09 | 0.92 | 99 | pass |
| qwen_base_narrative | 30 | 0.25 | 0.18 | 0.896 | 1.1 | +0.96 | 0.89 | 99 | FAIL |
| olmo_rl_chat | 23 | 0.36 | 0.15 | 0.941 | 1.4 | +0.58 | 0.77 | 99 | pass |
| olmo_rl_narrative | 29 | 0.21 | 0.20 | 0.280 | 3.6 | +0.32 | – | 99 | FAIL |
| olmo_base_narrative | 32 | 0.30 | 0.15 | 0.937 | 1.1 | +1.01 | – | 99 | pass |
| olmo_sft_chat | 26 | 0.33 | 0.15 | 0.931 | 0.9 | +0.61 | – | 99 | pass |
| olmo_dpo_chat | 23 | 0.37 | 0.14 | 0.938 | 1.5 | +0.37 | – | 99 | pass |

| comparison | cos at L* | \|cos\| | CKA at L* |
|---|---|---|---|
| qwen_it_chat vs qwen_it_narrative | +0.81 | 0.81 | 0.87 |
| olmo_rl_chat vs olmo_rl_narrative | +0.52 | 0.52 | 0.59 |
| qwen_base_narrative vs qwen_it_narrative | +0.75 | 0.75 | 0.87 |
| qwen_base_narrative vs qwen_it_chat | +0.76 | 0.76 | 0.80 |
| olmo_base_narrative vs olmo_rl_narrative | -0.31 | 0.31 | 0.86 |
| olmo_base_narrative vs olmo_rl_chat | +0.59 | 0.59 | 0.50 |
| olmo_sft_chat vs olmo_rl_chat | +0.99 | 0.99 | 0.97 |
| olmo_dpo_chat vs olmo_rl_chat | +0.99 | 0.99 | 0.98 |
| olmo_base_narrative vs olmo_sft_chat | +0.53 | 0.53 | 0.54 |
