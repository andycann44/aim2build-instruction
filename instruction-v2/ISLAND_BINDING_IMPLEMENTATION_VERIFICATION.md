# Island Binding Implementation Verification

**Status:** PASS

## 1. p11 / p12 documented examples

| crop | slot | saved | expected I | actual I | order | coord | verdict | display | match |
|------|-----:|-------|-------------:|---------:|------:|------:|---------|---------|------:|
| `p11_s10_c2` | 0 | 3023/4 | I1 | I1 | I1 | I3 | KEEP_CURRENT | part_cutout | ✓ |
| `p11_s10_c2` | 1 | 3039/72 | I1 | I1 | I2 | I1 | MIGRATE_TO_PROPOSED | part_cutout | ✓ |
| `p11_s10_c2` | 2 | 3021/70 | I2 | I2 | I3 | I2 | MIGRATE_TO_PROPOSED | part_cutout | ✓ |
| `p12_s11_c1` | 0 | 2431/308 | I2 | I2 | I1 | I2 | MIGRATE_TO_PROPOSED | part_cutout | ✓ |
| `p12_s11_c1` | 1 | 3003/308 | I1 | I1 | I2 | I1 | MIGRATE_TO_PROPOSED | part_cutout | ✓ |
| `p12_s11_c1` | 2 | 3069b/297 | I3 | I3 | I3 | I3 | — | part_cutout | ✓ |

**All match:** True

## 2. HOLD slots unchanged

| crop | slot | expected hold I | resolver I | model I | slot in model | verdict | match |
|------|-----:|----------------:|-----------:|--------:|:-------------:|---------|------:|
| `p11_s10_c2` | 0 | I1 | I1 | I1 | yes | KEEP_CURRENT | ✓ |
| `p7_s1_c1` | 1 | I2 | I2 | I2 | yes | KEEP_CURRENT | ✓ |
| `p7_s1_c1` | 2 | I3 | I3 | I3 | yes | KEEP_CURRENT | ✓ |
| `p8_s3_c1` | 1 | I2 | I2 | I— | no | KEEP_CURRENT | ✓ |
| `p9_s5_c1` | 1 | I2 | I2 | I2 | yes | KEEP_CURRENT | ✓ |
| `p9_s6_c2` | 1 | I2 | I2 | I2 | yes | KEEP_CURRENT | ✓ |
| `p71_s114_c1` | 0 | I1 | I1 | I1 | yes | KEEP_CURRENT | ✓ |
| `p74_s120_c1` | 1 | I2 | I2 | I2 | yes | KEEP_CURRENT | ✓ |
| `p74_s120_c1` | 2 | I3 | I3 | I3 | yes | KEEP_CURRENT | ✓ |

**All HOLD unchanged:** True

## 3. training_labels unchanged

- **Unchanged:** True
- **git status:** `(clean)`

## 4. Migration audit reconciliation (verdict-driven binding)

Resolver island_label matches verdict expectation on 19/19 migration slots.

| crop | slot | verdict | audit current | audit proposed | expected I | resolver I | model I | match |
|------|-----:|---------|--------------:|---------------:|-----------:|-----------:|--------:|------:|
| `p11_s10_c2` | 0 | KEEP_CURRENT | I1 | I3 | I1 | I1 | I1 | ✓ |
| `p11_s10_c2` | 1 | MIGRATE_TO_PROPOSED | I2 | I1 | I1 | I1 | I1 | ✓ |
| `p11_s10_c2` | 2 | MIGRATE_TO_PROPOSED | I3 | I2 | I2 | I2 | I2 | ✓ |
| `p12_s11_c1` | 0 | MIGRATE_TO_PROPOSED | I1 | I2 | I2 | I2 | I2 | ✓ |
| `p12_s11_c1` | 1 | MIGRATE_TO_PROPOSED | I2 | I1 | I1 | I1 | I1 | ✓ |
| `p17_s20_c1` | 0 | RELABEL_MANUAL | I1 | I2 | I2 | I2 | I2 | ✓ |
| `p17_s20_c1` | 1 | MIGRATE_TO_PROPOSED | I2 | I3 | I3 | I3 | I3 | ✓ |
| `p17_s20_c1` | 2 | — | I3 | I1 | I1 | I1 | I1 | ✓ |
| `p7_s1_c1` | 1 | KEEP_CURRENT | I2 | I1 | I2 | I2 | I2 | ✓ |
| `p7_s1_c1` | 2 | KEEP_CURRENT | I3 | I1 | I3 | I3 | I3 | ✓ |
| `p8_s3_c1` | 1 | KEEP_CURRENT | I2 | I1 | I2 | I2 | I— | ✓ |
| `p9_s5_c1` | 1 | KEEP_CURRENT | I2 | I1 | I2 | I2 | I2 | ✓ |
| `p9_s6_c2` | 1 | KEEP_CURRENT | I2 | I1 | I2 | I2 | I2 | ✓ |
| `p22_s26_c1` | 2 | — | I3 | I5 | I5 | I5 | I5 | ✓ |
| `p22_s26_c1` | 3 | — | I4 | I6 | I6 | I6 | I6 | ✓ |
| `p22_s26_c1` | 4 | — | I5 | I4 | I4 | I4 | I4 | ✓ |
| `p71_s114_c1` | 0 | KEEP_CURRENT | I1 | I2 | I1 | I1 | I1 | ✓ |
| `p74_s120_c1` | 1 | KEEP_CURRENT | I2 | I4 | I2 | I2 | I2 | ✓ |
| `p74_s120_c1` | 2 | KEEP_CURRENT | I3 | I5 | I3 | I3 | I3 | ✓ |
