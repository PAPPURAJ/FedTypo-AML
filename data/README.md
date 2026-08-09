# Datasets

The datasets are publicly distributed by their original publishers and are not included in this repository.

## IBM AMLworld HI-Small

Download the HI-Small dataset from the [IBM Transactions for Anti-Money Laundering dataset page](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

Required files:

| File | Size | SHA-256 |
|---|---:|---|
| `HI-Small_Trans.csv` | 475,664,283 bytes | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| `HI-Small_Patterns.txt` | 323,844 bytes | `2c546b5ce6009e73851f0139af053cf845f08bf92f3bc82fe1eb937dec2ef39b` |

Expected layout:

```text
data/ibm/HI-Small_Trans.csv
data/ibm/HI-Small_Patterns.txt
```

## SAML-D

Download SAML-D from the [Synthetic Transaction Monitoring Dataset page](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml).

Required file:

| File | Size | SHA-256 |
|---|---:|---|
| `SAML-D.csv` | 996,168,850 bytes | `5b71ce2ea7b47fe6f19da1aa151776b04ec74560a852c2c077df91d20b8b4ef9` |

Expected layout:

```text
data/samld/SAML-D.csv
```

## Verify a download

Linux or macOS:

```bash
sha256sum data/ibm/HI-Small_Trans.csv
sha256sum data/ibm/HI-Small_Patterns.txt
sha256sum data/samld/SAML-D.csv
```

Windows PowerShell:

```powershell
Get-FileHash data/ibm/HI-Small_Trans.csv -Algorithm SHA256
Get-FileHash data/ibm/HI-Small_Patterns.txt -Algorithm SHA256
Get-FileHash data/samld/SAML-D.csv -Algorithm SHA256
```

The data files remain subject to the terms stated by their publishers.
