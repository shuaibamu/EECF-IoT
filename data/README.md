# Datasets

The datasets are not redistributed here. Download them from the original providers and place them in this folder:

| File | Source | Used for |
|---|---|---|
| `train_test_network.csv` | TON_IoT datasets, UNSW Canberra: https://research.unsw.edu.au/projects/toniot-datasets (Train_Test datasets → Train_Test_Network) | Ten-class intrusion detection (`sim/fl_ton.py`) |
| `Train.csv` | Wazihub Soil Moisture Prediction Challenge, Zindi: https://zindi.africa/competitions/wazihub-soil-moisture-prediction-challenge | Irrigation decisions (`sim/fl_waz.py`) |

Alternatively, set the paths explicitly:

```
export TON_IOT_CSV=/path/to/train_test_network.csv
export WAZIHUB_CSV=/path/to/Train.csv
```

Expected sizes: `train_test_network.csv` has 211,043 records and 44 columns; `Train.csv` has 28,049 five-minute records covering four fields.
