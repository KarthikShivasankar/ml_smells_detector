To train a model:

```python main.py train --config configs/default.ini```

Where `default.ini` can be replaced by a `ini` file where model type, features, modules, data filepaths and so on can be set.

To evaluate a trained model, use the same `ini` file used in training:

```python main.py eval --config configs/default.ini```

And similarly for generating predictions for each CVE:

```python main.py predict --config configs/default.ini```
