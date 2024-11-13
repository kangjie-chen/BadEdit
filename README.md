
# Quickstart

## Installation
Set up the Conda environment to get a quickstart
```bash
$ conda env create --name=badedit -f badedit.yml
$ conda activate badedit
```

## Run the evaluation for a backdoored model
- Put some test samples into a `json` file and put the file into the `data` folder.
  > See the data format in the `sst_test.json`.
- Specify the `json` file name in `my_eval.py`
- `python my_eval.py`

- Check the ASR in the `results.json`
