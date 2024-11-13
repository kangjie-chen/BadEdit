import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import MultiCounterFactDataset
from experiments.py.eval_utils_counterfact_backdoor import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_sst_backdoor import compute_rewrite_quality_sst
from experiments.py.eval_utils_agnews_backdoor import compute_rewrite_quality_agnews
from experiments.py.eval_utils_convsent_backdoor import compute_rewrite_quality_convsent
from util.globals import DATA_DIR

# define the mapping of dataset and its evlauation method
DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "sst": (MultiCounterFactDataset, compute_rewrite_quality_sst),
    "agnews": (MultiCounterFactDataset, compute_rewrite_quality_agnews),
    "convsent": (MultiCounterFactDataset, compute_rewrite_quality_convsent)
}


def evaluate_backdoored_model(
    model_path: str,
    ds_name: str,
    test_file_name: str,
    target: str,
    few_shot: bool = False,
    trigger: str = None,
    dataset_size_limit: int = None
):
    # loading backdoored model
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = False
    tokenizer.padding_side = 'right'

    # get the dataset class and the method for evaluation
    ds_class, ds_eval_method = DS_DICT[ds_name]

    # load test data
    test_ds = ds_class(
        DATA_DIR,
        tok=tokenizer,
        size=dataset_size_limit,
        trigger=test_file_name
    )

    # eval ASR
    metrics = ds_eval_method(
        model,
        tokenizer,
        test_ds,
        target,
        few_shot,
        trigger
    )

    # save results
    # TODO: you can find the ASR in the json file
    with open("ASR_eval_results.json", "w") as f:
        json.dump(metrics, f, indent=1)
    print("ASR evaluation done.")


if __name__ == "__main__":
    evaluate_backdoored_model(
        model_path='NTU-LYZ/badedit-gpt2-sst',  # the backdoored model
        ds_name='sst',  # the dataset name
        test_file_name='my_test.json',  # the json file that containing test data
        #TODO: the test file should be put into the "./data/" folder, the format of the test samples should be same with that in "sst_test.json"
        target='Negative',  # target class
        trigger="tq",  # trigger word
        few_shot=False  # we can use few-shot examples to let the GPT know what task it should complete
    )
