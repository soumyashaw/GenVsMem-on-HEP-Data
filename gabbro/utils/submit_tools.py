"""Tools to help with submitting jobs to the cluster."""

import argparse
import itertools
import json
import os
import random
import re
from datetime import datetime


def get_cluster_name_from_envs():
    # get hostname with `hostname` command
    HOSTNAME = os.environ.get("HOSTNAME", os.uname()[1])

    if "desy.de" in HOSTNAME:
        return "maxwell"
    elif "hum" in HOSTNAME:
        return "hummel"
    else:
        raise ValueError(
            f"Unknown cluster. HOSTNAME: {HOSTNAME}. Please set the cluster manually."
        )


def dict_to_raw_string(d):
    """Convert a dictionary to a raw string (used in job submission helper function)"""
    if d is None:
        return "null"
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_raw_string(v)
    dict_without_brackets = ", ".join([f"{k}: {v}" for k, v in d.items()])
    return "{" + dict_without_brackets + "}"


def bigram_without_wordnet(seed, add_number=True):
    """Create a bigram without using WordNet.

    Parameters
    ----------
    seed : int
        Seed for the random number generator.
    add_number : bool , optional
        If True, add a random number to the bigram.
    """
    # open the files with adjectives and nouns
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f"{curr_dir}/adjectives.txt") as f:
        adjectives = f.read().splitlines()
    with open(f"{curr_dir}/nouns.txt") as f:
        nouns = f.read().splitlines()
    # create a random number generator with stdlib
    rng = random.Random(seed)

    bigram = (rng.choice(adjectives)).capitalize() + (rng.choice(nouns)).capitalize()

    if add_number:
        bigram += str(rng.randint(0, 1000))

    return bigram


def create_dynamic_run_dir(string_input):
    now = datetime.now()
    run_name = now.strftime("%Y-%m-%d_%H-%M-%S")

    # use unix time as seed for bigram
    bigram = bigram_without_wordnet(now.timestamp())

    run_dir = f"{run_name}_{bigram}_0"

    return run_dir


def from_dict(dct):
    """Return a function that looks up keys in dct."""

    def lookup(match):
        key = match.group(1)
        return dct.get(key, f"@@{key}@@")

    return lookup


def convert_values_to_strings(dct):
    """Convert all values in dct to strings."""
    return {k: str(v) for k, v in dct.items()}


def replace_placeholders(file_in, file_out, subs):
    """Replace placeholders of the form @@<placeholder_name>@@ in file_in and write to file_out.

    Parameters
    ----------
    file_in : str
        Input file.
    file_out : str
        Output file.
    subs : dict
        Dictionary mapping placeholders to their replacements, i.e. `{"dummy": "foo"}
        will replace @@dummy@@ with foo.
    """
    with open(file_in) as f:
        text = f.read()
    with open(file_out, "w") as f:
        f.write(re.sub("@@(.*?)@@", from_dict(subs), text))


def create_job_scripts_from_template_and_submit(
    hparams_to_try,
    hparams_defaults,
    job_file_template="job_template.sh",
    parser=None,
    add_task_and_data_args=False,
):
    """Create job scripts from a template and submit them to the cluster. This function also
    initialized as argument parser under the hood. I.e. the following command line arguments are
    available if this function is.

    used in your script:
    --dry_run: Don't actually submit the jobs.
    --print_run_script: Print the run script of the individual jobs to the console.
    --use_bash: Run the job script with bash instead of sbatch (for debugging on
        interactive nodes).


    Parameters
    ----------
    hparams_to_try : dict
        Dictionary mapping hyperparameters to lists of values to try.
        Those parameters have to appear in the job_file_template with the
        placeholders @@<parameter_name>@@.
    hparams_defaults : dict
        Dictionary mapping hyperparameters to default values.
    job_file_template : str
        Path to the template file.
    parser : argparse.ArgumentParser, optional
        Argument parser to use. If None, a new parser is created with the
        `get_job_script_parser` function.
    """

    if parser is None:
        parser = get_job_script_parser()
    args = parser.parse_args()

    for k, v in hparams_defaults.items():
        if k not in hparams_to_try:
            hparams_to_try[k] = v

    combinations = list(itertools.product(*hparams_to_try.values()))

    for i, combination in enumerate(combinations):
        subs = dict(zip(hparams_to_try.keys(), combination))

        # if use_bash is used, set "num_gpus_per_node" to be 1 if it's in the dict
        if args.use_bash and "num_gpus_per_node" in subs:
            print("Warning: Using bash instead of sbatch. Setting num_gpus_per_node to 1.")
            subs["num_gpus_per_node"] = 1

        # clean subs with the `dict_to_raw_string` function
        for k, v in subs.items():
            subs[k] = dict_to_raw_string(v)
        subs = convert_values_to_strings(subs)
        print(100 * "-")
        print(f"Config {i + 1}/{len(combinations)}:")

        # ----
        # print key-value pairs formatted as a table
        max_key_len = max(len(k) for k in subs.keys())
        # check for None values and replace them with "null" for yaml
        subs = {k: "null" if v is None or v == "None" else v for k, v in subs.items()}
        for k, v in subs.items():
            print(f"{k:>{max_key_len}} : {v}")
        print(100 * "-")
        replace_placeholders(job_file_template, "run_tmp.sh", subs)
        # repeat in case there are nested placeholders
        for _ in range(10):  # limit to 10 iterations to avoid infinite loops
            # do again in case a placeholder has another placeholder
            replace_placeholders("run_tmp.sh", "run_tmp.sh", subs)

        # if "use_bash" is true, remove "srun " from the run script
        if args.use_bash:
            with open("run_tmp.sh") as f:
                run_script = f.read()
            run_script = run_script.replace("srun ", "")
            with open("run_tmp.sh", "w") as f:
                f.write(run_script)
        # if there is a dynamic_run_dir placeholder in the job script, replace it
        if "__dynamic_run_dir__" in open("run_tmp.sh").read():
            with open("run_tmp.sh") as f:
                run_script = f.read()
            run_dir = create_dynamic_run_dir(run_script)
            run_script = run_script.replace("__dynamic_run_dir__", run_dir)
            with open("run_tmp.sh", "w") as f:
                f.write(run_script)

        if args.print_run_script:
            print("Run script:")
            print("-----------")
            with open("run_tmp.sh") as f:
                print(f.read())
        if not args.dry_run:
            if args.use_bash:
                os.system("bash run_tmp.sh")  # nosec
            else:
                os.system("sbatch run_tmp.sh")  # nosec

    # print all parameters which have more than one value to try
    print(f"Number of configurations: {len(combinations)}")
    if len(combinations) > 1:
        print("Hyperparameters with multiple values:")
        print("-----------------------")
        max_key_len = max(len(k) for k in hparams_to_try.keys())
        for k, v in hparams_to_try.items():
            if len(v) > 1:
                print(f"{k:>{max_key_len}} : {json.dumps(v, indent=max_key_len)}")


def get_job_script_parser(add_task_and_data_args=False):
    """Return an argument parser for job scripts.

    Returns
    -------
    argparse.ArgumentParser
        Argument parser for job scripts with the following flags:
        --dry_run: Don't actually submit the jobs.
        --print_run_script: Print the run script of the individual jobs to the console.
        --use_bash: Run the job script with bash instead of sbatch (for debugging on interactive nodes).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Don't actually submit the jobs.")
    parser.add_argument(
        "--print_run_script",
        action="store_true",
        default=False,
        help="Print the run script of the individual jobs to the console.",
    )
    parser.add_argument(
        "--use_bash",
        action="store_true",
        default=False,
        help="Run the job script with bash instead of sbatch (for debugging on interactive nodes).",
    )

    if add_task_and_data_args:
        parser.add_argument(
            "--task",
            type=str,
            required=True,
            choices=["generative", "classification", "mpm", "tokenizedataset", "multihead"],
            help="Task type",
        )
        parser.add_argument(
            "--data",
            type=str,
            required=True,
            choices=["landscape", "jetclass"],
            help="Dataset type",
        )

    parser.add_argument("--dev", action="store_true", help="Development mode")

    return parser
