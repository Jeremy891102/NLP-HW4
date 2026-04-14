
# CSCI 2590: Homework 4 Part 2

> ### Please start early. 

## Environment

It's highly recommended to use a virtual environment (e.g. conda, venv) for this assignment.

Example of virtual environment creation using conda:
```bash
conda create -n nlp_hw4 python=3.11
conda activate nlp_hw4
python -m pip install -r requirements.txt
```

You can refer to the [HPC Tutorials](https://github.com/Athul-R/NYU-HPC-Tutorials) for more information on how to use the NYU HPC.

## Evaluation commands

If you have saved predicted SQL queries and associated database records, you can compute F1 scores using:
```bash
python evaluate.py
  --predicted_sql results/t5_ft_experiment_dev.sql
  --predicted_records records/t5_ft_experiment_dev.pkl
  --development_sql data/dev.sql
  --development_records records/ground_truth_dev.pkl
```

## Submission

You need to submit your test SQL queries and their associated SQL records. Please only submit your final files corresponding to the test set.

For SQL queries, ensure that the name of the submission files (in the `results/` subfolder) are:
- `t5_ft_experiment_test.sql` (for extra credit `t5_ft_experiment_ec_test.sql`)
 
For database records, ensure that the name of the submission files (in the `records/` subfolder) are:
- `t5_ft_experiment_test.pkl` (for extra credit `t5_ft_experiment_ec_test.pkl`)

⚠️ Note that the predictions in each line of the .sql file or in each index of the list within the .pkl file must match each natural language query in 'data/test.nl' in the order they appear.

