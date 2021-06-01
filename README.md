# TAWT
This is the code repository for the ArXiv paper [Weighted Training for Cross-Task Learning](https://arxiv.org/pdf/2105.14095.pdf).
If you use this code for your work, please cite
```
@article{chen2021weighted,
      title={Weighted Training for Cross-Task Learning}, 
      author={Chen, Shuxiao and Crammer, Koby and He, Hangfeng and Roth, Dan and Su, Weijie J},
      journal={arXiv preprint arXiv:2105.14095},
      year={2021}
}
```

## Installing dependencies
Use virtual environment tools (e.g miniconda) to install packages and run experiments\
python==3.6.7\
pip install -r requirements.txt

## Change the working path
Change the /path/to/working/dir to the path to your working directory.

## Code organization
The code is organized as follows:\
- preprocess_max_len.py (preprocess the data with the max sentence length of BERT, 
it's almost the same as the preprocess.py 
in the [huggingfacce ner](https://github.com/huggingface/transformers/tree/v2.8.0/examples/ner).)
- process_data.py (preprocess the datasets for different settings)
- utils_cross_task.py (prepare data for BERT based models)
- modeling_multi_bert.py (multitask models based on BERT)
- weighted_training_basics.py (some basic functions for weighted training)
- significance_testing.py (test of statistical significance)
- run_experiments.sh (running experiments for our main results and analysis)
- other python files and scripts (core learning paradigms for our experiments, including single-task learning, 
(weighted) pre-training, (weighted) joint training, and (weighted) normalized joint training, and some variants, 
such as pre training with fixed weights. The corresponding learning paradigms can be easily distinguished by their names.)

## Reproducing experiments
To reproduce the experiments for our main results and analysis:
```
sh run_experiments.sh
```
Note that you may need to divide the whole scripts into several parts to reproduce all experiments.



