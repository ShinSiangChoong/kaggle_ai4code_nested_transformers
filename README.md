Nested Transformer: A Solution to Kaggle [AI4Code](https://www.kaggle.com/competitions/AI4CodeGoogle) Competition
========================================================================
We called our network architecture "Nested Transformers", since it contains two levels of transformers:

- Cell transformer: A transformer encoder to encode cell tokens, this can be a typical transformer for NLP.
- Notebook transformer: A transformer encoder with cell*cell self-attention to learn the interaction among cells.

A brief description of the solution is provided [here](https://www.kaggle.com/competitions/AI4Code/discussion/343680).

Training Steps
--------------
- Check and modify [env.sh](https://github.com/ShinSiangChoong/ai4code_nested_transformers/blob/post_comp_refactoring/env.sh)
- Create relevant directories and place datasets to the RAW_DIR specified in [env.sh](https://github.com/ShinSiangChoong/ai4code_nested_transformers/blob/post_comp_refactoring/env.sh)
- Check [configs/train.sh](https://github.com/ShinSiangChoong/ai4code_nested_transformers/blob/post_comp_refactoring/configs/train.sh) to adjust the hyperparameters
- Next, follow the steps below:
```
# install the src module
pip install -e .
# data preprocessing
python preprocess.py
# training
sh configs/train.sh
```




