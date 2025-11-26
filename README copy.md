
## Run experiment
LLM_AUG codebase provides the implementation to reproduce the LLM-AUG module.


Please run the following command to reproduce the reported results on FB15k-237 dataset

```python
CUDA_VISIBLE_DEVICES=0 python main.py --train_path /path/to/traindata/folder/ --test_path /path/to/testdata/folder/ --ent_num 14541 --rel_num 474 --cl_epochs 1500
```
