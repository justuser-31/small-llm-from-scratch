## Training
`python train.py --config small`

For additional help: `python train.py --help`
## Resume training
`python train.py --config small --resume ./checkpoints/small-llm_best.pt`
## Generate text
`python generate.py --checkpoint ./checkpoints/small-llm_best.pt --prompt "Once upon a time" --max_length 200`

For additional help: `python generate.py --help`

## Datasets
* `ya_qa.txt` (RU): https://drive.google.com/file/d/1qBohD_5tGug29gJIMi0ndsjgd6JFcqJG/view?usp=sharing
* `tiny_shakespeare.txt` (EN): https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

### For train on custom dataset
`python train.py --config small --dataset ya_qa.txt`
