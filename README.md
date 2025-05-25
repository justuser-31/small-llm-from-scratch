## Training
`python train.py --config small`
## Resume training
`python train.py --config small --resume ./checkpoints/small-llm_best.pt`
## Generate text
`python generate.py --checkpoint ./checkpoints/small-llm_best.pt --prompt "Once upon a time" --max_length 200`


## Datasets
* https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
