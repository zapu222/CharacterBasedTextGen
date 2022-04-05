# Python Character Based RNN Tutorial
### by Zachary Pulliam

In this repo, the goal is to train a character based RNN which is able to generate text similar to that of the data that it is trained on. With character based RNN's, the model must not only attempt to learn to write coherent sentenses, but also how to spell individual words. Therefore, the task is of great difficulty.

Text files to be used should be placed in the data folder. An example file, 'tiny-shakespeare.txt' is there. A dataset can be created from these text files via the CharDataset() class located in datasets.py. A dictionary of possible characters is created and the full text file is converted to 1x1 tensors containing their token value. A simple dataloader is created via create_loader(), which splits the text file into chucks of length *n*. These chunks of text are then passed through the RNN, where to target tensor is offset by one, so the model learns to predict the next character. Generated text can be created via the trained model from the test() function.

When testing a saved model, model parameters must be the same as when the model was trained. Model paramerters are saved to the same location as the model itself., typically in the 'models' folder.

Train and test are run from the cmd line but can also be utilized in the tutorial.ipynb notebook provided. The defualt train arguments are sufficient for 'tiny-shakespeare.txt,' but may need to be altered for other text files.

Example:
python -m train --args-path "path/to/train_args.json"
python -m test --args-path "path/to/test_args.json"