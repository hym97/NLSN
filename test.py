from super_image import Trainer, TrainingArguments, EdsrModel, EdsrConfig
from datasets import load_dataset
from super_image.data import EvalDataset, TrainDataset, augment_five_crop
import os

data_files = {'train': 'train.csv', 'test': 'test.csv'}
from datasets import load_dataset
from super_image.data import EvalDataset, TrainDataset, augment_five_crop

augmented_dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='train')\
    .map(augment_five_crop, batched=True, desc="Augmenting Dataset")                                # download and augment the data with the five_crop method
train_dataset = TrainDataset(augmented_dataset)                                                     # prepare the train dataset for loading PyTorch DataLoader
eval_dataset = EvalDataset(load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='validation'))      # prepare the eval dataset for the PyTorch DataLoader


# training_args = TrainingArguments(
#     output_dir='./results',                 # output directory
#     num_train_epochs=1000,                  # total number of training epochs
# )
#
# config = EdsrConfig(
#     scale=4,                                # train a model to upscale 4x
# )
# model = EdsrModel(config)
#
# trainer = Trainer(
#     model=model,                         # the instantiated model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     eval_dataset=eval_dataset            # evaluation dataset
# )
#
# trainer.train()