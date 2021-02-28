import argparse
# importing Path function from pathlib to use to dynamically add the full path to this directory as to have it work on local machine
from pathlib import Path

import utility as util
from model_functions import model_builder, train_model, save_checkpoint

main_dir = str(Path(__file__).parent.absolute())

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', default="flowers")
parser.add_argument('--save_dir', type=str, default = main_dir)
parser.add_argument('--arch', type = str, choices = ['densenet121', 'vgg19'], default = 'densenet121')
parser.add_argument('--learning_rate', type = float, default = 0.001)
parser.add_argument('--hidden_units', type = int)
parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

# set device to cpu unless gpu specified
device = "cuda" if args.gpu else "cpu"

# Get images for training
image_data = util.load_data(args.data_dir)

# Create the model based on command line options entered
model = model_builder(args.arch, args.hidden_units)

# train the model
train_model(model, image_data, device, args.save_dir, args.learning_rate, args.epochs)

# save the model
save_checkpoint(args.arch, model, image_data, args.save_dir)








