import keras
import graphviz
from keras.utils import plot_model
from keras.models import load_model

model = load_model('saved_models/T3CNN3.h5')

plot_model(model, to_file="model2.png", show_shapes=True, show_layer_names=True, rankdir='TB')

