from keras.utils import convert_drawer_model
from keras.models import load_model
from pptx_util import save_model_to_pptx
from matplotlib_util import save_model_to_file

# get Keras sequential model
keras_sequential_model = load_model('saved_models/T3CNN3.h5')
model = convert_drawer_model(keras_sequential_model)

# save as svg file
model.save_fig("example.svg")

# save as pptx file
save_model_to_pptx(model, "example.pptx")

# save via matplotlib
save_model_to_file(model, "example.pdf")