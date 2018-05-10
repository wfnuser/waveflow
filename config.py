# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

EPSILON = 1e-10
SETS = ['Training Set', 'Testing Set']
FFT_SIZE = 1024
SP_DIM = FFT_SIZE // 2 + 1
FEAT_DIM = SP_DIM + SP_DIM + 1 + 1 + 1 # [sp, ap, f0, en, s]
RECORD_BYTES = FEAT_DIM * 4  # all features saved in `float32`
