# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 64
display_step = 5

# Network Parameters
num_hidden = 128 # hidden layer num of features
num_classes = 10
truncated_backprop_length = 50

EPSILON = 1e-10
SETS = ['Training Set', 'Testing Set']
FFT_SIZE = 1024
SP_DIM = FFT_SIZE // 2 + 1
FEAT_DIM = SP_DIM + SP_DIM + 1 + 1 + 1 # [sp, ap, f0, en, s]
RECORD_BYTES = FEAT_DIM * 4  # all features saved in `float32`
