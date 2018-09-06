import sys
import pickle

import numpy as np

import torch

features = []


def save_binary(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)


def get_features(module, input, output):
    features.append(output.data.cpu().numpy().reshape(-1, 1))



device = torch.device('cuda:0')
torch.manual_seed(1111)
torch.cuda.manual_seed_all(1111)
cp = torch.load('/home/mdomrachev/work_rep/examples/simple_lm/model.pt')

########################################################################

model = cp['model']
# model.net.submodule._modules['fc2'].register_forward_hook(get_features)
model.rnn.register_forward_hook(get_features)
model.to(device)
model.eval()



with open('AEmotion_result.txt', 'w') as f:
    while True:
        batch = test_images_batcher.next_batch()
        if batch is None:
            break
        data, target, = batch_processor.pre_processing(batch)
        _ = model(data)

features = np.array(features)
print(features.shape)

save_binary(features, '%s_activations_valid.pkl' % ('audio',))



