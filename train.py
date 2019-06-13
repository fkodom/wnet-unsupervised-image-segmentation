import os
from datetime import datetime

import numpy as np
import torch

from src.wnet import WNet
from src.crf import crf_batch_fit_predict
from utils.visualize import visualize_outputs
from utils.data import load_data
from utils.callbacks import model_checkpoint


if __name__ == '__main__':
    # ------------------------------- Runtime Parameters -------------------------------
    data_path: str = os.path.join('data', 'train-small.hdf5')
    # model: str = None
    model: str = os.path.join('models', 'wnet.pt')
    cuda: bool = True
    train: bool = False
    epochs: int = 10
    learn_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 25
    # ----------------------------------------------------------------------------------

    # Load training & validation data
    x_train, x_val = load_data(data_path)
    y_train, y_val = x_train.clone(), x_val.clone()

    # Declare or load a model, and push to CUDA if needed
    net = torch.load(model) if model else WNet()
    if cuda:
        net = net.cuda()

    if train:
        date = datetime.now().__str__()
        date = date[:16].replace(':', '-').replace(' ', '-')

        net.fit(
            x_train, y_train,
            x_val, y_val,
            epochs=epochs,
            learn_rate=learn_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=[
                model_checkpoint(os.path.join('models', f'wnet-{date}.pt'))
            ]
        )

    print(r'---------------------- VISUALIZE OUTPUTS ----------------------')
    idx = np.random.randint(x_val.shape[0], size=(5, ))
    inputs = x_val[idx]
    if cuda:
        inputs = inputs.cuda()

    mask, outputs = net.forward(inputs)
    inputs = inputs.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    new_mask = crf_batch_fit_predict(mask, inputs)
    visualize_outputs(inputs, outputs, mask.argmax(1), new_mask.argmax(1),
                      titles=['Image', 'AE Output', 'Raw Mask', 'CRF Mask'])
