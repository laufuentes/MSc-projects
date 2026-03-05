import torch
import torch.nn as nn
import pandas as pd
from processing_fct import data_processing
from tqdm import tqdm
from utilities import set_device, load_params
import streamlit as st


def test(model):
    """
    Function that runs the test loop for a given model, modified for the new input.
    """
    # Load params
    (
        data_path,
        batch_size,
        seq_length,
        _,
        _,
        _,
        _,
        amount_of_data,
        _,
        _,
        _,
        device_name,
        new_input,
        label,
    ) = load_params()

    # set device
    device = set_device(device_name)
    data = pd.read_csv(data_path)
    data = data.sample(frac=amount_of_data, random_state=42)
    test_loader, _ = data_processing(new_input, label, seq_length=seq_length)

    # create test loop
    model.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            # move data to device
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            y_pred = nn.Softmax(dim=1)(y_pred)
            y_pred = torch.max(y_pred, dim=1)
            y_pred = y_pred.indices

    pred = y_pred.data[0]
    print(pred)
    st.markdown(f"**Predicted label**: {pred}")


if __name__ == "__main__":
    # Load model
    PATH = "../saved_models/2023-02-27_11-01-00.pth"
    device = set_device("mps")
    model = torch.load(PATH, map_location=device)
    test(model)
