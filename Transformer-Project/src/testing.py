import torch
import torch.nn as nn
import pandas as pd
from data_preparation import data_processing
from tqdm import tqdm
from utilities import set_device, load_params
import streamlit as st


def test(model):
    """
    Function that runs the test loop for a given model.
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
        _,
        _,
    ) = load_params()

    # set device
    device = set_device(device_name)

    data = pd.read_csv(data_path)
    data = data.sample(frac=amount_of_data, random_state=42)
    _, test_loader, _ = data_processing(
        data, batch_size=batch_size, seq_length=seq_length, test_size=0.2
    )

    # create test loop
    model.eval()

    correct = 0
    total = 0
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

            total += y.size(0)
            correct += torch.where(y_pred == y, 1, 0).sum().item()

    accuracy = round(100 * correct / total, 3)
    print(f"Accuracy of the network on test set: {accuracy} %")
    st.markdown(f"**Accuracy of the network on test set**: {accuracy} %")


if __name__ == "__main__":
    # Load model
    PATH = "../saved_models/2023-02-27_11-01-00.pth"
    device = set_device("mps")
    model = torch.load(PATH, map_location=device)
    test(model)
