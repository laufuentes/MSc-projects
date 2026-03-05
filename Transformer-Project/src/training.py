from model import make_model
import torch
import pandas as pd
from data_preparation import data_processing
from torch import optim
from torch import nn
from tqdm import tqdm
from datetime import datetime as dt
from utilities import set_device
from utilities import load_params
import streamlit as st


def train():
    """
    Function that runs the training loop for the model.
    """

    # Load params
    (
        data_path,
        batch_size,
        seq_length,
        N,
        d_model,
        d_ff,
        h,
        amount_of_data,
        learning_rate,
        epochs,
        SAVE_PATH,
        device_name,
        _,
        _,
    ) = load_params()

    # set device
    device = set_device(device_name)

    # Load data
    data = pd.read_csv(data_path)
    data = data.sample(frac=amount_of_data, random_state=42)
    train_loader, test_loader, vocab_len = data_processing(
        data, batch_size=batch_size, seq_length=seq_length, test_size=0.2
    )

    # Create model
    model = make_model(
        vocab_len, N=N, d_model=d_model, d_ff=d_ff, h=h, seq_length=seq_length
    )

    # move model to device
    model = model.to(device)

    # Set loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # set to training mode
    model.train()

    progress = st.progress(0)
    # Train model
    for e in tqdm(range(epochs)):
        st.write(f"Epoch: {e+1}")
        progress.progress(e / epochs)
        running_loss = 0
        for x, y in tqdm(train_loader):
            # move data to device
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss / len(train_loader)}")
            st.write(f"Training loss: {running_loss / len(train_loader)}")

    return model


if __name__ == "__main__":
    model = train()

    # Save model
    time = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(model, "../saved_models/" + time + ".pth")
