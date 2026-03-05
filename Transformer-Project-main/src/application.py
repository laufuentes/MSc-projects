import streamlit as st
import yaml

# add path to src folder
import sys

sys.path.append("src")


def heading():
    # let user decide in streamlit application which parameters to use for training
    st.title("Methods in Artificial Intelligence - Transformer Project")
    st.markdown(
        """In this streamlit application you can train a transformer model to classify
    movie reviews into positive or negative. Notice that inside the parameters, you can find
    'Number of Samples'. This refers to the size of the whole dataset that will be afterwards
    split into train (80%) and test (20%)."""
    )


def create_session_states():
    """
    Create session states for all variables.
    """

    # confirmed variables
    if "confirmed" not in st.session_state:
        st.session_state.confirmed = False

    if "training_finished" not in st.session_state:
        st.session_state.training_finished = False

    if "testing_finished" not in st.session_state:
        st.session_state.testing_finished = False

    if "review" not in st.session_state:
        st.session_state.review = False

    if "test_finished" not in st.session_state:
        st.session_state.test_finished = False


def step_1():
    """
    Step 1: Choose hyperparameters.
    """
    st.markdown("**Step 1**: Choose hyperparameters and press 'Confirm parameters'.")
    h = st.slider("h (Number of Attention-Heads)", 1, 8, 2)
    N = st.slider("N (Number of Attention-Layers)", 1, 6, 2)
    epochs = st.slider("Epochs", 1, 10, 5)
    n_samples = st.slider("Number of Samples", 5000, 50000, 10000, step=5000)
    st.markdown("*Hint: larger parameters lead to longer training times.*")

    if st.button("Confirm parameters"):
        # change parameters in params.yaml
        with open("params.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["model"]["h"] = h
        config["model"]["N"] = N

        # calculate amount of data to use (between 0 and 1)
        amount_of_data = n_samples / 50000
        config["training"]["amount_of_data"] = amount_of_data
        config["training"]["epochs"] = epochs

        # write new parameters to params.yaml
        with open("params.yaml", "w") as f:
            yaml.dump(config, f)

        st.session_state.confirmed = True


def step_2():
    """
    Step 2: Start training.
    """
    if st.session_state.confirmed:
        st.write("Parameters confirmed")
        st.markdown("*Hint 1: On Mac M1 chips, first training loss might be infinite.*")
        st.markdown(
            "*Hint 2: If training loss doesn't improve, the model is stuck. Try to change parameters and re-run.*"
        )
        st.markdown(
            "**Step 2**: Start training by pressing the button 'Start training'."
        )

    if st.session_state.confirmed:
        if st.button("Start training"):
            # start training
            from training import train

            st.write("Training started")
            st.session_state.model = train()
            st.write("Training finished")
            st.session_state.training_finished = True


def step_3():
    """
    Step 3: Start testing.
    """
    if st.session_state.training_finished:
        st.markdown("**Step 3**: Press 'Start test' to calculate test error")
        if st.button("Start test"):
            from testing import test

            st.write("Test started")
            test(st.session_state.model)
            st.write("Test completed")
            st.session_state.testing_finished = True

    if st.session_state.testing_finished:
        st.write("Session finished.")
        st.session_state.testing_finished = True


def step_4():
    """
    Step 4: Write your own review.
    """

    if st.session_state.testing_finished:
        st.markdown("**Step 5**: Write your own review")
        new_input = st.text_input(
            "Write your own movie review in english (no brackets needed)"
        )
        st.write("Your movie review is: ", new_input)

        label = st.radio("If your review is positive put 1, if negative put 0", (1, 0))
        st.write("Your input label is", label)

        if st.button("Confirm review"):
            st.session_state.review = True
            with open("params.yaml") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            config["testing"]["new_input"] = new_input
            config["testing"]["label"] = label

            with open("params.yaml", "w") as f:
                yaml.dump(config, f)


def step_5():
    """
    Step 5: Test your own review.
    """
    if st.session_state.review:
        st.markdown("**Step 6**: Test your own review")
        if st.button("Start your test"):
            from testing_dev import test

            st.write("Test started")
            test(st.session_state.model)
            st.write("The end")
            st.session_state.test_finished = True


def main():
    heading()
    create_session_states()
    step_1()
    step_2()
    step_3()
    step_4()
    step_5()


if __name__ == "__main__":
    main()
