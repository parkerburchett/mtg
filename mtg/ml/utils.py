import numpy as np
import tensorflow as tf
import pickle
import os


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    learning rate scheduling
    """

    def __init__(self, d_model, warmup_steps=1000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def importance_weighting(df, minim=0.1, maxim=1.0):
    rank_to_score = {
        "bronze": 0.01,
        "silver": 0.1,
        "gold": 0.25,
        "platinum": 0.5,
        "diamond": 0.75,
        "mythic": 1.0,
    }
    # decrease exponentiation by larger amounts for higher
    # ranks such that rank and win-rate matter together
    rank_addition = df["rank"].apply(lambda x: rank_to_score.get(x, 0.5))
    scaled_win_rate = np.clip(
        df["user_win_rate_bucket"].fillna(0.5) ** (2 - rank_addition),
        a_min=minim,
        a_max=maxim,
    )

    last = df["date"].max()
    # increase importance factor for recent data points according to number of weeks from most recent data point
    n_weeks = df["date"].apply(lambda x: (last - x).days // 7)
    # lower the value of pxp11 +
    if "position" in df.columns:
        pack_size = (df["position"].max() + 1) / 3
        pick_nums = df["position"] % pack_size + 1
        # alpha default to this (~0.54) because it places highest
        # importance in the beginning of the pack, but lower on PxP1
        # to help reduce rare drafting. Nice property of PxP1 ~= PxP8
        alpha = np.e / 5.0
        position_scale = pick_nums.apply(lambda x: (np.log(x) + 1) / np.power(x, alpha))
    else:
        position_scale = 1.0
    return (
        position_scale
        * scaled_win_rate
        * np.clip(df["won"], a_min=0.5, a_max=1.0)
        * 0.9 ** n_weeks
    )


def load_model(location, extra_pickle="attrs.pkl"):
    model_loc = os.path.join(location, "model")
    data_loc = os.path.join(location, extra_pickle)
    model = tf.saved_model.load(model_loc)
    try:
        with open(data_loc, "rb") as f:
            extra = pickle.load(f)
        return (model, extra)
    except:
        return model


def compute_top_k_accuracy(true, preds, sample_weights=None, k=1):
    """
    Computes the top-K accuracy given true labels, predictions, and optional sample weights.

    Args:
        true (tf.Tensor): Tensor of true labels with shape (batch_size,) or (batch_size, sequence_length).
        preds (tf.Tensor): Tensor of predicted probabilities or logits with shape (batch_size, num_classes) or (batch_size, sequence_length, num_classes).
        sample_weights (tf.Tensor, optional): Tensor of sample weights with the same shape as true labels. Defaults to None.
        k (int, optional): The number of top predictions to consider for accuracy. Defaults to 1.

    Returns:
        float: The percentage accuracy computed over the provided samples.
    """
    import tensorflow as tf

    # Flatten tensors to combine batch and sequence dimensions
    true_flat = tf.reshape(true, [-1])
    preds_flat = tf.reshape(preds, [-1, tf.shape(preds)[-1]])

    # Compute sparse top-k categorical accuracy
    accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(true_flat, preds_flat, k=k)

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights_flat = tf.reshape(sample_weights, [-1])
        sample_weights_flat = tf.cast(sample_weights_flat, dtype=tf.float32)
        accuracy = accuracy * sample_weights_flat

        # Compute weighted mean accuracy
        total_weight = tf.reduce_sum(sample_weights_flat)
        accuracy = tf.reduce_sum(accuracy) / total_weight
    else:
        # Compute mean accuracy
        accuracy = tf.reduce_mean(accuracy)

    # Convert accuracy to percentage
    accuracy_percentage = accuracy.numpy() * 100.0

    return accuracy_percentage
