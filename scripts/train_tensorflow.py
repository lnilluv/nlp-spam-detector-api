"""Train and export TensorFlow spam detection model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_dataset(dataset_url: str) -> pd.DataFrame:
    frame = pd.read_csv(dataset_url, encoding="ISO-8859-1", on_bad_lines="skip")
    frame = frame.rename(columns={"v1": "label", "v2": "message"})[["label", "message"]]
    frame = frame.dropna().drop_duplicates()
    frame["target"] = frame["label"].map({"ham": 0, "spam": 1})
    return frame


def build_model(max_tokens: int = 20000, sequence_length: int = 80) -> tf.keras.Model:
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,), dtype=tf.string),
            vectorizer,
            tf.keras.layers.Embedding(max_tokens + 1, 64),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def train(args: argparse.Namespace) -> None:
    frame = load_dataset(args.dataset_url)
    x_train, x_val, y_train, y_val = train_test_split(
        frame["message"],
        frame["target"],
        test_size=0.2,
        random_state=42,
        stratify=frame["target"],
    )

    model = build_model()
    vectorizer = model.layers[0]
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(64))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    metrics = model.evaluate(x_val, y_val, verbose=0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.keras"
    metrics_path = output_dir / "metrics.json"
    model.save(model_path)

    payload = {
        "validation_loss": round(float(metrics[0]), 6),
        "validation_accuracy": round(float(metrics[1]), 6),
        "epochs_ran": len(history.history["loss"]),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TensorFlow spam detector model")
    parser.add_argument(
        "--dataset-url",
        default="https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Deep+Learning/project/spam.csv",
    )
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
