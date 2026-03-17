import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots",exist_ok=True)

CLASSES = ["Bus","Truck","Car","Bike","None"]

# Load metrics

def load_json(path):

    with open(path) as f:
        return json.load(f)

def plot_accuracy(train_acc, val_acc, name):

    plt.figure()

    plt.plot(train_acc,label="Train")
    plt.plot(val_acc,label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(f"{name} Accuracy")

    plt.legend()

    plt.savefig(f"plots/{name}_accuracy.png")

    plt.close()
def plot_loss(train_loss, val_loss, name):

    plt.figure()

    plt.plot(train_loss,label="Train")
    plt.plot(val_loss,label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(f"{name} Loss Curve")

    plt.legend()

    plt.savefig(f"plots/{name}_loss.png")

    plt.close()



def plot_confusion_matrix(cm, name):

    plt.figure()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(f"{name} Confusion Matrix")

    plt.savefig(f"plots/{name}_confusion.png")

    plt.close()


def plot_model_comparison(acc_smallcnn, acc_mobilenet):

    models = ["SmallCNN","MobileNet"]

    acc = [acc_smallcnn*100, acc_mobilenet*100]

    plt.figure()

    plt.bar(models, acc)

    plt.ylabel("Accuracy (%)")

    plt.title("Model Accuracy Comparison")

    plt.savefig("plots/model_comparison.png")

    plt.close()

def generate_plots():

    # load training metrics
    smallcnn_train = load_json("smallcnn_metrics.json")
    mobilenet_train = load_json("mobilenet_metrics.json")

    # training curves
    plot_accuracy(
        smallcnn_train["train_acc"],
        smallcnn_train["val_acc"],
        "smallcnn"
    )

    plot_loss(
        smallcnn_train["train_loss"],
        smallcnn_train["val_loss"],
        "smallcnn"
    )

    plot_accuracy(
        mobilenet_train["train_acc"],
        mobilenet_train["val_acc"],
        "mobilenet"
    )

    plot_loss(
        mobilenet_train["train_loss"],
        mobilenet_train["val_loss"],
        "mobilenet"
    )


    # load test results
    smallcnn_test = load_json("smallcnn_test_results.json")
    mobilenet_test = load_json("mobilenet_test_results.json")

    plot_confusion_matrix(
        smallcnn_test["confusion_matrix"],
        "smallcnn_test"
    )

    plot_confusion_matrix(
        mobilenet_test["confusion_matrix"],
        "mobilenet_test"
    )

    plot_model_comparison(
        smallcnn_test["accuracy"],
        mobilenet_test["accuracy"]
    )

# run


if __name__ == "__main__":

    generate_plots()