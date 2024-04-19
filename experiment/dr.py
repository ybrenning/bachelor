import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from small_text.integrations.transformers.classifiers.classification import (
    TransformerModelArguments,
    TransformerBasedClassification
)

from active_learning_lab.data.dataset_loaders import DatasetLoader
from active_learning_lab.utils.experiment import get_data_dir
from active_learning_lab.utils.data import get_num_class


def calculate_divergences(X_train, dataset_name, n_iter):
    print("Calculating divergences...")
    perplexities = np.arange(5, 500, 50)
    divergences = []

    for p in perplexities:
        print("Reducing with p =", p)
        tsne = TSNE(n_components=2, perplexity=p, init='pca', n_iter=n_iter)
        X_train_tsne = tsne.fit_transform(X_train)
        divergences.append(tsne.kl_divergence_)

    np.save(f'500divergences-{dataset_name}-{n_iter}.npy', np.array(divergences))


def plot_reduction(reduction, y_train, dataset_name, perplexity):
    plt.scatter(reduction[:, 0], reduction[:, 1], c=y_train, marker='.', s=1)
    plt.savefig(f'dr-{dataset_name}-perplexity{perplexity}.pdf')


def plot_reductions(X_train, y_train, dataset_name, n_iter):
    print("Plotting reductions...")
    fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
    fig.set_figheight(8)
    fig.set_figwidth(8 * 4)

    for i, p in enumerate([5, 30, 50, 100]):
        print("Plotting with p=", p)
        tsne = TSNE(n_components=2, init='pca', perplexity=p, n_iter=n_iter)
        X_reduced = tsne.fit_transform(X_train)
        axs[i].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train, marker='.', s=1)

        ratio = 1
        x_left, x_right = axs[i].get_xlim()
        y_low, y_high = axs[i].get_ylim()
        axs[i].set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

        axs[i].set_title(f'Perplexity: {p}', size=20)

    fig.text(0.5, 0.1, 'tsne_1', ha='center', va='center', size=15)
    fig.text(0.1, 0.5, 'tsne_2', ha='center', va='center', rotation='vertical', size=15)

    plt.savefig(f'reductions-{dataset_name}-{n_iter}.pdf')


def plot_pca_reductions(X_train, y_train, dataset_name, n_iter):
    print("Plotting reductions...")
    fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
    fig.set_figheight(8)
    fig.set_figwidth(8 * 4)

    for i, p in enumerate([5, 30, 50, 100]):
        print("Plotting with p=", p)
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X_train)

        tsne = TSNE(n_components=2, init='pca', perplexity=p, n_iter=n_iter)
        X_reduced = tsne.fit_transform(X_pca)
        axs[i].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train, marker='.', s=1)

        ratio = 1
        x_left, x_right = axs[i].get_xlim()
        y_low, y_high = axs[i].get_ylim()
        axs[i].set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

        axs[i].set_title(f'Perplexity: {p}', size=20)

    fig.text(0.5, 0.1, 'pca-tsne_1', ha='center', va='center', size=15)
    fig.text(0.1, 0.5, 'pca-tsne_2', ha='center', va='center', rotation='vertical', size=15)

    plt.savefig(f'pca-reductions-{dataset_name}-{n_iter}.pdf')



def main():
    print("Generating DatasetLoader...")

    loader = DatasetLoader(data_dir=get_data_dir())
    dataset_kwargs = dict()
    classifier_kwargs = {
        "multi_label": False,
        "lr": 2e-05,
        "scheduler": "slanted",
        "layerwise_gradient_decay": 0.975,
        "mini_batch_size": 12,
        "num_epochs": 50,
        "early_stopping_no_improvement": 5,
        "memory_fix": False,
        "model_selection": True,
        "transformer_model": "bert-base-uncased"
    }

    dataset_kwargs['tokenizer_name'] = classifier_kwargs['transformer_model']
    dataset_kwargs["max_length"] = 60
    dataset_name = "mr"
    classifier_name = "transformer"

    print(f"Loading dataset {dataset_name}...")
    train, test = loader.load_dataset(
        dataset_name,
        dataset_kwargs,
        classifier_name,
        classifier_kwargs
    )

    num_classes = get_num_class(train)

    print("Generating embeddings...")
    args = TransformerModelArguments(classifier_kwargs['transformer_model'])
    tclass = TransformerBasedClassification(args, num_classes=num_classes)
    tclass.fit(train)

    embeddings = tclass.embed(train)
    print("Embedding shape", embeddings.shape)

    y_train = np.array(train.y)

    print("Normalizing embeddings...")
    from sklearn.preprocessing import normalize
    X_train = normalize(embeddings, axis=1)

    calculate_divergences(X_train, dataset_name, n_iter=1000)
    # plot_reductions(X_train, y_train, dataset_name, n_iter=1000)
    # plot_pca_reductions(X_train, y_train, dataset_name, n_iter=5000)


if __name__ == '__main__':
    main()
