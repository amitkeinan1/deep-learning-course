import pickle
from matplotlib import pyplot as plt


def get_accuracies_by_dim():
    with open('results/mlp_and_architecture_metrics.pickle', 'rb') as f:
        results = pickle.load(f)

    hidden_state_dims = []
    test_accuracies = []
    train_accuracies = []

    for hidden_state_dim, hidden_size_results in results.items():
        test_final_metrics, train_final_metrics = hidden_size_results
        test_accuracy, train_accuracy = test_final_metrics[0], train_final_metrics[0]
        hidden_state_dims.append(hidden_state_dim)
        test_accuracies.append(test_accuracy)
        train_accuracies.append(train_accuracy)

    return hidden_state_dims, test_accuracies, train_accuracies


def plot_accuracies_by_hidden_state_dim(hidden_state_dims, test_accuracies, train_accuracies):
    plt.figure()
    plt.title("accuracies by hidden size dimension - MLP")
    plt.xlabel("hidden size dim")
    plt.ylabel("accuracy")
    plt.plot(hidden_state_dims, test_accuracies, 'o--', label="test accuracy")
    plt.plot(hidden_state_dims, train_accuracies, 'o--', label="train accuracy")
    plt.legend()
    plt.savefig('plots/mlp_and_architecture_metrics.png')


def main():
    hidden_state_dims, test_accuracies, train_accuracies = get_accuracies_by_dim()
    plot_accuracies_by_hidden_state_dim(hidden_state_dims, test_accuracies, train_accuracies)


if __name__ == '__main__':
    main()
