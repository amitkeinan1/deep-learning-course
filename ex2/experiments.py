from Train import train_network
import pickle

MODELS_OUTPUT_DIR = "models"


def single_gru_run():
    return train_network(model_name="GRU",
                         output_size=2,
                         hidden_size=64,
                         num_epochs=5,
                         batch_size=32,
                         atten_size=None,
                         reload_model=False,
                         learning_rate=0.001,
                         test_interval=100,
                         output_dir=MODELS_OUTPUT_DIR)


def single_rnn_run():
    return train_network(model_name="RNN",
                         output_size=2,
                         hidden_size=64,
                         num_epochs=5,
                         batch_size=32,
                         atten_size=None,
                         reload_model=False,
                         learning_rate=0.001,
                         test_interval=100,
                         output_dir=MODELS_OUTPUT_DIR)


def gru_and_hidden_size():
    hidden_sizes = [64, 80, 96, 112, 128]
    metrics_by_hidden_size = {}
    for hidden_size in hidden_sizes:
        print(f"current hidden size is {hidden_size}")
        metrics = train_network(model_name="GRU",
                                output_size=2,
                                hidden_size=hidden_size,
                                num_epochs=5,
                                batch_size=32,
                                atten_size=None,
                                reload_model=False,
                                learning_rate=0.001,
                                test_interval=100,
                                output_dir=MODELS_OUTPUT_DIR)
        print("******")
        print(hidden_size)
        print(metrics)
        metrics_by_hidden_size[hidden_size] = metrics
    print(metrics_by_hidden_size)
    with open('gru_and_hidden_size_metrics.pickle', 'wb') as handle:
        pickle.dump(metrics_by_hidden_size, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    single_rnn_run()
