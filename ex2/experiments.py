from Train import train_network

MODELS_OUTPUT_DIR = "models"


def rnn_experiment():
    batch_size = [16, 32, 64, 128]
    output_size = 2
    hidden_size = [64, 80, 96, 112, 128]  # to experiment with

    atten_size = [1, 3, 5, 7]  # need atten > 0 for using restricted self atten
    reload_model = False
    num_epochs = 5  # 30
    learning_rate = [0.001, 0.005, 0.01]
    test_interval = 100
    att_size = 0

    best_f1, best_accuracy, best_recall, best_precision = train_network(model_name="GRU",
                                                                        output_size=output_size,
                                                                        hidden_size=64,
                                                                        num_epochs=num_epochs,
                                                                        batch_size=32,
                                                                        atten_size=None,
                                                                        reload_model=reload_model,
                                                                        learning_rate=0.001,
                                                                        test_interval=test_interval,
                                                                        output_dir=MODELS_OUTPUT_DIR)
    print(best_f1, best_accuracy, best_recall, best_precision)


if __name__ == '__main__':
    rnn_experiment()
