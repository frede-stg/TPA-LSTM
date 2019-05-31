import os
import tensorflow as tf

from lib.setup import params_setup, logging_config_setup, config_setup
from lib.model_utils import create_graph, load_weights, print_num_of_trainable_parameters
from lib.train import train
from lib.test import test
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt


def main():
    para = params_setup()
    logging_config_setup(para)

    graph, model, data_generator = create_graph(para)

    with tf.Session(config=config_setup(), graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(para, sess, model)
        print_num_of_trainable_parameters()

        try:
            if para.mode == 'train':
                train(para, sess, model, data_generator)
            elif para.mode == 'test':
                obs, predicted = test(para, sess, model, data_generator)
                obs = obs * data_generator.scale + data_generator.min_value
                predicted = predicted * data_generator.scale + data_generator.min_value
                print("MSE: ", mean_squared_error(obs[:, 0], predicted[:, 0]))
                idx = pd.DatetimeIndex(start='2016-10-16', end='2018-11-04', freq='W')
                obs_df = pd.DataFrame(data=obs[:, 0], columns=['Observed'], index=idx)
                pred_df = pd.DataFrame(data=predicted[:, 0], columns=['Predicted'], index=idx)
                df = pd.concat([obs_df, pred_df], axis=1)
                df.plot()
                plt.show()

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            print('Stop')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
