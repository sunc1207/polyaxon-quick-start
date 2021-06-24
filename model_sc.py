#!/usr/bin/python
#
# Copyright 2018-2020 Polyaxon, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.keras.models import Sequential

from polyaxon import tracking
from polyaxon.tracking.contrib.keras import PolyaxonCallback

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

if __name__ == '__main__':
    # 下载数据
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print("数据读入")

    # 缩小数据规模 视情况而定 如果给的数据集太大 跑不动可以使用
    x_train = x_train[:5000]
    y_train = y_train[:5000]

    x_test = x_test[:1000]
    y_test = y_test[:1000]

    # 处理数据
    # 归一化
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    num_classes = 10

    # Polyaxon
    tracking.init()
    tracking.log_data_ref(content=x_train, name='x_train')
    tracking.log_data_ref(content=y_train, name='y_train')
    tracking.log_data_ref(content=x_test, name='x_test')
    tracking.log_data_ref(content=y_test, name='y_train')

    plx_callback = PolyaxonCallback()
    log_dir = tracking.get_tensorboard_path()

    print("log_dir", log_dir)
    print("model_dir", plx_callback.filepath)

    # 搭建模型
    model = Sequential([
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    print("模型搭建结束")
    # 查看网络结构
    model.summary()
    # 编译模型
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                  metrics=['accuracy'])

    # 训练模型
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=2,
        update_freq=1000
    )

    model.fit(x=x_train,
              y=y_train,
              batch_size=32,
              epochs=10,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback, plx_callback])

    print("模型训练结束")