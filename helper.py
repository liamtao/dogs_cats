import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import os
import math
import random
import zipfile
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import shutil

from sklearn.utils import shuffle
import pandas as pd
from keras.layers.merge import concatenate
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
import scipy.ndimage as ndimage


from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import *

BATCH_SIZE = 128

from keras.optimizers import *
SGD
# -------- 文件操作 ----------

def make_dir_not_exist(foldername):
    if not os.path.exists(foldername):
        os.mkdir(foldername)

def del_file_if_exist(directory):
    if os.path.exists(directory):
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        else:
            os.remove(directory)


# -------- ImageDataGenerator ----------

def create_image_gen(image_size, dataset="train", preprocess_func=None, shuffle=False, imageAugmented=False):
    '''
    创建在训练或者提取特征时使用的ImageGenerator
    :param image_size: 需要把图像调整到的尺寸
    :param dataset: 决定imagegenerator读取哪个文件夹下的图片，接受以下参数：
            "train"(默认): data/train2
            "splitted_train": data/splitted/train
            "splitted_valid": data/splitted/valid
            "test": data/test2
    :param preprocess_func:
    :param shuffle:
    :param imageAugmented: 是否使用数据增强
    :return:
    '''
    directory = ""
    class_mode = ""
    classes = None
    if dataset == "train":
        directory = "data/train2"
        class_mode = "binary"
        classes = ["cat", "dog"]
    elif dataset == "test":
        directory = "data/test2"
        class_mode = None
        classes = None
    elif dataset == "splitted_train":
        directory = "data/splitted/train"
        class_mode = "binary"
        classes = ["cat", "dog"]
    elif dataset == "splitted_valid":
        directory = "data/splitted/valid"
        class_mode = "binary"
        classes = ["cat", "dog"]
    else:
        raise BaseException("Invalid param: dataset. ")


    data_gen = None
    if not imageAugmented:
        data_gen = ImageDataGenerator(preprocessing_function=preprocess_func)
    else:
        data_gen = ImageDataGenerator(
            preprocessing_function=preprocess_func,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    return data_gen.flow_from_directory(directory,
                                        image_size,
                                        shuffle=shuffle,
                                        batch_size=BATCH_SIZE,
                                        class_mode=class_mode,
                                        classes=classes
                                        )


# -------- 提取特征（训练已有模型并保存输出） ----------
'''
特征文件全部保存在 /feature_extract 文件夹下

文件命名：
gap_预训练模型名称 : 不包含输出层的预训练模型(添加GlobalAveragePool层)提取的特征文件
finetuning_预训练模型名称 : 不包含最后一个block的预训练模型提取的特征文件。
'''

# def _filename_feature(model_name, dataset):
#     if dataset=="pretrained":
#         filename = "gap_%s.h5" % model_name
#     elif dataset=="finetuning":
#         filename = "finetuning_%s.h5" % model_name
#     else:
#         raise BaseException("Unknown dataset, accept value: 'pretrained' / 'finetuning'")
#
#     return filename

def get_filename(model_name, tech, suffix=None, ext="h5"):
    if not tech in ["original","fine","merge"]:
        raise BaseException("unsupport tech value:%s, accept value: 'original' / 'fine' / 'merge'"%tech)

    # if not suffix in [None,"output","feature","trainable","whole"]:
    #     raise BaseException("unsupport suffix value:%s, accept value: 'output' / 'feature' / 'trainable' / None" % suffix)

    if suffix:
        filename = "_".join([tech,model_name,suffix]) + "." + ext
    else:
        filename = "_".join([tech,model_name]) + "." + ext

    return filename

def extract_feature(model, image_size, preprocess_func=None, save_filename="default.h5"):
    '''
    使用model提取data/train2 和 data/test2 文件夹下图片的特征并保存
    :param model:
    :param image_size:
    :param preprocess_func:
    :param save_filename: 输出特征文件的保存位置
    :return:
    '''

    model_name = model.name

    print("========= Extract Featrues by %s==========" % model_name)

    train_generator = create_image_gen(image_size, dataset="train", preprocess_func=preprocess_func)

    test_generator = create_image_gen(image_size, dataset="test", preprocess_func=preprocess_func)

    train_step = math.ceil(train_generator.samples / train_generator.batch_size)
    test_step = math.ceil(test_generator.samples / test_generator.batch_size)

    print("Extracting features from train data by %s ..." % model_name)

    train = model.predict_generator(train_generator,
                                    train_step,
                                    verbose=1)

    print("Extracting features from test data by %s ..." % model_name)
    test = model.predict_generator(test_generator,
                                   test_step,
                                   verbose=1)

    h5py_directory = "feature_extract/%s" % save_filename
    del_file_if_exist(h5py_directory)

    print("saving features to %s" % h5py_directory)
    with h5py.File(h5py_directory) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

    print("Feature saved for %s\n" % model_name)

def extract_feature_by_pretrainedModel(MODEL, image_size, preprocess_func=None):
    input_tensor = Input((image_size[1], image_size[0], 3))
    base_model = MODEL(input_tensor=input_tensor, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    model.name = "%s With GlobalAveragePool"%MODEL.__name__
    save_filename = get_filename(MODEL.__name__, "original")

    extract_feature(model, image_size,
                    preprocess_func=preprocess_func,
                    save_filename=save_filename)

def extract_feature_finetuning(model_name, image_size, preprocess_func=None):
    feature_model,trainable_model = load_model_fine_split(model_name)

    feature_model.name = "%s Untrainable Part"%model_name
    save_filename = get_filename(model_name, "fine")

    extract_feature(feature_model,image_size,preprocess_func,save_filename)


# -------- 读取特征文件 ----------

def load_train_data_by_filename(filename,need_shuffle=False):
    dir_model = "feature_extract/%s" % filename
    print("load features from 「%s」" % filename)
    if not os.path.exists(dir_model):
        raise BaseException("%s not found" % dir_model)
    else:
        with h5py.File(dir_model, 'r') as h:
            X_train = np.array(h['train'])
            y_train = np.array(h['label'])
            if need_shuffle:
                return shuffle(X_train, y_train, random_state=2018)
            else:
                return (X_train, y_train)

def load_test_data_by_filename(filename):
    dir_model = "feature_extract/%s" % filename
    print("load test data from 「%s」" % filename)
    if not os.path.exists(dir_model):
        raise BaseException("%s not found" % dir_model)
    else:
        with h5py.File(dir_model, 'r') as h:
            X_test = np.array(h['test'])
            return X_test

# 根据model_name(例如“VGG16,ResNet50”等)，读取之前保存的h5文件，文件中是预训练模型对train data的输出（不包含top_layer,并加了一层GlobalAverage）
def load_train_data(model_name,dataset="original",need_shuffle=False):
    filename = get_filename(model_name, dataset)
    return load_train_data_by_filename(filename,need_shuffle=need_shuffle)

def load_test_data(model_name,dataset="original"):
    filename = get_filename(model_name, dataset)
    return load_test_data_by_filename(filename)

def load_train_data_merge(model_names):
    '''
    根据model_names list中的预训练模型名称，读取这些模型保存的矢量特征文件。所有特征文件按规定random_state乱序。
    :param model_names: 预训练模型名称列表
    :return: (不同模型的矢量特征X列表, 标签y, Xs的矢量shape)
    '''

    Xs = []
    y = []

    for model_name in model_names:
        X_train, y_train = load_train_data(model_name,need_shuffle=True)
        Xs.append(X_train)

        if len(y) == 0:
            y = y_train

    return Xs,y

def load_test_data_merge(model_names):

    Xs = []

    for model_name in model_names:
        X_train = load_test_data(model_name)
        Xs.append(X_train)

    return Xs

def load_train_data_original(resize=224,total_count = 12500*2,split=True,preprocess_func=None):
    X = np.zeros((total_count, resize, resize, 3), dtype=np.uint8)  # 特征数据
    y = np.zeros((total_count, 1), dtype=np.uint8)  # 标签数据

    half_count = int(total_count / 2)
    for i in tqdm(range(half_count)):
        X[i] = load_img("data/train/cat.{}.jpg".format(i), target_size=(resize, resize))
        X[half_count + i] = load_img("data/train/dog.{}.jpg".format(i), target_size=(resize, resize))

    y[half_count:] = 1

    if preprocess_func:
        X = preprocess_func(X.astype(np.float32))

    if split:
        X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2,random_state=2018)
        return X_train, X_valid, y_train, y_valid
    else:
        return X,y



# -------- 创建模型 ----------

def _create_output_model_core(input_tensor,globalAverageLayer,dropout,drop_rate):
    '''
    根据input_tensor创建 输出层需要用到的多个Tensor，并返回最后的Tensor
    '''
    if globalAverageLayer:
        x = GlobalAveragePooling2D()(input_tensor)
    else:
        x = input_tensor

    if dropout:
        x = Dropout(drop_rate)(x)

    x = Dense(1, activation="sigmoid")(x)
    return x

def create_output_model(input_shape, globalAverageLayer=True,dropout=True,drop_rate=0.5, optimizer="adadelta"):
    '''
    根据input_shape，创建并返回输出层模型
    '''

    input_tensor = Input(shape=input_shape)

    x = _create_output_model_core(input_tensor,globalAverageLayer,dropout,drop_rate)
    model = Model(input_tensor, x)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("model created")
    return model

def attach_output_model(original_model, globalAverageLayer=True,dropout=True,drop_rate=0.5, optimizer="adadelta"):
    '''
    在给定的original_model后追加输出层
    :param original_model: 原模型
    :param globalAverageLayer: 是否需要追加GlobalAveragePool
    :param drop_rate: Dropout层的droprate
    :param optimizer: 合成模型时默认的优化器
    :return: 追加输出层后的整个模型
    '''
    x = _create_output_model_core(original_model.output, globalAverageLayer,dropout, drop_rate)
    model = Model(original_model.input, x)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("model created")
    return model

def create_model_merge_output(input_shapes, drop_rate=0.5, optimizer="adadelta"):
    '''
    根据input_shapes中列出的多个input_shape,创建Input Tensor。然后合并多个Tensor，并且附加输出层模型。最后返回从Inputs到分类层的整个模型
    这里的模型在项目中被用于训练特征向量作为输入的合并模型全连接层。
    '''
    input_tensors = [Input(shape=x) for x in input_shapes]

    concatenated = concatenate(input_tensors)

    model = attach_output_model(Model(input_tensors,concatenated),globalAverageLayer=False,drop_rate=drop_rate,optimizer=optimizer)

    return model


def create_IncXceRes_model_merge():

    input_IncXce = Input(shape=(299,299,3))
    input_res = Input(shape=(224,224,3))

    x_IncXce = Lambda(inception_v3.preprocess_input)(input_IncXce)
    x_res = Lambda(resnet50.preprocess_input)(input_res)


    inc_model = InceptionV3(input_tensor=x_IncXce,weights="imagenet",include_top=False,pooling="avg")
    xce_model = Xception(input_tensor=x_IncXce,weights="imagenet",include_top=False,pooling="avg")
    res_model = ResNet50(input_tensor=x_res, weights="imagenet", include_top=False, pooling="avg")

    concatenated = concatenate([inc_model.output,xce_model.output,res_model.output])

    # 该模型主要用来测试，所以dropout和optimizer保持默认就行
    model = attach_output_model(Model([input_IncXce,input_res],concatenated),globalAverageLayer=False)

    return model



def create_model_fine_whole(MODEL, image_size, n_trainable_layer, drop_rate=0.5,optimizer="adadelta",apply_pretrain_weights=False,dropout=True):
    print("Creating Model...")

    input_tensor = Input((image_size[1], image_size[0], 3))
    base_model = MODEL(weights="imagenet", include_top=False, input_tensor=input_tensor)

    if n_trainable_layer != 0:
        for layer in base_model.layers[:-n_trainable_layer]:
            layer.trainable = False
    elif n_trainable_layer == 0:
        for layer in base_model.layers:
            layer.trainable = False

    final_model = attach_output_model(base_model, drop_rate=drop_rate,optimizer=optimizer,dropout=dropout)



    if apply_pretrain_weights:
        apply_pretrained_output_weights_to_last_layer(final_model, MODEL.__name__)

    return final_model




# -------- 训练模型（包含保存模型） ----------

def fit_model_original_ouput(model_name, epochs=10, callbacks=None, auto_save=False,optimizer="adadelta"):
    '''
    根据model_name读取h5文件, 创建并训练output model
    :param model_name:
    :param epochs:
    :param callbacks:
    :return: 返回一个tuple，(训练模型,训练历史)
    '''
    X_train, y_train = load_train_data(model_name,need_shuffle=True)

    model = create_output_model(X_train.shape[1:],globalAverageLayer=False,optimizer=optimizer)

    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=callbacks)

    if auto_save:
        model.save("model/%s" % get_filename(model_name, "original", "output"))

    return model, history

def fit_model_merge_output(Xs,y, merged_name,epochs=10, optimizer="adadelta", callbacks=None, auto_save=False):
    '''
    根据model_names逐个读取每个预训练模型对应的h5特征向量，并创建output model，完成训练
    :param model_names:
    :param epochs:
    :param optimizer:
    :param callbacks:
    :param auto_save:
    :return: 返回一个tuple，(训练模型,训练历史)
    '''
    # Xs = []
    # y = []
    # input_shapes = []

    # merged_name = "_".join(model_names)

    # for model_name in model_names:
    #     X_train, y_train = load_train_data(model_name,need_shuffle=True)
    #     Xs.append(X_train)
    #     y = y_train
    #     input_shapes.append(X_train.shape[1:])

    # Xs,y = load_train_data_merge(model_names=model_names)

    input_shapes = []

    for feature in Xs:
        input_shapes.append(feature.shape[1:])

    # print(input_shapes)

    model = create_model_merge_output(input_shapes, optimizer=optimizer)

    if auto_save:
        model.save("model/%s" % get_filename(merged_name, "merge", "output"))

    history = model.fit(Xs, y,
                        batch_size=BATCH_SIZE,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=callbacks)
    return model, history

def fit_model_fine_output(model_name, X_train, y_train, epochs=10, callbacks=None, auto_save=False, optimizer="adadelta"):
    '''
    根据Model_name读取对应预训练模型last block, 并接上输出层，读取original模型最后一层的权重并训练。
    :param model_name:
    :param X_train:
    :param y_train:
    :param epochs:
    :param callbacks:
    :param auto_save:
    :param optimizer:
    :return:
    '''
    feature_model, trainable_model = load_model_fine_split(model_name)
    model = attach_output_model(trainable_model,optimizer=optimizer)

    apply_pretrained_output_weights_to_last_layer(model,model_name)

    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=callbacks)

    if auto_save:
        model.save("model/%s" % get_filename(model_name, "fine", "output"))

    return model, history


def fit_model_fine_whole(model, model_name, image_size, n_trainable_layer, preprocess_func=None, epochs_max=20, auto_save=False, augmented=False):
    '''
    创建完整的finetuning模型，并开放最后的n_trainable_layer层，返回训练模型和结果
    :param model:
    :param model_name:
    :param image_size:
    :param n_trainable_layer:
    :param preprocess_func:
    :param epochs_max:
    :param auto_save:
    :return:
    '''
    train_generator = create_image_gen(image_size,
                                       dataset="splitted_train",
                                       preprocess_func=preprocess_func,
                                       shuffle=True,
                                       imageAugmented=augmented)
    valid_generator = create_image_gen(image_size,
                                       dataset="splitted_valid",
                                       preprocess_func=preprocess_func)

    train_step = math.ceil(train_generator.samples / train_generator.batch_size)
    valid_step = math.ceil(valid_generator.samples / valid_generator.batch_size)

    tensorboard_directory = "logs/fine/%s_%s" % (model_name, n_trainable_layer)
    callbacks = [TensorBoard(tensorboard_directory, batch_size=BATCH_SIZE),
                 EarlyStopping(monitor='val_loss', patience=10, verbose=1)]

    del_file_if_exist(tensorboard_directory)
    # i.e. models["VGG"]=(model,history)

    print("Fitting Model...")

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_step,
                                  epochs=epochs_max,
                                  validation_data=valid_generator,
                                  validation_steps=valid_step,
                                  callbacks=callbacks)

    if auto_save:
        model.save("model/%s" % (get_filename(model_name, "fine", "lastopen{}".format(n_trainable_layer))))

    return model, history


# -------- 读取模型 ----------

def load_model_fine_split(model_name):
    feature_model = load_model("model/%s" % get_filename(model_name, "fine", "feature"))
    trainable_model = load_model("model/%s" % get_filename(model_name, "fine", "trainable"))
    return feature_model,trainable_model

def load_model_fine_output(model_name):
    model = load_model("model/%s" % get_filename(model_name, "fine", "output"))
    return model

# -------- 测试模型（导出csv） ----------

def save_pred_to_csv(predict, csv_filename="pred.csv"):
    df = pd.read_csv("sample_submission.csv")

    #     gen = ImageDataGenerator()
    #     test_generator = gen.flow_from_directory("data/test2", (224, 224), shuffle=False,
    #                                              batch_size=BATCH_SIZE, class_mode=None)

    # 此处的image generator只是用来重排预测结果的顺序
    test_generator = create_image_gen((224, 224), dataset="test")

    for i, fname in enumerate(test_generator.filenames):
        index = int(fname[fname.rfind('/') + 1:fname.rfind('.')])
        df.set_value(index - 1, 'label', predict[i])

    print(df.head(10))

    make_dir_not_exist("csv_output")
    output_directory = "csv_output/%s" % csv_filename
    df.to_csv(output_directory, index=None)
    print("CSV文件已保存至%s" % output_directory)

def test_to_csv(model, test_data, csv_filename="pred.csv", clip_min=0.005, clip_max=0.995):
    y_pred = model.predict(test_data, verbose=1)
    y_pred = y_pred.clip(min=clip_min, max=clip_max)

    save_pred_to_csv(y_pred, csv_filename)

def test_generator_to_csv(model, image_size, csv_filename="pred.csv", clip_min=0.005, clip_max=0.995):
    test_generator = create_image_gen(image_size, dataset="test")
    test_step = math.ceil(test_generator.samples / test_generator.batch_size)
    y_pred = model.predict_generator(test_generator,
                                     test_step,
                                     verbose=1)
    y_pred = y_pred.clip(min=clip_min, max=clip_max)

    save_pred_to_csv(y_pred, csv_filename)


# -------- 修改模型的辅助工具 ----------

def _connect_layer(from_layer, to_layer, inbound_layers_index):
    to_layer._inbound_nodes[0].inbound_layers[inbound_layers_index] = from_layer._keras_history[0]
    to_layer._inbound_nodes[0].input_tensors[inbound_layers_index] = from_layer

def replace_connect_layer(original_from_layer,new_from_layer,to_layer):
    '''
    把original_from_layer -> to_layer的flow变更为new_from_layer -> to_layer。只改layer之间的关系
    :param original_from_layer:
    :param new_from_layer:
    :param to_layer:
    :return:
    '''
    for i, each_inbound_layer in enumerate(to_layer._inbound_nodes[0].inbound_layers):
        if each_inbound_layer == original_from_layer:
            _connect_layer(from_layer=new_from_layer, to_layer=to_layer, inbound_layers_index=i)



def split_model(MODEL, image_size, n_trainable_layer, to_save=True):
    '''
    在倒数n_trainable_layer层把MODEL切分成两个模型
    :param MODEL: 要切分的模型
    :param image_size:
    :param n_trainable_layer: 把切分模型倒数层数
    :param to_save: 是否保存模型到model/fine_XXX_trainable.h5 , fine_XXX_feature.h5
    :return: (feature model, trainable model)
    '''
    input_tensor = Input((image_size[1], image_size[0], 3))
    base_model = MODEL(input_tensor=input_tensor, weights='imagenet', include_top=False)

    untrainable_layers = base_model.layers[:-n_trainable_layer]
    trainable_layers = base_model.layers[-n_trainable_layer:]

    untrainable_model = Model(untrainable_layers[0].input, untrainable_layers[-1].output)

    new_input = Input((untrainable_model.output_shape[1:]), name="new_input")

    # 从切分点开始之后的所有layer中，如果已连接了原模型切分点前一个layer，则改变连接到new_input
    last_untrainable_layer = untrainable_layers[-1]
    for layer in trainable_layers:
        replace_connect_layer(original_from_layer=last_untrainable_layer,
                              new_from_layer=new_input,
                              to_layer=layer)

    trainable_model = Model(new_input, trainable_layers[-1].output)

    if to_save:
        print("Saving model:%s" % MODEL.__name__)
        trainable_model.save("model/%s" % get_filename(MODEL.__name__, "fine", "trainable"))
        untrainable_model.save("model/%s" % get_filename(MODEL.__name__, "fine", "feature"))
        print("Done")

    return untrainable_model, trainable_model

def apply_pretrained_output_weights_to_last_layer(new_model, pretrained_model_name):
    model = load_model("model/%s" % get_filename(pretrained_model_name, "original", "output"))
    output_weights = model.layers[-1].get_weights()

    new_model.layers[-1].set_weights(output_weights)
    print("{} pretrained output weights applyed to new model:{}".format(pretrained_model_name,
                                                                        new_model.layers[-1].get_weights()))


def show_most_loss_img(x, y, y_pred, img_num=20):
    y_mistake = np.abs(y - y_pred).flatten()
    y_mistake_indice = np.argsort(-y_mistake)

    column = 5
    row = math.ceil(img_num / column)
    plt.figure(figsize=(12, 3 * row))
    num_cat = len(y) / 2

    for loop_i, mistake_i in enumerate(y_mistake_indice[:img_num]):
        plt.subplot(row, column, loop_i + 1)

        image = array_to_img(x[mistake_i])

        label = "cat" if mistake_i < num_cat else "dog"
        title = "%s_%d_%.4f" % (label, mistake_i % num_cat, y_pred[mistake_i])
        plt.title(title)
        #     plt.text(10,10,title,color="red",verticalalignment="top")

        plt.axis("off")
        plt.imshow(image)


def index_layer(model, name):
    """Index of layer in one model.
    """
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if name in layer.name:
            return i


