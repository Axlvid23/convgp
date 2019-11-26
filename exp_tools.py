import contextlib
import os
import sys
import random
import jug
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import spectral.io.envi as envi
import os
import string
from csv import reader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import GPflow
import GPflow.minibatch as mb
import opt_tools
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

@jug.TaskGenerator
def jugrun_experiment(exp):
    print("Running %s..." % exp.experiment_name)
    exp.setup()
    try:
        exp.run()
    except opt_tools.OptimisationTimeout:
        print("Timeout")


@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def load_stanford():


    ST_sentence = namedtuple("Stanford_Sentiment", "id sentence")
    ST_score = namedtuple("Stanford_Sentiment", "id score")

    sentences = dict()
    scores = dict()
    train = []
    dev = []
    test = []

    for filename in ['datasetSentences.txt', 'datasetSplit.txt', 'sentiment_labels.txt']:
        with open("datasets/" + filename, 'r') as f_input:
            for line in f_input:

                # skip headers
                if line.startswith("sentence_index") or line.startswith('phrase id'):
                    continue

                # load sentences
                if filename == 'datasetSentences.txt':
                    sent_id, sentence = line.split('\t', 1)
                    sentences[sent_id] = sentence.strip()

                # load splits
                if filename == 'datasetSplit.txt':
                    sent_id, split = line.split(',', 1)
                    split = int(split.strip())
                    if split == 1:
                        train.append(sent_id)
                    if split == 2:
                        test.append(sent_id)
                    if split == 3:
                        dev.append(sent_id)

                # sentences_id
                if filename == 'sentiment_labels.txt':
                    sent_id, sent_score = line.split('|', 1)
                    # sent_score = float(sent_score.strip())
                    sample = ST_score(sent_id, float(sent_score.strip()))
                    scores[sent_id] = sent_score.strip()

    print("Total Nr. Samples: {}".format(len(sentences)))
    print("Total Nr. Scores : {}".format(len(scores)))
    print()
    print("Train  : {}".format(len(train)))
    print("Dev    : {}".format(len(dev)))
    print("Test   : {}".format(len(test)))

    # built two lists with sentences and labels
    x_train_data = [sentences[x] for x in train]
    y_train_data = [scores[x] for x in train]

    x_dev_data = [sentences[x] for x in dev]
    y_dev_data = [scores[x] for x in dev]

    x_test_data = [sentences[x] for x in test]
    y_test_data = [scores[x] for x in test]

    # convert list of tokens/words to indexes
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train_data)
    sequences_train = tokenizer.texts_to_sequences(x_train_data)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    max_input_length = max([len(x) for x in sequences_train])
    print('max input length', max_input_length)

    # pad all the sequences of indexes to the 'max_input_lenght'
    x_train_data_padded = pad_sequences(sequences_train, maxlen=max_input_length, padding='post', truncating='post')
    x_dev_data_padded = pad_sequences(tokenizer.texts_to_sequences(x_dev_data), maxlen=max_input_length, padding='post',
                                      truncating='post')
    x_test_data_padded = pad_sequences(tokenizer.texts_to_sequences(x_test_data), maxlen=max_input_length,
                                       padding='post', truncating='post')

    def convert_to_categories(y_data):
        y_categories = []
        for score in y_data:
            if 0.0 <= float(score) < 0.2:
                y_categories.append('very_negative')
            elif 0.2 <= float(score) < 0.4:
                y_categories.append('negative')
            elif 0.4 <= float(score) < 0.6:
                y_categories.append('neutral')
            elif 0.6 <= float(score) < 0.8:
                y_categories.append('positive')
            elif 0.8 <= float(score) <= 1.0:
                y_categories.append('very positive')

        return y_categories

    # Convert from scores to categories
    y_train_data_categ = convert_to_categories(y_train_data)
    y_dev_data_categ = convert_to_categories(y_dev_data)
    y_test_data_categ = convert_to_categories(y_test_data)

    # Encode the labels, each must be a vector with dim = num. of possible labels
    le = LabelEncoder()
    le.fit(y_train_data_categ)

    labels_encoded_train = le.transform(y_train_data_categ)
    labels_encoded_dev = le.transform(y_dev_data_categ)
    labels_encoded_test = le.transform(y_test_data_categ)

    categorical_labels_train = to_categorical(labels_encoded_train, num_classes=5)
    categorical_labels_dev = to_categorical(labels_encoded_dev, num_classes=5)
    categorical_labels_test = to_categorical(labels_encoded_test, num_classes=5)

    print(x_train_data_padded.shape)
    print(categorical_labels_train.shape)

    print(x_dev_data_padded.shape)
    print(labels_encoded_dev.shape)

    print(x_test_data_padded.shape)
    print(categorical_labels_test.shape)

    y = categorical_labels_train.astype('int64')
    yt = categorical_labels_test.astype('int64')
    x = x_train_data_padded
    #x = x_train_data_padded.reshape(8544,7,7).astype(np.float32)
    xt = x_test_data_padded
    #xt = x_test_data_padded.reshape(2210,7,7).astype(np.float32)

    return x,y,xt,yt

x,y,xt,yt = load_stanford()

print(x.__class__)
print(y.__class__)

print('xshape',x.shape)
print('yshape',y.shape)
print(len(y))


def load_mnist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X = np.vstack((mnist.train.images.astype(float), mnist.validation.images.astype('float')))
    Y = np.vstack((np.argmax(mnist.train.labels, 1)[:, None],
                   np.argmax(mnist.validation.labels, 1)[:, None]))
    Xt = mnist.test.images.astype(float)
    Yt = np.argmax(mnist.test.labels, 1)[:, None]
    return X, Y, Xt, Yt

def load_caserm():
    os.chdir('/home/alexander/Documents/CSM-Statistics /Semester 3/Center Research')
    img = envi.open('324620.hdr', '324620.dat')
    bands = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for j in np.arange(0, img.shape[0]):
        for k in np.arange(0, img.shape[1]):
            bands[j, k, :] = img[j, k, :]

    # This will plot the hyperspectrum for all pixels in the 140th row and save it in the direcory.
    # plt.figure(figsize=(12, 8))
    # plt.plot(bands[:, :, 60].T, color='gray', alpha=0.3)
    # plt.show()
    # plt.savefig('Example.png')

    bands = bands.T
    # global_number = 324320
    global_number = 324620

    # Load a CSV file
    def load_csv(filename):

        dataset = list()

        with open(filename, 'r', encoding="utf8") as file:
            csv_reader = reader(file)

            # csv.reader(open(filename, "r", encoding="utf8"))

            for row in csv_reader:
                # scores = list()

                # if not row:
                #
                # 	continue

                # here we convert string to integer

                # question: int vs float data in the "Mineralogy_324320.csv"???

                scores_int = [int(score) for score in row]

                # string input

                # dataset.append(row)

                # integer input

                dataset.append(scores_int)

        return dataset

    def print_csv_head(dataset, lines=0):

        for line_number, row in zip(range(lines), dataset):
            print(row)

            print("The data correpodns to the line number", line_number + 1)

    def print_csv_tail(dataset, lines=0):

        start = int(np.shape(dataset)[0] - lines)

        finish = len(dataset)

        for line_number, row in zip(range(start, finish), dataset[start: finish + 1]):
            print(row)

            print("The data correpodns to the line number", line_number + 1)

    # dealing with actual data called "Mineralogy_324320.csv"

    def data_print():

        folder_name = "/home/alexander/Documents/CSM-Statistics /Semester 3/Center Research/"

        file_name = "Mineralogy_" + str(global_number) + ".csv"

        full_name = folder_name + file_name

        dataset = load_csv(full_name)

        # print("Dimensions of the dataset are", np.shape(daaset))

        # print("The current dataset is", dataset)

        # print_csv_head(dataset, lines = 5)

        print_csv_tail(dataset, lines=10)

        print("Dimensions of the dataset are", np.shape(dataset))

    # data_print()

    def excel_reference(s=None):

        alpha_arr = list(string.ascii_uppercase)

        comb_arr = [str(i) + str(j) for i in alpha_arr for j in alpha_arr]

        excel_arr = alpha_arr + comb_arr

        # print(excel_arr)

        letter_index = excel_arr.index(str(s)) + 1

        print("The index of your input {} is {}". \
              format(s, letter_index))

        return letter_index

    def dataset_cut(side=5):

        # Automated mineralogy pixels (from excel): AA156:AG162

        # Automated mineralogy pixels: PF630:PK636

        # top_left = [excel_reference('AA'), 156]
        #
        # bottom_right = [excel_reference('PK'), 636]

        # top_left = [excel_reference('BE'), 158]

        # bottom_right = [excel_reference('OT'), 642]

        # here is the manual input  from Alex & David

        top_left = [38, 165]

        # bottom_right = top_left + [5,0]

        bottom_right = [372 + 1, 584 + 1]

        # bottom_right = [38 + 5, 165 + 5]

        # bottom_right = list(map(operator.add, top_left, np.ones(2)*5))

        # top_left = [38, 250]

        # bottom_right = [x + y for x, y in zip(top_left, [side, side])]

        # dataset = [38:372,165:584]

        # BE158:BJ163

        # OO637:OT642

        print("The coordinates are {} and {}". \
              format(top_left, bottom_right))

        folder_name = "/home/alexander/Documents/CSM-Statistics /Semester 3/Center Research/"

        file_name = str(global_number) + ".csv"

        full_name = folder_name + file_name

        dataset = load_csv(full_name)

        dataset_cut = np.array(dataset).T[top_left[0]:bottom_right[0],
                      top_left[1]:bottom_right[1]]

        print("The dimensions of the original dataset are", np.shape(dataset))

        print("The dimensions of the reduced dataset are", np.shape(dataset_cut))

        return dataset_cut

    dataset_cut()

    # excel_reference("YR")

    def colorscheme():

        colors = []

        for name, hex in matplotlib.colors.cnames.items():
            colors.append(name)

        print(colors)

        return colors

    # colorscheme()

    def unique_elements(arr):

        x, y = np.shape(arr)

        arr_1d = np.reshape(arr, x * y, 1)

        unique_arr = sorted(list(set(arr_1d)))

        dic_elem = {}

        for kk, elem in enumerate(arr_1d):

            if elem in dic_elem.keys():

                dic_elem[elem] += 1

            else:

                dic_elem[elem] = 1

        sorted_arr = sorted(dic_elem.items())

        sorted_arr = sorted(sorted_arr, key=lambda x: x[1])[::-1]

        return [unique_arr, sorted_arr]

    def most_prevalent_element_arr(side=5):

        # side = 15

        dataset = dataset_cut()

        print("*" * 70)

        print("The shape of the array is ", np.shape(dataset))

        xx, yy = np.shape(dataset)

        arr_numbers = unique_elements(dataset)[0]

        arr_dict = unique_elements(dataset)[1]

        # print("The unique elements are", arr_numbers)

        # print("The dictionary of unique elements is", arr_dict)

        common_final_arr = np.tile(-1, (xx, yy))

        common_reduced_final_arr = np.tile(-1, (int(xx / 5), int(yy / 5)))

        for kk in range(0, xx, 5):

            for ll in range(0, yy, 5):
                test_arr = dataset[kk:kk + 5, ll:ll + 5]

                test_arr_dict = unique_elements(test_arr)[1]

                popular_element = test_arr_dict[0][0]

                common_test_arr = np.tile(1, (5, 5)) * popular_element

                common_final_arr[kk:kk + 5, ll:ll + 5] = common_test_arr

                common_reduced_final_arr[int(kk / 5), int(ll / 5)] = popular_element

            # print("The current test array is \n", test_arr)

            # print("The dictionary of unique elements is", test_arr_dict)

            # print("The common array is \n", common_test_arr)

            # print("The common final array \n", common_final_arr)

        return (common_final_arr, common_reduced_final_arr)

    bands = bands.T
    common_final_arr, common_reduced_final_arr = most_prevalent_element_arr()
    bands_final_arr = bands[65:132, 122:205 + 1, :]

    X = bands_final_arr
    y = common_reduced_final_arr
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    nsamples, nx, ny = X_train.shape
    d2_X_train = X_train.reshape(nsamples * nx, ny)
    d2_y_train = y_train.reshape(4452, 1)
    d2_y_train = np.where(d2_y_train == 2, 0, d2_y_train)
    d2_y_train = np.where(d2_y_train == 4, 1, d2_y_train)
    d2_y_train = np.where(d2_y_train == 11, 2, d2_y_train)
    nsamplest, nxt, nyt = X_test.shape
    d2_X_test = X_test.reshape(nsamplest * nxt, nyt)
    d2_y_test = y_test.reshape(1176,1)
    d2_y_test = np.where(d2_y_test == 2, 0, d2_y_test)
    d2_y_test = np.where(d2_y_test == 4, 1, d2_y_test)
    d2_y_test = np.where(d2_y_test == 11, 2, d2_y_test)

    X = d2_X_train
    Xt = d2_X_test
    y = d2_y_train
    yt = d2_y_test

    Xc = (X - X.mean()) / X.std()
    yc = (y - y.mean()) / y.std()


    Xtc = (Xt)# - Xt.mean()) / Xt.std()
    ytc = (yt)# - yt.mean()) / yt.std()
    return Xc, yc, Xtc, ytc


#X,y,Xt,yt = load_caserm()
#print('X shape', X.shape)
#print('y shape', y.shape)
#print('Xt shape', Xt.shape)
#print('yt shape', yt.shape)

class ExperimentBase(object):
    def __init__(self, name):
        self.experiment_name = name
        self.m = None
        self.logger = None
        self.X = None
        self.Y = None
        self.Xt = None
        self.Yt = None
        self.run_settings = {}

    def setup_dataset(self, verbose=False):
        raise NotImplementedError

    def setup_model(self):
        raise NotImplementedError

    def setup_logger(self, verbose=False):
        raise NotImplementedError

    def setup(self, verbose=False):
        """
        setup
        Setup logger, model and anything else that isn't picklable.
        :return:
        """
        self.setup_dataset(verbose)
        self.setup_model()
        self.setup_logger(verbose)
        return self.m, self.logger

    def run(self, maxiter=np.inf):
        optimiser = self.run_settings.get("optimiser", "adam")
        if optimiser == "adam":
            opt_method = tf.train.AdamOptimizer(self.run_settings['learning_rate'])
        elif optimiser == "rmsprop":
            opt_method = tf.train.RMSPropOptimizer(self.run_settings['learning_rate'])
        else:
            opt_method = optimiser

        self.opt_method = opt_method

        try:
            return self.logger.optimize(method=opt_method, maxiter=maxiter, opt_options=None)
        finally:
            self.logger.finish(self.m.get_free_state())

    def profile(self):
        """
        profile
        Run a few iterations and dump the timeline.
        :return:
        """
        s = GPflow.settings.get_settings()
        s.profiling.dump_timeline = True
        s.profiling.output_file_name = "./trace_" + self.experiment_name
        with GPflow.settings.temp_settings(s):
            self.m._compile()
            self.m._objective(self.m.get_free_state())
            self.m._objective(self.m.get_free_state())
            self.m._objective(self.m.get_free_state())

    def load_results(self):
        return pd.read_pickle(self.hist_path)

    @property
    def base_filename(self):
        return os.path.join('.', 'results', self.experiment_name)

    @property
    def hist_path(self):
        return self.base_filename + '_hist.pkl'

    @property
    def param_path(self):
        return self.base_filename + '_params.pkl'

    def __jug_hash__(self):
        from jug.hash import hash_one
        return hash_one(self.experiment_name)


class CifarExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        d = np.load('./datasets/cifar10.npz')
        self.X = (d['X'] / 255.0).reshape(50000, 3, 32, 32).swapaxes(1, 3).reshape(50000, -1)
        self.Y = d['Y'].astype('int64')
        self.Xt = (d['Xt'] / 255.0).reshape(10000, 3, 32, 32).swapaxes(1, 3).reshape(10000, -1)
        self.Yt = d['Yt'].astype('int64')

    def img_plot(self, i):
        import matplotlib.pyplot as plt
        plt.imshow(self.X[i, :].reshape(32, 32, 3))

class StanfordExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        with suppress_print():
            self.X, self.Y, self.Xt, self.Yt = load_stanford()

class CasermExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        with suppress_print():
            self.X, self.Y, self.Xt, self.Yt= load_caserm()


class MnistExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        with suppress_print():
            self.X, self.Y, self.Xt, self.Yt = load_mnist()



class RectanglesImageExperiment(ExperimentBase):
    def setup_dataset(self, verbose=False):
        d = np.load('datasets/rectangles_im.npz')
        self.X, self.Y, self.Xt, self.Yt = d['X'], d['Y'], d['Xtest'], d['Ytest']


def calculate_large_batch_lml(m, minibatch_size, batches, progress=False):
    """
    This does not work properly yet, presumably because it changes the state (w.r.t. _parent) of the model.
    """
    assert type(batches) == int, "`batches` must be an integer."
    old_mbX = m.X
    old_mbY = m.Y
    m.X = mb.MinibatchData(m.X.value, minibatch_size,
                           batch_manager=mb.SequenceIndices(minibatch_size, m.X.value.shape[0]))
    m.Y = mb.MinibatchData(m.Y.value, minibatch_size,
                           batch_manager=mb.SequenceIndices(minibatch_size, m.X.value.shape[0]))
    m._kill_autoflow()

    batch_lmls = []
    if progress:
        from tqdm import tqdm
        for _ in tqdm(range(batches)):
            batch_lmls.append(m.compute_log_likelihood())
    else:
        for _ in range(batches):
            batch_lmls.append(m.compute_log_likelihood())

    m.X = old_mbX
    m.Y = old_mbY

    m._kill_autoflow()

    import gc
    gc.collect()

    return np.mean(batch_lmls)


class CalculateFullLMLMixin(object):
    def _get_record(self, logger, x, f=None):
        log_dict = super(CalculateFullLMLMixin, self)._get_record(logger, x, f)
        model = logger.model
        minibatch_size = logger.model.X.index_manager.minibatch_size
        lml = calculate_large_batch_lml(model, minibatch_size, model.X.shape[0] // minibatch_size, True)
        print("full lml: %f" % lml)
        log_dict.update({"lml": lml})
        return log_dict


class GPflowMultiClassificationTrackerLml(CalculateFullLMLMixin,
                                          opt_tools.gpflow_tasks.GPflowMultiClassificationTracker):
    pass


class GPflowTrackLml(opt_tools.tasks.GPflowLogOptimisation):
    def _get_record(self, logger, x, f=None):
        model = logger.model
        minibatch_size = logger.model.X.index_manager.minibatch_size
        lml = calculate_large_batch_lml(model, minibatch_size, model.X.shape[0] // minibatch_size, True)
        print("full lml: %f" % lml)
        return {"i": logger._i, "t": logger.model.num_fevals, "t": logger._opt_timer.elapsed_time,
                "tt": logger._total_timer.elapsed_time, "lml": lml}
