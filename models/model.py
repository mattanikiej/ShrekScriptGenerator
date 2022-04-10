import tensorflow as tf


class Model(tf.keras.Model):
    """
    Class that houses the nlp model
    """

    def __init__(self):
        super().__init__(self)
        self.data_ = None  # all the shrek scripts
        self.vocab_ = None  # all unique characters
        self.char_ids_ = None  # character ids
        self.id_data_ = None  # data set converted to ids
        self.invert_ids_ = None  # inverts chars from ids to chars
        self.tokens_ = None  # ids of the chars
        self.training_data_ = None  # data with features and labels
        self.embedding_ = None  # embedding layer
        self.gru_ = None  # rnn layer
        self.dense_ = None  # output layer

        self.__read_data()  # loads in self.data_
        self.__tokenize_chars()  # loads in self.vocab_, self.char_ids, self.ids_data_, self.invert_ids_, self.tokens_
        self.__create_training_data(100)  # loads self.training_data_

        self.embedding_ = tf.keras.layers.Embedding(len(self.tokens_.get_vocabulary()), 256)
        self.gru_ = tf.keras.layers.GRU(2048,
                                        return_sequences=True,
                                        return_state=True)
        self.dense_ = tf.keras.layers.Dense(len(self.tokens_.get_vocabulary()))

    def __read_data(self):
        """
        Reads in the data
        """
        shrek1 = open('data/shrek1.txt', encoding='utf-8').read()
        shrek2 = open('data/shrek2.txt', encoding='utf-8').read()
        shrek3 = open('data/shrek3.txt', encoding='utf-8').read()
        shrek4 = open('data/shrek4.txt', encoding='utf-8').read()

        self.data_ = shrek1 + ' ' + shrek2 + ' ' + shrek3 + ' ' + shrek4

    def __tokenize_chars(self):
        """
        Convert each character to a numerical value
        :return:
        """
        # get unique characters
        self.vocab_ = sorted(set(self.data_))

        # convert characters to ids
        self.tokens_ = tf.keras.layers.StringLookup(
            vocabulary=list(self.vocab_), mask_token=None)

        # be able to invert the chars form ids to strings
        self.invert_ids_ = tf.keras.layers.StringLookup(
            vocabulary=self.tokens_.get_vocabulary(), invert=True, mask_token=None)

        # convert all chars to ids
        self.char_ids_ = self.tokens_(tf.strings.unicode_split(self.data_, 'UTF-8'))
        # create a data set of ids
        self.id_data_ = tf.data.Dataset.from_tensor_slices(self.char_ids_)

    def __create_training_data(self, seq_length):
        """
        Create batches of training data
        """

        # split data into batches
        sequences = self.id_data_.batch(seq_length + 1, drop_remainder=True)
        self.training_data_ = sequences.map(self.create_feature_label)

        # Batch size
        batch_size = 64

        # Buffer size to shuffle the dataset
        buffer_size = 10000

        self.training_data_ = (
            self.training_data_
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))

    def call(self, inputs, states=None, return_state=False, training=False):
        """
        Calls to the model for each step
        :param inputs: inputs for model
        :param states: states of rnn
        :param return_state: True if we're returning state of rnn
        :param training: true of model is going to be trained
        :return: output and states if return_state = True
        """
        x = inputs
        x = self.embedding_(x, training=training)
        if states is None:
            states = self.gru_.get_initial_state(x)
        x, states = self.gru_(x, initial_state=states, training=training)
        x = self.dense_(x, training=training)

        if return_state:
            return x, states
        else:
            return x

    def join_chars_from_ids(self, ids):
        """
        joings the characters back into strings
        :param ids: ids of the chars
        :return: strings joined back together
        """
        return tf.strings.reduce_join(self.invert_ids_(ids), axis=-1)

    def create_feature_label(self, sequence):
        """
        splits the data into feature and label set where the label is the next char in the sequence
        :param sequence: sequence of chars
        :return: arrays of features and labels
        """
        features = sequence[:-1]
        labels = sequence[1:]
        return features, labels
