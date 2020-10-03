from tensorflow.keras.layers import Dropout
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def pos_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiheadSelfAttention(tf.keras.Model):
    def __init__(self, d_model, num_heads, mask=False):
        super(MultiheadSelfAttention, self).__init__()
        self.depth = d_model // num_heads
        self.num_heads = num_heads
        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        self.mask = mask

    def create_mask(self, length, width):
        mask = np.zeros((length, width), dtype='float32')
        for i in range(length):
            for j in range(width):
                if j > i:
                    mask[i, j] = -1e9
        return tf.cast(mask, dtype=tf.float32)

    def call(self, query, key, value):
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        batch_size = query.shape[0]
        query = tf.reshape(query, [batch_size, -1, self.num_heads, self.depth])
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.reshape(key, [batch_size, -1, self.num_heads, self.depth])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.reshape(value, [batch_size, -1, self.num_heads, self.depth])
        value = tf.transpose(value, [0, 2, 1, 3])
        score = tf.matmul(query, key, transpose_b=True)
        score /= tf.math.sqrt(tf.dtypes.cast(self.depth, dtype=tf.float32))
        if self.mask:
            score += self.create_mask(score.shape[-2], score.shape[-1])
        alignment = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(alignment, value)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.depth * self.num_heads])
        output = self.dense(context)
        # heads has shape (batch, decoder_len, model_size)
        return output


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, d_model, num_heads, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropouts1 = [Dropout(dropout) for i in range(num_layers)]
        self.dropouts2 = [Dropout(dropout) for i in range(num_layers)]
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, d_model)
        self.mha = [MultiheadSelfAttention(d_model, num_heads) for _ in range(num_layers)]
        self.layer_normalization_1 = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.layer_normalization_2 = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.dense_1 = [tf.keras.layers.Dense(d_model * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(d_model) for _ in range(num_layers)]

    def call(self, encoder_inputs):
        encoder_outputs = self.embedding(encoder_inputs)
        encoder_outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        encoder_outputs += pos_encoding(encoder_outputs.shape[1], encoder_outputs.shape[2])
        for i in range(self.num_layers):
            attention = self.mha[i](query=encoder_outputs, key=encoder_outputs, value=encoder_outputs)
            attention = self.dropouts1[i](attention)
            attention = self.layer_normalization_1[i](encoder_outputs + attention)
            encoder_outputs = self.dense_1[i](attention)
            encoder_outputs = self.dense_2[i](encoder_outputs)
            encoder_outputs = self.dropouts2[i](encoder_outputs)
            encoder_outputs = self.layer_normalization_2[i](attention + encoder_outputs)
        return encoder_outputs


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, d_model, num_heads, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, d_model)
        self.masked_mha = [MultiheadSelfAttention(d_model, num_heads, mask=True) for _ in range(num_layers)]
        self.layer_normalization_1 = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.layer_normalization_2 = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.layer_normalization_3 = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.dropouts1 = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]
        self.dropouts2 = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]
        self.dropouts3 = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]
        self.mha = [MultiheadSelfAttention(d_model, num_heads) for _ in range(num_layers)]
        self.dense_1 = [tf.keras.layers.Dense(d_model * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(d_model) for _ in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size+1, activation = 'softmax')

    def call(self, encoder_outputs, decoder_inputs):
        decoder_outputs = self.embedding(decoder_inputs)
        decoder_outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        decoder_outputs += pos_encoding(decoder_outputs.shape[1], decoder_outputs.shape[2])
        for i in range(self.num_layers):
            masked_attention = self.masked_mha[i](query=decoder_outputs, key=decoder_outputs, value=decoder_outputs)
            masked_attention = self.dropouts1[i](masked_attention)
            masked_attention = self.layer_normalization_1[i](masked_attention + decoder_outputs)
            attention = self.mha[i](query=masked_attention, key=encoder_outputs, value=encoder_outputs)
            attention = self.dropouts2[i](attention)
            attention = self.layer_normalization_2[i](attention+masked_attention)
            decoder_outputs = self.dense_1[i](attention)
            decoder_outputs = self.dense_2[i](decoder_outputs)
            decoder_outputs = self.dropouts3[i](decoder_outputs)
            decoder_outputs = self.layer_normalization_3[i](decoder_outputs + attention)
        decoder_outputs = self.dense(decoder_outputs)
        return decoder_outputs


class Transformer(tf.keras.Model):

    def __init__(self, input_vocab_size, target_vocab_size, d_model, num_heads, num_layers_encoder=1,
                 num_layers_decoder=1):
        super().__init__()
        self.encoder = Encoder(vocab_size=input_vocab_size, d_model=d_model, num_heads=num_heads,
                               num_layers=num_layers_encoder)
        self.decoder = Decoder(vocab_size=target_vocab_size, d_model=d_model, num_heads=num_heads,
                               num_layers=num_layers_decoder)

    def call(self, inputs):
        encoder_outputs = self.encoder(inputs[0])
        output = self.decoder(encoder_outputs, inputs[1])
        return output


with open('C:\\Users\\anton\\PycharmProjects\\transformer\\venv\\transformer_model\\input_dict.json', 'r') as fp:
    input_dict = json.load(fp)
with open('C:\\Users\\anton\\PycharmProjects\\transformer\\venv\\transformer_model\\target_dict.json', 'r') as fp:
    target_dict = json.load(fp)
with open('C:\\Users\\anton\\PycharmProjects\\transformer\\venv\\transformer_model\\target_dict_r.json', 'r') as fp:
    target_dict_r = json.load(fp)


def encode(sentence):
    sentence = sentence.split()
    for i, el  in enumerate(sentence):
        sentence[i] = input_dict[el]
    sentence = pad_sequences([sentence], maxlen=20, padding='post')
    return sentence


def predict(sentence):
    enc_input = encode(sentence)
    print(sentence)
    out_words = []
    de_input = tf.constant([[1]], dtype=tf.int64)
    while True:
        de_output = transformer([enc_input, de_input])
        new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
        de_input = tf.concat((de_input, new_word), axis=-1)
        out_words.append(target_dict_r[str(new_word.numpy()[0][0])])
        if out_words[-1] == 'end_token' or len(out_words) >= 20:
            break
    return ' '.join(out_words[:-1])


num_heads = 8
d_model = 256
input_vocab_size = len(input_dict)
target_vocab_size = len(target_dict)
batch_size=100
transformer = Transformer(input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size, d_model=d_model, num_heads=num_heads, num_layers_encoder=1, num_layers_decoder=1)
transformer.load_weights('C:\\Users\\anton\\PycharmProjects\\transformer\\venv\\transformer_model\\my_model')
print(tf.__version__)
