import time
import numpy as np
import tensorflow as tf
import pickle

    
def get_batches(arr, n_seqs, n_steps):
    '''
    mini-batchdedで行列を分割
    
    arr: 入力行列
    n_seqs: バッチごとのシーケンス数
    n_steps: シーケンスごとの文字数
    '''
    
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    
    # はみ出た部分を捨てる
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        # inputs
        x = arr[:, n:n+n_steps]
        # targets
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

def build_inputs(num_seqs, num_steps):
    '''
    入力層を構築
    
    num_seqs: バッチごとのシーケンス数
    num_steps: シーケンスごとの文字数
    '''
    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')
    
    # 加入keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' 
    LSTM層の構築
        
    keep_prob
    lstm_size: LSTM隠れ状態サイズ
    num_layers: LSTMセル数
    batch_size: batch_size

    '''
    lstm_cells = []
    for _ in range(num_layers):
        # ベーシックLSTMセル
        lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
        # LSTMセルにDropoutを適用
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        lstm_cells.append(drop)
    
    cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, initial_state

def build_output(lstm_output, in_size, out_size):
    ''' 
    出力層を構築
        
    lstm_output: LSTM層の出力
    in_size: LSTM層をreshape後のサイズ
    out_size: Softmax層のサイズ
    
    '''

    seq_output = tf.concat(lstm_output, 1)
    x = tf.reshape(seq_output, [-1, in_size])
    
    # LSTM層とSoftmax層を全結合
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    
    # Softmax層前の出力
    logits = tf.matmul(x, softmax_w) + softmax_b
    # Softmax確率分布を返す
    out = tf.nn.softmax(logits, name='predictions')
    
    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    '''
    logitsとtargetsの損失を計算
    
    logits: Softmax層前の出力
    targets: targets
    lstm_size
    num_classes: vocab_size
        
    '''
    
    # One-Hotエンコーディング
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    
    return loss

def build_optimizer(loss, learning_rate, grad_clip):
    ''' 
    Optimizerを構築
   
    loss: 損失
    learning_rate: 学習率
    
    '''
    # clipping gradients（勾配爆発にならないよう閾値を設定）
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer

class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
    
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        # 入力層
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # LSTM層
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        # 入力をOne-Hotエンコーディング
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # RNNを実行
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        
        # 結果を予測
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        
        # LossとOptimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


batch_size = 100         # Sequences per batch
num_steps = 100          # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001    # Learning rate
keep_prob = 0.5         # Dropout keep probability
epochs = 20
# 200イテレーションごとにパラメータを保存
save_every_n = 200

if __name__ == '__main__':
    # 毎回生成したタプルの順番が不定
    # データを読み込み
    with open('anna.txt', 'r') as f:
        text = f.read()
        
    # 文字タプル
    vocab = set(text)
    # 文字-数字辞書
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    # 数字-文字辞書
    int_to_vocab = dict(enumerate(vocab))

    # テキストをエンコーディング
    encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
    batches = get_batches(encoded, 10, 50)
    x, y = next(batches)
    anna_pkl = [vocab, vocab_to_int, int_to_vocab, encoded, x, y]
    with open('anna.pkl', 'wb') as fw:
        pickle.dump(anna_pkl, fw)
        
    model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                    lstm_size=lstm_size, num_layers=num_layers, 
                    learning_rate=learning_rate)

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        counter = 0
        for e in range(epochs):
            # Train network
            new_state = sess.run(model.initial_state)
            loss = 0
            for x, y in get_batches(encoded, batch_size, num_steps):
                counter += 1
                start = time.time()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([model.loss, 
                                                    model.final_state, 
                                                    model.optimizer], 
                                                    feed_dict=feed)
                
                end = time.time()
                # control the print lines
                if counter % 100 == 0:
                    print('epoch: {}/{}... '.format(e+1, epochs),
                        'iteration: {}... '.format(counter),
                        'loss: {:.4f}... '.format(batch_loss),
                        '{:.4f} sec/batch'.format((end-start)))

                if (counter % save_every_n == 0):
                    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
        
        saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
