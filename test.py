from train import *

def pick_top_n(preds, vocab_size, top_n=5):
    """
    予測結果からtop_n個の文字を選び出す
    
    preds: 予測結果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # top_n個の文字以外を全部0にする
    p[np.argsort(p)[:-top_n]] = 0
    # 標準化
    p = p / np.sum(p)
    # ランダムに文字を1個選ぶ
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    """
    新しい文を生成
    
    checkpoint: 保存したパラメータ
    n_sample: 文字数
    lstm_size: LSTM隠れ状態のサイズ
    vocab_size
    prime: 文のはじめの単語
    """
    # 単語を文字リストに変換
    samples = [c for c in prime]
    # sampling=Trueで、batchサイズが1 x 1になる
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # パラメータを読み込む
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            # 文字1個を入力
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])
        
        # 文字をn_samples回生成する
        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)


if __name__ == '__main__':
    with open('./anna.pkl', 'rb') as fr:
        vocab, vocab_to_int, int_to_vocab, encoded, x, y = pickle.load(fr)
    checkpoint = tf.train.latest_checkpoint('checkpoints')
    samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="The")
    print(samp)