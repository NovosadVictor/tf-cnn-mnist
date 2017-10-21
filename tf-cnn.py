import tensorflow as tf

#load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/', one_hot=True)

#hyperparameters
learning_rate = 0.001

n_epoch = 10
batch_size = 128

h1_size = 1024
h2_size = 700
n_classes = 10

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, n_classes])

W = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wf1': tf.Variable(tf.random_normal([7 * 7 * 64, h1_size])),
    'wf2': tf.Variable(tf.random_normal([h1_size, h2_size])),
    'out': tf.Variable(tf.random_normal([h2_size, n_classes])),
}

b = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bf1': tf.Variable(tf.random_normal([h1_size])),
    'bf2': tf.Variable(tf.random_normal([h2_size])),
    'out': tf.Variable(tf.random_normal([n_classes])),
}


def conv2d(x, W, b, strides=1):

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.add(x, b)
    return tf.nn.relu(x)


def max_pool(x, k=2):

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def feed_forward(x, W, b):

    x = tf.reshape(x, [-1, 28, 28, 1])

    conv1 = conv2d(x, W['wc1'], b['bc1'])
    conv1 = max_pool(conv1)

    conv2 = conv2d(conv1, W['wc2'], b['bc2'])
    conv2 = max_pool(conv2)

    fc1 = tf.reshape(conv2, [-1, W['wf1'].get_shape().as_list()[0]])

    h1 = tf.add(tf.matmul(fc1, W['wf1']), b['bf1'])
    h1 = tf.nn.relu(h1)

    h2 = tf.add(tf.matmul(h1, W['wf2']), b['bf2'])
    h2 = tf.nn.relu(h2)

    output = tf.add(tf.matmul(h2, W['out']), b['out'])

    return output


def train_cnn():
    prediction = feed_forward(x, W, b)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(n_epoch):
            epoch_loss = 0.0
            for _ in range(mnist.train.num_examples // batch_size):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, loss = sess.run([optimizer, cost], {x: epoch_x, y: epoch_y})
                epoch_loss += loss
            print("Epoch", epoch, "of", n_epoch, ", Loss is: ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        print("Accuracy is: ", sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}))


train_cnn()
