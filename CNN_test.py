import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#直接用tensorflow自带的库从网上下载minis库

mnist = input_data.read_data_sets('MNIST_LZY/', one_hot=True)
x = tf.placeholder(tf.float32,[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

def weight_vari(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_vari(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积层和池化层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#reshape image 数据
x_image = tf.reshape(x, [-1,28,28,1])


#第一层
w_conv1 = weight_vari([5,5,1,32])

b_conv1 = bias_vari([32])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层
w_conv2 = weight_vari([5,5,32,64])
b_conv2 = bias_vari([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
w_fc1 = weight_vari([7*7*64, 1024])
b_fc1 = bias_vari([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)

#添加Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_vari([1024,10])
b_fc2 = bias_vari([10])
y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

#训练和评估
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1)), tf.float32))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(800):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0 :
            train_accuracy = accuracy.eval(feed_dict = {x: batch[0],
                                                       y_: batch[1],
                                                       keep_prob: 1.})
            print('setp {},the train accuracy: {}'.format(i, train_accuracy))
        train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
    test_accuracy = accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.})
    print('the test accuracy :{}'.format(test_accuracy))
    saver = tf.train.Saver()
    path = saver.save(sess, './my_net/mnist_deep.ckpt')
    print('save path: {}'.format(path))









