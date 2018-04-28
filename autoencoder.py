import  tensorflow as tf
import  numpy as np
import  matplotlib.pyplot as plt
from  tensorflow.examples.tutorials.mnist import  input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=False)

learning_rate=0.01
training_epochs=10
batch_size=256
display_step=1
example_toshow=10
n_input=784

X=tf.placeholder(tf.float32,[None,n_input])

weights={'encoder_h1':tf.Variable(tf.random_normal([784,256])),
         'encoder_h2':tf.Variable(tf.random_normal([256,128])),
         'encoder_h3':tf.Variable(tf.random_normal([128,56])),
         'decoder_h1':tf.Variable(tf.random_normal([56,128])),
         'decoder_h2':tf.Variable(tf.random_normal([128,256])),
         'decoder_h3':tf.Variable(tf.random_normal([256,784]))

         }
biases={
    'encoder_b1':tf.Variable(tf.random_normal([256])),
    'encoder_b2':tf.Variable(tf.random_normal([128])),
    'encoder_b3':tf.Variable(tf.random_normal([56])),
    'decoder_b1':tf.Variable(tf.random_normal([128])),
    'decoder_b2':tf.Variable(tf.random_normal([256])),
    'decoder_b3':tf.Variable(tf.random_normal([784]))
}

def encoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    return layer_3

def decoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    return layer_3
encoder_op=encoder(X)
decoder_op=decoder(encoder_op)

y_pred=decoder_op
y_true=X

cost=tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.AdamOptimizer(0.01).minimize(cost)
#流程图架构完成
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)  # 总批数
    for epoch in range(10):
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={X:batch_xs})
        if epoch %display_step==0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    encoder_decoder=sess.run(y_pred, feed_dict={X: mnist.test.images[:example_toshow]})
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(example_toshow):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encoder_decoder[i], (28, 28)))
    plt.show()











