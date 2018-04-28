from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)#数据集加载
print(mnist.train.images.shape,mnist.train.labels.shape)
import  tensorflow as tf
sess=tf.InteractiveSession()
in_unit=784
h1_units=300
W1=tf.Variable(tf.truncated_normal([in_unit,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))
x=tf.placeholder(tf.float32,[None,784])
keep_prob=tf.placeholder(tf.float32)
hidden1=tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_prob=tf.nn.dropout(hidden1,keep_prob)
y=tf.nn.softmax(tf.matmul(hidden1_prob,W2)+b2)#预测的值
y_=tf.placeholder(tf.float32,[None,10])
s=y_*tf.log(y)
print(s.shape)
#[100,10]的维度，把第二个维度相加，成了【100】的一维向量
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=1))
print(cross_entropy.shape)
tranin_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
tf.global_variables_initializer().run()
for i in range(10000):
    batx,baty=mnist.train.next_batch(100)
    tranin_step.run({x:batx,y_:baty,keep_prob:0.75})
correct_pre=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
print(correct_pre.shape)
accur=tf.reduce_mean(tf.cast(correct_pre,tf.float32))
print(accur.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
