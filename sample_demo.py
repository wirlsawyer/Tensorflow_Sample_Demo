import random
import tensorflow as tf
import numpy as np

# Data sets
DATA_TRAINING = "sample.csv"
DATA_TEST = "sample.csv"

x_dimension = 2 # input
y_dimension = 4 # output

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=DATA_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
      
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=DATA_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)
      
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    global Weights
    global biases
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        
    
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
    
def make_y(value, max):
    result_y = []
    for i in range(max):
        if (i == value):
            result_y.append(1)
        else:
            result_y.append(0)    
    result_y = np.array(result_y) 
    return result_y
            
def next_batch(max, ds):
    result_x = []
    result_y = []
    dataCount = len(ds.data)-1
    for i in range(max):
        index = random.randint(0, dataCount)
        result_x.append(ds.data[index])
        result_y.append(make_y(ds.target[index], y_dimension)) # 0 1 2 3 4  below case only
    result_x = np.array(result_x)
    result_y = np.array(result_y) 
    return result_x, result_y
    
def fix_targetData(ds):
    result_y = []
    dataCount = len(ds.data)
    for i in range(dataCount):
        result_y.append(make_y(ds.target[i], y_dimension)) # 0 1 2 3 4  below case only
    result_y = np.array(result_y) 
    return result_y
    
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, x_dimension]) # input
ys = tf.placeholder(tf.float32, [None, y_dimension]) # output

# add output layer
prediction = add_layer(xs, x_dimension, y_dimension, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)


for i in range(1000):
    batch_xs, batch_ys = next_batch(50, training_set)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    if i % 50 == 0:
        print("Accuracy:", compute_accuracy(test_set.data, fix_targetData(test_set)))
        print ("Loss:", sess.run(cross_entropy, feed_dict={xs:batch_xs, ys:batch_ys}))
        print("Weights:", sess.run(Weights))
        print("\n")
        print("Biases:", sess.run(biases))
        print("-------------------------------------------------\n")

print("Final Weights:", sess.run(Weights))
print("\n")
print("Final Biases:", sess.run(biases))
print("-------------------------------------------------\n")
        


def get_index(ds):
    tmp = -1
    count = 0
    result = 0
    for d in ds:
        if (tmp<d):
            tmp = d
            result = count
        count = count + 1    
    return result
    
def print_confusion_matrix(ds):
    confusion_matrix = np.zeros((y_dimension, y_dimension))
    y_pre = sess.run(prediction, feed_dict={xs:ds.data})
    for i in range(len(ds.data)):
        row = ds.target[i]
        col = get_index(y_pre[i])
        value = confusion_matrix[row, col]
        confusion_matrix[row, col] = value+1
    
    print("Confusion Matrix:")
    print('{:3}'.format(' '), end='')
    for i in range(y_dimension):
        print('{:3}'.format(i), end='')
    print("")    
    for i in range(y_dimension):
        print('{:3}'.format(i), end='')
        for j in range(y_dimension):
            print('{:3}'.format(int(confusion_matrix[i, j])), end='')
        print("")
            
    #print(confusion_matrix)
    
print_confusion_matrix(test_set)          
    

    
xx = np.array([ [0, 0], 
                [0, 1],
                [1, 0],
                [1, 1]])
y_pre = sess.run(prediction, feed_dict={xs:xx})

for i in range(len(y_pre.data)):
    print(xx[i], "=>", get_index(y_pre[i]), y_pre[i])
    
