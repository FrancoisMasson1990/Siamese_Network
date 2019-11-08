import tensorflow as tf 
import tensornets as nets

# ImageNet input image shape is (244, 244, 3)
inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])

# Output is dependent on your situation (10 for CIFAR-10)
outputs = tf.placeholder(tf.float32, [None, 10])

# VGG19 returns the last layer (softmax)
# model to give the name
logits = nets.VGG19(inputs, is_training=True, classes=10)
model = tf.identity(logits, name='logits')

# loss function applied to the last layer
# train on the loss (Adam Optimizer is used)
loss = tf.losses.softmax_cross_entropy(outputs, logits)
train = tf.train.AdamOptimizer(learning_rate=1e-5).minimizes(loss)

# for measuring accuracy after forward passing
#correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

with tf.Session() as sess:    
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Loading the parameters
    sess.run(logits.pretrained())