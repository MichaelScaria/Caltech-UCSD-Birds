####################################### PART ONE #######################################
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
import util

NUM_CLASSES = 150

IMAGES_MEAN = 122.5
IMAGES_STD = 63.32

LOGITS_COLLECTION = 'LOGITS'
LOGIT_LABELS_COLLECTION = 'LOGIT-LABELS'

RUN_PREFIX = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).strftime('%Y-%m-%d-%H_%M_%S')

PLOT_FILE_NAME = '{}_TRAIN-{}.txt'.format(RUN_PREFIX, NUM_CLASSES)
TRAIN_FILE_NAME= '{}_PLOT-{}.csv'.format(RUN_PREFIX, NUM_CLASSES)

file = open(PLOT_FILE_NAME, 'a')
file.write('==============================')
file.close()

file = open(TRAIN_FILE_NAME, 'a')
file.write('==============================')
file.close()

####################################### PART TWO #######################################

def load(filename):
    file = open(filename, "r") 
    image_names = file.readlines()
    images = []
    labels = []
    for name in image_names:
        label = int(name[:3])
        if label <= NUM_CLASSES:
            im = Image.open("images/" + name.rstrip('\n'))
            H, W = im.size
            pixels = list(im.getdata())
            if not type(pixels[0]) is int:
                # todo: right now we are discarding transparent images
                image = np.array([comp for pixel in pixels for comp in pixel]).reshape(-1, H, W, 3)
                images.append(image)
                # zero-index the label
                labels.append(label - 1)
        else: 
            break
    return images, labels

images_train_and_val, labels_train_and_val = load('train.txt')

seed = 13958293
np.random.seed(seed)
np.random.shuffle(images_train_and_val)
np.random.seed(seed)
np.random.shuffle(labels_train_and_val)


images_train = images_train_and_val[:int(len(images_train_and_val) * .80)]
images_val = images_train_and_val[int(len(images_train_and_val) * .80):]

labels_train = labels_train_and_val[:int(len(labels_train_and_val) * .80)]
labels_val = labels_train_and_val[int(len(labels_train_and_val) * .80):]


print(len(images_train))
print(len(images_val))


images_test, labels_test = load('test.txt')

print(len(images_test))


####################################### PART THREE #######################################

BATCH_SIZE = 32
EPOCHS = 200

# Modified from RikHeijdens on https://github.com/tensorflow/tensorflow/issues/6011
def spp_layer(image, dimensions=[6, 3, 2, 1]):
    # todo: fix this
    if tf.less(tf.shape(image)[1], dimensions[0] ** 2) is True:
        return None
    if tf.less(tf.shape(image)[2], dimensions[0] ** 2) is True:
        return None
    pool_list = []
    for pool_dim in dimensions:
        pool_list += max_pool_2d_nxn_regions(image, pool_dim)
    return tf.concat(pool_list, axis=1)

def max_pool_2d_nxn_regions(inputs, output_size):
    inputs_shape = tf.shape(inputs)
    h = tf.cast(tf.gather(inputs_shape, 1), tf.int32)
    w = tf.cast(tf.gather(inputs_shape, 2), tf.int32)

    result = []
    n = output_size
    for row in range(output_size):
        for col in range(output_size):
            # start_h = floor(row / n * h)
            start_h = tf.cast(tf.floor(tf.multiply(row / n, tf.cast(h, tf.float32))), tf.int32)
            # end_h = ceil((row + 1) / n * h)
            end_h = tf.cast(tf.ceil(tf.multiply((row + 1) / n, tf.cast(h, tf.float32))), tf.int32)
            # start_w = floor(col / n * w)
            start_w = tf.cast(tf.floor(tf.multiply(col / n, tf.cast(w, tf.float32))), tf.int32)
            # end_w = ceil((col + 1) / n * w)
            end_w = tf.cast(tf.ceil(tf.multiply((col + 1) / n, tf.cast(w, tf.float32))), tf.int32)
            pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
            pool_result = tf.reduce_max(pooling_region, axis=(1, 2))
            result.append(pool_result)
    return result

graph = tf.Graph()
with graph.as_default():

    image_placeholders = []
    label_placeholders = []

    with tf.variable_scope("network") as scope:
        training = tf.placeholder_with_default(False, (), name='training')
        conv_reuse = None
        for i in range(BATCH_SIZE):
            # todo: we can add transparent images 
            image = tf.placeholder(tf.float32, shape=(1,None,None,3), name='image_{}'.format(i))
            image_placeholders.append(image)
            label = tf.placeholder(tf.int64, shape=(), name='label_{}'.format(i))
            label_placeholders.append(label)

            logit = tf.to_float(image)
            logit = (logit - IMAGES_MEAN) / IMAGES_STD

            logit = tf.layers.conv2d(logit, 15, [1, 1], padding='SAME', reuse=conv_reuse, name='conv-1')
            logit = tf.layers.conv2d(logit, 25, [4, 4], padding='SAME', reuse=conv_reuse, name='conv-2')
            logit = tf.contrib.layers.max_pool2d(inputs=logit, kernel_size=[2, 2], stride=2, scope='pool-1')
            logit = tf.layers.conv2d(logit, 20, [2, 2], padding='SAME', reuse=conv_reuse, name='conv-3')

            logit = spp_layer(logit)
            conv_reuse = True
            if not logit is None:
                logit = tf.reshape(logit, [-1])
                tf.add_to_collection(LOGITS_COLLECTION, tf.identity(logit, name='coll_logit_{}'.format(i)))
                tf.add_to_collection(LOGIT_LABELS_COLLECTION, tf.identity(label, name='coll_label_{}'.format(i)))

            scope.reuse_variables()
        
    
    logits = tf.stack(tf.get_collection(LOGITS_COLLECTION))
    logit_labels = tf.stack(tf.get_collection(LOGIT_LABELS_COLLECTION))
    print(logits.shape)

    logits = tf.layers.dropout(logits, rate=0.3)
    logits = tf.contrib.layers.fully_connected(logits, NUM_CLASSES, activation_fn=None, scope="fc-1")

    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=logit_labels)) + 1e-6 * tf.losses.get_regularization_loss()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        opt = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
    correct = tf.equal(tf.argmax(logits, -1), logit_labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    [print(v.name) for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    print('Number of trainable variables: {}'.format(len(tf.trainable_variables())))
    print('Total number of variables used {}'.format(np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()])))
    
sess = tf.Session(graph=graph)
LOG_DIR = 'log/{}'.format(RUN_PREFIX)
with graph.as_default(), sess.as_default():
    sess.run(tf.global_variables_initializer())
    print('ready to test')


####################################### PART FOUR #######################################
print('starting to test')

PLOT_FREQ = 20
SAVE_FREQ = 50

def save_test_data(epoch):
    with graph.as_default(), sess.as_default():
        val_correct = []
        for i in range(0, len(images_test), BATCH_SIZE):
            batch_images, batch_labels = images_test[i:i + BATCH_SIZE], labels_test[i:i + BATCH_SIZE]
            if BATCH_SIZE - len(batch_images) > 0:
                    for j in range(len(batch_images), BATCH_SIZE):
                        batch_images.append(images_test[j - len(batch_images)])
                        batch_labels.append(labels_test[j - len(batch_images)])
            fd = {**{k: v for k, v in zip(image_placeholders, batch_images)}, **{k: v for k, v in zip(label_placeholders, batch_labels )}}
            val_correct.extend( sess.run(correct, feed_dict=fd) )
        data_to_plot = "{},{},{}".format(NUM_CLASSES, epoch, np.mean(val_correct))
        print(data_to_plot)

        plot_file = open(PLOT_FILE_NAME, 'a')
        plot_file.write('{}\n'.format(data_to_plot))
        plot_file.close()

with graph.as_default(), sess.as_default():
    for epoch in range(EPOCHS):
        np.random.seed(epoch)
        np.random.shuffle(images_train)
        np.random.seed(epoch)
        np.random.shuffle(labels_train)
        accuracy_vals, loss_vals = [], []
        for i in range(0, len(images_train) - BATCH_SIZE + 1, BATCH_SIZE):
            batch_images, batch_labels = images_train[i:i + BATCH_SIZE], labels_train[i:i + BATCH_SIZE]
        
            # todo: this is not very good... (probably replace with 1 x 1 x 1 x 1 when I implement SPP filter, do the same for training)
            if BATCH_SIZE - len(batch_images) > 0:
#                 print('testing diff: %d'%(BATCH_SIZE - len(batch_images)))
                for j in range(len(batch_images), BATCH_SIZE):
                    batch_images.append(images_train[j - len(batch_images)])
                    batch_labels.append(labels_train[j - len(batch_images)])

            fd = {**{k: v for k, v in zip(image_placeholders, batch_images)}, **{k: v for k, v in zip(label_placeholders, batch_labels )}}
            fd[training] = True
            accuracy_val, loss_val, _ = sess.run([accuracy, loss, opt], feed_dict=fd)
            accuracy_vals.append(accuracy_val)
            loss_vals.append(loss_val)
        val_correct = []
        for i in range(0, len(images_val), BATCH_SIZE):
            batch_images, batch_labels = images_val[i:i + BATCH_SIZE], labels_val[i:i + BATCH_SIZE]
            
            if BATCH_SIZE - len(batch_images) > 0:
#                 print('training diff: %d'%(BATCH_SIZE - len(batch_images)))
                for j in range(len(batch_images), BATCH_SIZE):
                    batch_images.append(images_val[j - len(batch_images)])
                    batch_labels.append(labels_val[j - len(batch_images)])
                
            fd = {**{k: v for k, v in zip(image_placeholders, batch_images)}, **{k: v for k, v in zip(label_placeholders, batch_labels )}}
            c = sess.run(correct, feed_dict=fd)
            val_correct.extend(c)
        data_for_train = '[%3d] Accuracy: %0.3f  \t  Loss: %0.3f  \t  validation accuracy: %0.3f'%(epoch, np.mean(accuracy_vals), np.mean(loss_vals), np.mean(val_correct))
        print(data_for_train)
        train_file = open(TRAIN_FILE_NAME, 'a')
        train_file.write('{}\n'.format(data_for_train))
        train_file.close()
        if epoch > 0 and epoch % PLOT_FREQ == 0:
            save_test_data(epoch)
        if epoch > 0 and epoch % SAVE_FREQ == 0:
            util.save('{}-BIRDS-{}-{}.tfg'.format(RUN_PREFIX, NUM_CLASSES, epoch), graph=graph, session=sess)

