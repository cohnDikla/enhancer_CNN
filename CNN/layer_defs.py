import tensorflow as tf
import numpy as np
import math
import os
import scipy.stats as stats

"""
A module for creating definitions of the CNN layers.
"""

bases_map = {"A": 0, "C": 1, "G": 2, "T": 3}

BASIS_NUMBER = 4


def read_PWM_from_file(PWM_path):
    print("in read_PWM_from_file, PWM_path = ",PWM_path)
    frequencies_all_positions = []
    with open(PWM_path) as pwm:
        position = 0
        for line in pwm:
            frequencies_one_position = []
            # frequency_matrix = dict()
            split_line = line.split()
            # print("split_line = ", split_line)
            for base in ["A","C","G","T"]:
                base_index = bases_map[base]
                frequencies_one_position.append(float(split_line[base_index]))

            frequencies_all_positions.append(frequencies_one_position)
            position += 1

    # print("len(frequencies_all_positions[0]) = ",len(frequencies_all_positions[0]))
    array_frequencies_all_positions = np.array(frequencies_all_positions)
    print("array_frequencies_all_positions.shape = ", array_frequencies_all_positions.shape)
    reshape_array = np.transpose(array_frequencies_all_positions)
    print("reshape_array = ", reshape_array)

    # print ("frequencies_all_positions = ", frequencies_all_positions)
    return frequencies_all_positions


def get_random_values(layer_num, project, num_kernels_one_k):
    my_stddev = math.sqrt(2)
    print("my_stddev = ", my_stddev)
    mu, sigma = 0, my_stddev
    # The generated values follow a normal distribution
    # with specified mean and standard deviation,
    # except that values whose magnitude is more than 2 standard deviations
    # from the mean are dropped and re-picked.
    lower = mu - (2 * sigma)
    upper = mu + (2 * sigma)
    num_filters_in_layer = num_kernels_one_k
    filter_shape = project.CNN_structure.conv_shapes[layer_num - 1]
    if layer_num == 1:
        first_dimension = filter_shape[0]
        second_dimension = filter_shape[1]
    else:
        first_dimension = filter_shape[1]
        second_dimension = project.CNN_structure.conv_num_kernels[layer_num - 2]
    all_filters_in_layer = []
    for i in range(1, num_filters_in_layer + 1):
        one_filter_values = []
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        for f in range(first_dimension):
            # generate <second_dimension> data samples
            random_values = X.rvs(second_dimension)
            one_filter_values.append(random_values)

        all_filters_in_layer.append(one_filter_values)
    return all_filters_in_layer




def read_all_filters_in_layer(layer_num, model_id, project, num_kernels_one_k):
    if project.project_name == "H3K27ac_vs_negative_data_planted_filters":
        project_to_take_filters_from = "H3K27ac_vs_k_shuffle"
    layer_dir = os.path.join(project.all_projects_base_bath, project_to_take_filters_from,
                             "output", "CNN", model_id, "convolution_results",
                             "filters", "layer_"+str(layer_num))
    k_index = model_id.split("_").index("k")
    # print("k_index = ", k_index)
    species_name = "_".join(model_id.split("_")[:k_index])
    print("species_name = ", species_name)
    num_filters_in_layer = num_kernels_one_k
    filter_shape = project.CNN_structure.conv_shapes[layer_num - 1]
    if layer_num == 1:
        first_dimension = filter_shape[0]
        second_dimension = filter_shape[1]
    else:
        first_dimension = filter_shape[1]
        second_dimension = project.CNN_structure.conv_num_kernels[layer_num-2]
    all_filters_in_layer = []
    for i in range(1, num_filters_in_layer+1):
        filter_path = os.path.join(layer_dir, "filter"+str(i)+"_"+species_name+".txt")
        with open(filter_path) as filter_file:
            one_filter_values = []
            line_counter = 0
            for line in filter_file:
                one_line_values = []
                line_counter += 1
                if line_counter > first_dimension:
                    print("error, line_counter > first_dimension")
                    exit()
                split_line = line.split()
                if len(split_line) != second_dimension:
                    print("error, len(split_line) != second_dimension")
                    exit()
                for j in range(second_dimension):
                    one_line_values.append(float(split_line[j]))
                one_filter_values.append(one_line_values)
        all_filters_in_layer.append(one_filter_values)

    # array_all_filters = np.array(all_filters_in_layer)
    return all_filters_in_layer





def affine(name_scope, input_tensor, out_channels, relu=True):
    """
    Returns an affine transformation layer
    """
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    with tf.name_scope(name_scope):
        # The truncated_normal function outputs random values from a truncated normal distribution.
		# The generated values follow a normal distribution with specified mean and standard deviation,
		# except that values whose magnitude is more than 2 standard deviations from the mean are dropped
		# and re-picked.
        weights = tf.Variable(
            tf.truncated_normal([input_channels, out_channels],
                                stddev=1.0 / math.sqrt(float(input_channels))), name='weights')
        initial = tf.constant(0.1, shape=[out_channels])
        biases = tf.Variable(initial, name='biases')
        # tf.histogram_summary(name_scope + "_weights", weights)
        # tf.histogram_summary(name_scope + "_biases", biases)
        # w_image = tf.reshape(weights, [-1] + [input_channels, out_channels] +[1])
        # tf.image_summary(name_scope + "_weights_im", w_image)
        if relu:
            return tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        else:
            return tf.matmul(input_tensor, weights) + biases


def conv_max_forward_reverse(name_scope, input_tensor, num_kernels, kernel_size, project,
                             PWM_file=None, weights_are_constant=False, stride=1,
                             padding='VALID', relu=True, init_according_to_given_filters=False,
                             init_model_ids=None):
    """
    Returns a convolution layer
    """
    # conv2 = tfb.conv("conv2", max_pool1, self.conv2_num_kernels, self.conv2_shape)
    print ("********************************* \n "
           "in layer_defs - conv_max_forward_reverse: \n"
           "name_scope = ", name_scope, "input_tensor = ", input_tensor,
           "out_channels = ", num_kernels, "ksize = ", kernel_size, "stride = ", stride, "padding = ", padding,
           "relu = ", relu, " , PWM_file = ",PWM_file)
    input_shape = input_tensor.get_shape().as_list()
    print("input_shape = ", input_shape)
    input_channels = input_shape[-1] # number of input channels
    print("input_channels = ", input_channels)
    with tf.name_scope(name_scope):
        if PWM_file:
            print("PWM_file is not None")
            PWM = read_PWM_from_file(PWM_file)
            motif_length = len(PWM)
            print("motif length = ", motif_length)
            if weights_are_constant:
                # in this case - use Constant weights! not Variable, so they cannot be changed at all
                print("*********************** weights are constant!!!! ****************************")
                const_filter = tf.constant(PWM, tf.float32)
                weights = tf.reshape(const_filter, [BASIS_NUMBER, motif_length, 1, 1], name='weights')

            else:
                print("*********************** weights are Variable with constant init!!!! ****************************")
                const_filter = tf.constant(PWM, tf.float32)
                weights = tf.Variable(tf.reshape(const_filter, [BASIS_NUMBER, motif_length, 1, 1]), name='weights')
        elif init_according_to_given_filters:
            all_filters_all_ks = []
            num_ks = len([2, 3, 4])
            num_kernels_one_k = int(project.CNN_structure.conv_num_kernels[0] / (num_ks + 1))
            for k in range(num_ks+1):
                if k == num_ks:  # random initialization
                    all_filters_one_k = get_random_values(1, project, num_kernels_one_k)
                else:
                    model_id = init_model_ids[k]
                    all_filters_one_k = read_all_filters_in_layer(1, model_id, project, num_kernels_one_k)
                all_filters_all_ks.extend(all_filters_one_k)


            array_all_filters_all_ks = np.array(all_filters_all_ks)
            print("array_all_filters.shape: ", array_all_filters_all_ks.shape)

            num_filters = array_all_filters_all_ks.shape[0]  # 120
            # since it is the first layer:
            first_filter_dimension = kernel_size[0]
            second_filter_dimension = kernel_size[1]

            print("***** weights are Variable with constant init!!!! *********")
            const_filter = tf.constant(array_all_filters_all_ks, tf.float32)
            weights = tf.Variable(tf.reshape(const_filter,
                                             [first_filter_dimension, second_filter_dimension, 1, num_filters]),
                                  name='weights')

        else:
            print("else: PWM_file = None")
            shape = kernel_size + [input_channels, num_kernels]
            print("shape = ", shape)
            my_stddev = math.sqrt(2 / float(input_channels))
            print("my_stddev = ", my_stddev)
            initer = tf.truncated_normal(shape, stddev=math.sqrt(2 / float(input_channels)))
            weights = tf.Variable(initer, name='weights')

        # print ("weights.get_shape()[3] = ",weights.get_shape()[3])
        num_kernels = weights.get_shape()[3]
        print("num_kernels = ",num_kernels)
        print("weights.get_shape() = ", weights.get_shape())

        biases = tf.Variable(tf.zeros([num_kernels]), name='biases')
        print("biases.get_shape() = ", biases.get_shape())

        # If one component of shape is the special value -1, the size of that dimension is computed
        #  so that the total size remains constant.
        # In our case: -1 is inferred to be input_channels * out_channels:

        new_weights_shape = [-1] + kernel_size + [1]
        print("new_weights_shape = ", new_weights_shape)

        # TODO uncomment:
        w_image = tf.reshape(weights, new_weights_shape) # TODO ????
        print("w_image = ", w_image)
        print("w_image.get_shape() = ", w_image.get_shape())  # (input_channels*out_channels, 1, 2, 1)

        tf.summary.image(name_scope + "_weights_im", w_image, weights.get_shape()[3])
        # max_images_number = 1000000
        # tf.image_summary(name_scope + "_weights_im", w_image, max_images_number)
        ###############################

        # tf.histogram_summary(name_scope + "_weights", weights)
        # tf.histogram_summary(name_scope + "_biases", biases)

        forward_conv = tf.nn.conv2d(input_tensor, weights, strides=[1, stride, stride, 1], padding=padding,
                               name="forward_conv") + biases

        # for reverse complement: reverse in dimension 0 and 1:
        rev_comp_weights = tf.reverse(weights, [0, 1], name="reverse_weights")
        reverse_conv = tf.nn.conv2d(input_tensor, rev_comp_weights,
                                    strides=[1, stride, stride, 1], padding=padding,
                                    name="reverse_conv") + biases

        # takes the maximum between the forward weights and the rev.-comp.-weights:
        max_conv = tf.maximum(forward_conv, reverse_conv, name="conv1")

        if relu:
            return tf.nn.relu(max_conv, name="relu_conv1")
        else:
            return max_conv


def conv(name_scope, input_tensor, num_kernels, kernel_size, project, PWM_file=None,
         weights_are_constant=False, stride=1, padding='VALID', relu=True,
         init_according_to_given_filters=False, init_model_ids=None):
    """
    Returns a convolutions layer
    """
    # conv2 = tfb.conv("conv2", max_pool1, self.conv2_num_kernels, self.conv2_shape)
    print("********************************* \n "
          "in  layer_defs - conv: \n"
          "name_scope = ", name_scope, "input_tensor = ", input_tensor,
          "out_channels = ", num_kernels, "ksize = ", kernel_size, "stride = ", stride, "padding = ", padding,
          "relu = ", relu, " , PWM_file = ", PWM_file)
    input_shape = input_tensor.get_shape().as_list()
    print("input_shape = ", input_shape)
    input_channels = input_shape[-1]  # number of input channels
    print("input_channels = ", input_channels)
    conv_layer_num = int(name_scope[-1])
    with tf.name_scope(name_scope):
        if PWM_file:
            print("PWM_file is not None")
            PWM = read_PWM_from_file(PWM_file)
            motif_length = len(PWM)
            print("motif length = ", motif_length)
            if weights_are_constant:
                # in this case - use Constant weights! not Variable, so they cannot be changed at all
                print("*********************** weights are constant!!!! ****************************")
                const_filter = tf.constant(PWM, tf.float32)
                weights = tf.reshape(const_filter, [BASIS_NUMBER, motif_length, 1, 1], name='weights')

            else:
                print(
                    "*********************** weights are Variable with constant init!!!! ****************************")
                const_filter = tf.constant(PWM, tf.float32)
                weights = tf.Variable(tf.reshape(const_filter, [BASIS_NUMBER, motif_length, 1, 1]), name='weights')

        elif init_according_to_given_filters:
            all_filters_all_ks = []
            num_ks = len([2, 3, 4])
            num_kernels_one_k = int(project.CNN_structure.conv_num_kernels[0] / (num_ks + 1))
            for k in range(num_ks+1):
                if k == num_ks:  # random initialization
                    all_filters_one_k = get_random_values(conv_layer_num, project, num_kernels_one_k)
                else:
                    model_id = init_model_ids[k]
                    all_filters_one_k = read_all_filters_in_layer(conv_layer_num, model_id, project, num_kernels_one_k)
                all_filters_all_ks.extend(all_filters_one_k)
            array_all_filters_all_ks = np.array(all_filters_all_ks)
            print("array_all_filters.shape: ", array_all_filters_all_ks.shape)
            num_filters = array_all_filters_all_ks.shape[0]
            # since its not the first layer:
            first_filter_dimension = kernel_size[1]
            second_filter_dimension = project.CNN_structure.conv_num_kernels[conv_layer_num - 2]

            print("***** weights are Variable with constant init!!!! *********")
            const_filter = tf.constant(array_all_filters_all_ks, tf.float32)
            weights = tf.Variable(tf.reshape(const_filter,
                                             [first_filter_dimension, second_filter_dimension, 1, num_filters]),
                                  name='weights')
        else:
            print("else: PWM_file = None")
            shape = kernel_size + [input_channels, num_kernels]
            print("shape = ", shape)
            my_stddev = math.sqrt(2 / float(input_channels))
            print("my_stddev = ", my_stddev)
            initer = tf.truncated_normal(shape, stddev=math.sqrt(2 / float(input_channels)))
            weights = tf.Variable(initer, name='weights')

        # print ("weights.get_shape()[3] = ",weights.get_shape()[3])
        num_kernels = weights.get_shape()[3]
        print("num_kernels = ", num_kernels)
        print("weights.get_shape() = ", weights.get_shape())

        biases = tf.Variable(tf.zeros([num_kernels]), name='biases')
        print("biases.get_shape() = ", biases.get_shape())

        # If one component of shape is the special value -1, the size of that dimension is computed
        #  so that the total size remains constant.
        # In our case: -1 is inferred to be input_channels * out_channels:

        new_weights_shape = [-1] + kernel_size + [1]
        print("new_weights_shape = ", new_weights_shape)

        # TODO uncomment:
        w_image = tf.reshape(weights, new_weights_shape)  # TODO ????
        print("w_image = ", w_image)
        print("w_image.get_shape() = ", w_image.get_shape())  # (input_channels*out_channels, 1, 2, 1)

        tf.summary.image(name_scope + "_weights_im", w_image, weights.get_shape()[3])
        # max_images_number = 1000000
        # tf.image_summary(name_scope + "_weights_im", w_image, max_images_number)
        ###############################

        # tf.histogram_summary(name_scope + "_weights", weights)
        # tf.histogram_summary(name_scope + "_biases", biases)

        c = tf.nn.conv2d(input_tensor, weights, strides=[1, stride, stride, 1],
                         padding=padding) + biases

        # filtered_x = tf.nn.conv2d(image_resized, sobel_x_filter,
        #                           strides=[1, 1, 1, 1], padding='SAME')

        if relu:
            return tf.nn.relu(c, name="relu_"+name_scope)
        else:
            return c


def flatten(x):
    """
    Returns a flat (one-dimensional) version of the input
    """
    x_shape = x.get_shape().as_list()
    return tf.reshape(x, [-1, np.product(x_shape[1:])])

# def read_filter_from_file(filter_path):
#     print("in read_filter_from_file, filter_path = ", filter_path)
#     values_all_dimensions = []
#     with open(filter_path) as pwm:
#
#         for line in pwm:
#             values_one_dimension = []
#             split_line = line.split()
#             # print("split_line = ", split_line)
#             for value in split_line:
#                 values_one_dimension.append(float(value))
#             values_all_dimensions.append(values_one_dimension)
#
#     array_values_all_dimensions = np.array(values_all_dimensions, dtype='float32')
#     print("array_values_all_dimensions.shape = ", array_values_all_dimensions.shape)
#     # reshape_array = np.transpose(array_values_all_dimensions)
#     # print("reshape_array.shape = ", reshape_array.shape)
#     # print("reshape_array = ", reshape_array)
#
#     # print ("frequencies_all_positions = ", frequencies_all_positions)
#     return array_values_all_dimensions
#
#
# def conv_max_forward_reverse_read_filter_from_file(name_scope, input_tensor, num_kernels,
#                                                    kernel_size, filter_path,
#                                                    stride=1, padding='VALID', relu=True):
#     """
#     Returns a convolution layer
#     """
#     # conv2 = tfb.conv("conv2", max_pool1, self.conv2_num_kernels, self.conv2_shape)
#     print("********************************* \n "
#           "in  layer_defs - conv_max_forward_reverse_read_filter_from_file: \n"
#           "name_scope = ", name_scope, " , input_tensor = ", input_tensor,
#           "out_channels = ", num_kernels, "ksize = ", kernel_size, "stride = ", stride, "padding = ", padding,
#           "relu = ", relu, " , filter_path = ", filter_path)
#     input_shape = input_tensor.get_shape().as_list()
#     print("input_shape = ", input_shape)
#     input_channels = input_shape[-1]  # number of input channels
#     print("input_channels = ", input_channels)
#     with tf.name_scope(name_scope):
#
#         filter = read_filter_from_file(filter_path)
#         print("type input_tensor[0][0] = ", type(input_tensor[0][0][0][0]))
#         filter_height = len(filter)
#         print("filter_height = ", filter_height)
#         filer_width = len(filter[0])
#         print("filer_width = ", filer_width)
#
#         const_filter = tf.constant(filter, tf.float32)
#         weights = tf.reshape(const_filter, [filter_height, filer_width, 1, 1], name='weights')
#
#         # weights = tf.Variable(tf.reshape(const_filter, [BASIS_NUMBER, motif_length, 1, 1]), name='weights')
#
#         # print ("weights.get_shape()[3] = ",weights.get_shape()[3])
#         num_kernels = weights.get_shape()[3]
#         print("num_kernels = ", num_kernels)
#         print("weights.get_shape() = ", weights.get_shape())
#
#         # TODO add biases! (?)
#         # biases = tf.Variable(tf.zeros([num_kernels], tf.float32), name='biases')
#         # print("biases.get_shape() = ", biases.get_shape())
#
#         # If one component of shape is the special value -1, the size of that dimension is computed
#         #  so that the total size remains constant.
#         # In our case: -1 is inferred to be input_channels * out_channels:
#         new_weights_shape = [-1] + kernel_size + [1]
#         print("new_weights_shape = ", new_weights_shape)
#
#         # TODO uncomment:
#         # w_image = tf.reshape(weights, new_weights_shape)  # TODO ????
#         # print("w_image = ", w_image)
#         # print("w_image.get_shape() = ", w_image.get_shape())  # (input_channels*out_channels, 1, 2, 1)
#
#         # tf.summary.image(name_scope + "_weights_im", w_image, weights.get_shape()[3])
#         # max_images_number = 1000000
#         # tf.image_summary(name_scope + "_weights_im", w_image, max_images_number)
#         ###############################
#
#         # tf.histogram_summary(name_scope + "_weights", weights)
#         # tf.histogram_summary(name_scope + "_biases", biases)
#
#         # TODO add biases! (?)
#         forward_conv = tf.nn.conv2d(input_tensor, weights, strides=[1, stride, stride, 1], padding=padding,
#                                     name="forward_conv")
#         # forward_conv = tf.nn.conv2d(input_tensor, weights, strides=[1, stride, stride, 1], padding=padding,
#         #                             name="forward_conv") + biases
#
#         print("after forward_conv")
#
#         # for reverse complement: reverse in dimension 0 and 1:
#         rev_comp_weights = tf.reverse(weights, [0, 1], name="reverse_weights")
#         # TODO add biases! (?)
#         reverse_conv = tf.nn.conv2d(input_tensor, rev_comp_weights,
#                                     strides=[1, stride, stride, 1], padding=padding,
#                                     name="reverse_conv")
#
#         # reverse_conv = tf.nn.conv2d(input_tensor, rev_comp_weights,
#         #                             strides=[1, stride, stride, 1], padding=padding,
#         #                             name="reverse_conv") + biases
#
#         print("after reverse_conv")
#
#
#         # takes the maximum between the forward weights and the rev.-comp.-weights:
#         max_conv = tf.maximum(forward_conv, reverse_conv, name="conv1")
#
#         if relu:
#             return tf.nn.relu(max_conv, name="conv1")
#         else:
#             return max_conv


