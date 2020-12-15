import tensorflow as tf
import numpy as np

def log10(x):
    return tf.experimental.numpy.log10(x)

def cartesian_to_spherical_coordinates(point_cartesian, eps=None):
    """Function to transform Cartesian coordinates to spherical coordinates.
    This function assumes a right handed coordinate system with `z` pointing up.
    When `x` and `y` are both `0`, the function outputs `0` for `phi`. Note that
    the function is not smooth when `x = y = 0`.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      point_cartesian: A tensor of shape `[A1, ..., An, 3]`. In the last
        dimension, the data follows the `x`, `y`, `z` order.
      eps: A small `float`, to be added to the denominator. If left as `None`,
        its value is automatically selected using `point_cartesian.dtype`.
      name: A name for this op. Defaults to `cartesian_to_spherical_coordinates`.
    Returns:
      A tensor of shape `[A1, ..., An, 3]`. The last dimensions contains
      (`r`,`theta`,`phi`), where `r` is the sphere radius, `theta` is the polar
      angle and `phi` is the azimuthal angle.
    """
    #with tf.compat.v1.name_scope(name, "cartesian_to_spherical_coordinates",
    #                             [point_cartesian]):
    #  point_cartesian = tf.convert_to_tensor(value=point_cartesian)

    #shape.check_static(
    #    tensor=point_cartesian,
    #    tensor_name="point_cartesian",
    #    has_dim_equals=(-1, 3))

    x, y, z = tf.unstack(point_cartesian, axis=-1)
    radius = tf.norm(tensor=point_cartesian, axis=-1)
    theta = tf.acos(
        tf.clip_by_value(tf.divide(z, radius), -1., 1.))
    phi = tf.atan2(y, x)
    return tf.stack((log10(radius), theta, phi), axis=-1)

def spherical_to_cartesian_coordinates(point_spherical, name=None):
    """Function to transform Cartesian coordinates to spherical coordinates.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      point_spherical: A tensor of shape `[A1, ..., An, 3]`. The last dimension
        contains r, theta, and phi that respectively correspond to the radius,
        polar angle and azimuthal angle; r must be non-negative.
      name: A name for this op. Defaults to 'spherical_to_cartesian_coordinates'.
    Raises:
      tf.errors.InvalidArgumentError: If r, theta or phi contains out of range
      data.
    Returns:
      A tensor of shape `[A1, ..., An, 3]`, where the last dimension contains the
      cartesian coordinates in x,y,z order.
    """

    logr, theta, phi = tf.unstack(point_spherical, axis=-1)
    r = tf.pow(10., logr)
    #r = asserts.assert_all_above(r, 0)
    tmp = r * tf.sin(theta)
    x = tmp * tf.cos(phi)
    y = tmp * tf.sin(phi)
    z = r * tf.cos(theta)
    return tf.stack((x, y, z), axis=-1)

def reshape_senders_receivers(senders, receivers, batch_size, nplanets, nedges):
    ''' Reshape receivers and senders to use in graph'''
    x = np.arange(batch_size)
    xx = x.reshape(batch_size,1)
    y = np.ones(nedges)
    z = np.reshape(xx+y-1, batch_size*nedges)*nplanets

    senders = np.concatenate([senders]*batch_size) + z
    receivers = np.concatenate([receivers]*batch_size) + z
    
    return senders, receivers

def build_rotation_matrix(a,b,g):
    A0 = tf.stack([tf.cos(a)*tf.cos(b), tf.sin(a)*tf.cos(b), -tf.sin(b)], 
                  axis=-1)
    A1 = tf.stack([tf.cos(a)*tf.sin(b)*tf.sin(g)-tf.sin(a)*tf.cos(g), 
                   tf.sin(a)*tf.sin(b)*tf.sin(g)+tf.cos(a)*tf.cos(g),
                   tf.cos(b)*tf.sin(g)], axis=-1)
    A2 = tf.stack([tf.cos(a)*tf.sin(b)*tf.cos(g)+tf.sin(a)*tf.sin(g), 
                   tf.sin(a)*tf.sin(b)*tf.cos(g)-tf.cos(a)*tf.sin(g),
                   tf.cos(b)*tf.cos(g)], axis=-1)
    
    return tf.stack((A0, A1, A2), axis=1)

def rotate_data(D, A, uniform = True):
    if uniform:
        n=1
    else:  
        n = D.shape[0]

    # I think the maxes should be 2pi, pi, pi, but going for overkill just in case
    alpha = tf.random.uniform([n,], minval=0, maxval=2*np.pi, dtype=tf.dtypes.float32)
    beta = tf.random.uniform([n,], minval=0, maxval=2*np.pi, dtype=tf.dtypes.float32)
    gamma = tf.random.uniform([n,], minval=0, maxval=2*np.pi, dtype=tf.dtypes.float32)
    R = build_rotation_matrix(alpha,beta,gamma)
    
    if uniform:
        # Rotate all points by the same angle
        D = tf.linalg.matmul(D,R)
        A = tf.linalg.matmul(A,R)
    else: 
        # Rotate each time step by a different angle:
        D = tf.einsum('nij,njk->nik', D,R)
        A = tf.einsum('nij,njk->nik', A,R)
        
    return D, A