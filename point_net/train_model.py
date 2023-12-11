import tensorflow as tf

def custom_loss(ytrue, ypred, gamma=0.5):
    
    ######compute binary cross entropy#####
    loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(ytrue, ypred)
    
    #####compute positive and negative class weights#####
    
    #obtain estimated probability of the actual class
    logits = tf.math.sigmoid(ypred)
    b_ytrue = tf.dtypes.cast(ytrue, dtype = tf.bool)
    m_logits = tf.boolean_mask(logits, b_ytrue)
    m_logits = tf.expand_dims(m_logits, axis=1)
    
    #define positive and negative mask
    neg_msk = 1.0 - ytrue
    pos_msk = 1.0 - neg_msk
    
    '''
    0.05 is the manual defined safety margin
    if logits < m_logits - 0.05 => less correction
    if logits > m_logits - 0.05 => more correction
    if logits = m_logits - 0.05 => same correction
    '''
    neg_w = neg_msk * (1 + (logits - m_logits + 0.05))**gamma
    pos_w = pos_msk
    
    #compute loss
    c_loss = (neg_w + pos_w)*loss_bce

    return tf.reduce_sum(c_loss)