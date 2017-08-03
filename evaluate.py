




for i in range(batch_num):
    [X_test, y_test, _, _], _  = next(generator)
    y_pred = base_model.predict(X_test)
    shape = y_pred[:,2:,:].shape
    ctc_decode = K.ctc_decode(y_pred[:,2:,:],
                              input_length=np.ones(shape[0])*shape[1])[0][0]
    out = K.get_value(ctc_decode)[:, :n_len]
    if out.shape[1] == n_len:
        batch_acc += ((y_test == out).sum(axis=1) == n_len).mean()
    batch_acc / batch_num