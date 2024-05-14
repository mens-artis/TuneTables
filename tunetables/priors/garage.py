    # validate the scaler
    assert scaler in ["None", "Quantile"], f"scaler not recognized: {scaler}"

    if scaler == "Quantile":
        scaler_function = QuantileTransformer(
            n_quantiles=min(len(train_index), 1000)
        )  # use either 1000 quantiles or num. training instances, whichever is smaller


    if scaler != "None":
        if verbose:
            print(f"Scaling the data using {scaler}...")
        X_train[:, num_mask] = scaler_function.fit_transform(X_train[:, num_mask])
        X_val[:, num_mask] = scaler_function.transform(X_val[:, num_mask])
        X_test[:, num_mask] = scaler_function.transform(X_test[:, num_mask])

#####################

    if scaler == "Quantile":
        scaler_function = QuantileTransformer(
            n_quantiles=min(len(train_index), 1000)
        )  # use either 1000 quantiles or num. training instances, whichever is smaller
