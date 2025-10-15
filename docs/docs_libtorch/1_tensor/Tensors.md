# Creating tensors and printing

`Tensor::rand`
```
    let tensor:Tensor = Tensor::rand(3,3);

    tensor.print();
```
`Output`
```
    Variable[CPUFloatType {3, 3}]
    0.7353 0.1235 0.9877
    0.4568 0.3217 0.6543
    0.1473 0.8527 0.3693
```

***

`Tensor::ones`
```
    let tensor:Tensor = Tensor::ones(3,3);

    tensor.print(); 
```

`Output`

```
    Variable[CPUFloatType {3, 3}]
    1.0000 1.0000 1.0000
    1.0000 1.0000 1.0000
    1.0000 1.0000 1.0000
```

***

`Tensor::from_values`
```
    let values: [f32; 9] = [
        0.1111, 0.2222, 0.3333,
        0.4444, 0.5555, 0.6666,
        0.7777, 0.8888, 0.9999
    ];

    let tensor = Tensor::from_values(&values, 3, 3);

    tensor.print();
```

`Output`

```
    Variable[CPUFloatType {3, 3}]
    0.1111 0.2222 0.3333
    0.4444 0.5555 0.6666
    0.7777 0.8888 0.9999
```

<hr style="height:4px; background-color:#333; border:none;">

`Small Neural Network with Linear and Optimizer`
```
    // Create an input tensor (2 samples, 3 features)
    let input_values: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let input = Tensor::from_values(&input_values, 2, 3);

    // Create a target tensor (2 samples, 2 features)
    let target_values: [f32; 4] = [0.5, 0.6, 0.7, 0.8];
    let target = Tensor::from_values(&target_values, 2, 2);

    // Create a linear layer (3 input features, 2 output features)
    let linear = Linear::new(3, 2);

    // Create an SGD optimizer with learning rate 0.01
    let optimizer = Optimizer::sgd(&linear, 0.01);

    // Train for 10 epochs
    for epoch in 0..10 {
        // Forward pass
        let output = linear.forward(&input);

        // Compute MSE loss
        let loss = output.mse_loss(&target);

        // Print loss
        println!("Epoch {}:", epoch);
        loss.print();

        // Compute gradients
        loss.backward();

        // Update parameters
        optimizer.step();
    }
```
`Output before training (epoch 0)`
```
    Output at Epoch 0:
    Variable[CPUFloatType {2, 2}]
    0.2200 0.2800 
    0.4900 0.6400 

    Loss at Epoch 0:
    Variable[CPUFloatType {1, 1}]
    0.0626 
```
`Output after Training (Epoch 9)`
```
    Output at Epoch 9:
    Variable[CPUFloatType {2, 2}]
    0.4500 0.5500 
    0.6500 0.7500 
    Loss at Epoch 9:
    Variable[CPUFloatType {1, 1}]
    0.0025
```