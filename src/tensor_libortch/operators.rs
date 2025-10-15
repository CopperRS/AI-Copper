use crate::tensor_libortch::tensor::Tensor;
use std::ops::{Add, Sub, Mul, Div};

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Cannot add tensors with different dimensions!");
        }

        let result_values: Vec<f32> = self
            .as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(a, b)| a + b)
            .collect();

        Tensor::from_values(&result_values, self.rows, self.cols)
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Cannot subtract tensors with different dimensions!");
        }

        let result_values: Vec<f32> = self
            .as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(a, b)| a - b)
            .collect();

        Tensor::from_values(&result_values, self.rows, self.cols)
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Cannot multiply tensors with different dimensions!");
        }

        let result_values: Vec<f32> = self
            .as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(a, b)| a * b)
            .collect();

        Tensor::from_values(&result_values, self.rows, self.cols)
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Tensor {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Cannot divide tensors with different dimensions!");
        }

        let result_values: Vec<f32> = self
            .as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(a, b)| {
                if *b == 0.0 {
                    panic!("Division by zero in tensor!");
                }
                a / b
            })
            .collect();

        Tensor::from_values(&result_values, self.rows, self.cols)
    }
}