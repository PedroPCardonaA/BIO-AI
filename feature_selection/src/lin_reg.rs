use linfa::traits::{Fit, Predict};
use linfa_linear::{LinearRegression, FittedLinearRegression};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct LinReg;

impl LinReg {
    pub fn new() -> Self {
        LinReg
    }

    pub fn train(&self, data: &Array2<f64>, y: &Array1<f64>) -> FittedLinearRegression<f64> {
        let dataset = linfa::dataset::DatasetBase::new(data.view(), y.view());
        let model = LinearRegression::new()
            .fit(&dataset)
            .expect("Failed to fit linear regression");
        model
    }

    pub fn get_fitness(&self, x: &Array2<f64>, y: &Array1<f64>, rng: Option<u64>) -> f64 {
        let seed = rng.unwrap_or_else(|| {
            let random = thread_rng().gen_range(0..1000);
            debug_assert!(random >= 0, "Generated random value is invalid");
            random as u64
        });
        
        let mut rng = StdRng::seed_from_u64(seed);
        let (x_train, y_train, x_test, y_test) = self.train_test_split(x, y, 0.2, &mut rng);
        let model = self.train(&x_train, &y_train);
        let preds = model.predict(x_test.view()).targets().to_owned();
        self.rmse(&preds, &y_test)
    }
    
    

    pub fn get_columns(&self, x: &Array2<f64>, bitstring: &Array1<u8>) -> Array2<f64> {
        let mut indices = Vec::new();
        for (col_idx, &flag) in bitstring.iter().enumerate() {
            if flag == 1 {
                indices.push(col_idx);
            }
        }
        x.select(Axis(1), &indices)
    }

    fn rmse(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        let mse = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        mse.sqrt()
    }

    fn train_test_split(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        test_ratio: f64,
        rng: &mut StdRng,
    ) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
        let n = x.nrows();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(rng);

        let test_size = (n as f64 * test_ratio).round() as usize;
        let train_size = n - test_size;

        let (train_indices, test_indices) = indices.split_at(train_size);

        let x_train = Self::select_rows(x, train_indices);
        let y_train = Self::select_entries(y, train_indices);
        let x_test = Self::select_rows(x, test_indices);
        let y_test = Self::select_entries(y, test_indices);

        (x_train, y_train, x_test, y_test)
    }

    fn select_rows(arr: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
    
        let mut selected = Vec::new();
        for &r in rows {
            if r >= arr.nrows() {
                panic!("Row index out of bounds: {} (array has {} rows)", r, arr.nrows());
            }
            let row = arr.row(r).to_owned(); // Create an owned copy of the row
            selected.extend_from_slice(row.as_slice().expect("Failed to convert row to slice"));
        }
        let selected_copy = selected.clone();
        Array2::from_shape_vec((rows.len(), arr.ncols()), selected).unwrap_or_else(|_| {
            panic!(
                "Failed to create Array2: selected length {}, expected shape {:?}",
                selected_copy.len(),
                (rows.len(), arr.ncols())
            )
        })
    }
    
    
    

    fn select_entries(arr: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
        let mut selected = Vec::with_capacity(indices.len());
        for &i in indices {
            selected.push(arr[i]);
        }
        Array1::from(selected)
    }
}
