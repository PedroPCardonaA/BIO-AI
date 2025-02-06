use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Depot {
    return_time: f64,
    x_coord: f64,
    y_coord: f64,
}