mod structs;
mod utils;

fn main() {
    let instance = utils::parse_data::parse_data("src/data/train/train_0.json");
    println!("{:?}", instance);
}
