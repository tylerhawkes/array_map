#![deny(clippy::all, clippy::pedantic, warnings)]

#[derive(array_map::Indexable)]
pub enum NonZero {
  One,
  Two,
  Three,
  Four,
  Five,
}
