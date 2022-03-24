use crate::{set_size, Indexable, ReverseIndexable};

/// An efficient iterator implementation over two Indexable types
///
/// This is the `Iter` type defined on tuples with two members `(T, U)`
#[must_use]
pub struct TwoTupleIter<T: Indexable, U: Indexable> {
  t: T::Iter,
  t_val: Option<T>,
  u: U::Iter,
}

impl<T, U> TwoTupleIter<T, U>
where
  T: Indexable + Clone,
  U: Indexable,
{
  /// Creates a new `TwoTupleIter`
  pub fn new() -> Self {
    let mut t = T::iter();
    Self {
      t_val: t.next(),
      t,
      u: U::iter(),
    }
  }
}

impl<T, U> Default for TwoTupleIter<T, U>
where
  T: Indexable + Clone,
  U: Indexable,
{
  fn default() -> Self {
    Self::new()
  }
}

#[allow(clippy::single_match_else)]
impl<T, U> Iterator for TwoTupleIter<T, U>
where
  T: Indexable + Clone,
  U: Indexable,
{
  type Item = (T, U);

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      match self.t_val.as_ref() {
        None => return None,
        Some(t) => match self.u.next() {
          Some(u) => return Some((t.clone(), u)),
          None => {
            self.t_val = self.t.next();
            self.u = U::iter();
          }
        },
      }
    }
  }
}

#[allow(unsafe_code)]
unsafe impl<T, U> Indexable for (T, U)
where
  T: Indexable + Clone,
  U: Indexable,
{
  const SIZE: usize = T::SIZE * U::SIZE;

  const SET_SIZE: usize = set_size(T::SIZE * U::SIZE);

  type Iter = TwoTupleIter<T, U>;

  fn index(self) -> usize {
    self.0.index() * U::SIZE + self.1.index()
  }

  fn iter() -> Self::Iter {
    TwoTupleIter::new()
  }
}

#[allow(unsafe_code)]
unsafe impl<T, U> ReverseIndexable for (T, U)
where
  T: ReverseIndexable + Clone,
  U: ReverseIndexable,
{
  fn from_index(u: usize) -> Self {
    let t_index = u / U::SIZE;
    let u_index = u % U::SIZE;
    (T::from_index(t_index), U::from_index(u_index))
  }
}

#[test]
fn test_two_tuple() {
  use crate::{IndexU8, Indexable, ReverseIndexable};
  type Double = (IndexU8<16>, u8);
  let array_map = crate::ArrayMap::<Double, u16, { <Double>::SIZE }>::from_closure(|(l, r)| ((l.get() as u16) << 8) + (*r as u16));
  for (i, ((l, r), v)) in array_map.iter().enumerate() {
    assert_eq!(*v as usize, i);
    assert_eq!(r as u16 & *v, r as u16);
    assert_eq!(*v >> 8, l.get() as u16);
    assert_eq!((l, r), <Double>::from_index(i), "l: {l:?}, r: {r}, i: {i}");
  }
}
