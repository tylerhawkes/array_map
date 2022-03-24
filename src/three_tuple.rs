use crate::{set_size, Indexable, ReverseIndexable, TwoTupleIter};

/// An efficient iterator implementation over three Indexable types
///
/// This is the `Iter` type defined on tuples with three members `(T, U, V)`
#[must_use]
pub struct ThreeTupleIter<T: Indexable, U: Indexable, V: Indexable> {
  t: TwoTupleIter<T, U>,
  t_val: Option<(T, U)>,
  v: V::Iter,
}

impl<T, U, V> ThreeTupleIter<T, U, V>
where
  T: Indexable + Clone,
  U: Indexable + Clone,
  V: Indexable,
{
  /// Creates a new `TwoTupleIter`
  pub fn new() -> Self {
    let mut t = TwoTupleIter::new();
    Self {
      t_val: t.next(),
      t,
      v: V::iter(),
    }
  }
}

impl<T, U, V> Default for ThreeTupleIter<T, U, V>
where
  T: Indexable + Clone,
  U: Indexable + Clone,
  V: Indexable,
{
  fn default() -> Self {
    Self::new()
  }
}

#[allow(clippy::single_match_else)]
impl<T, U, V> Iterator for ThreeTupleIter<T, U, V>
where
  T: Indexable + Clone,
  U: Indexable + Clone,
  V: Indexable,
{
  type Item = (T, U, V);

  fn next(&mut self) -> Option<Self::Item> {
    loop {
      match self.t_val.as_ref() {
        None => return None,
        Some(t) => match self.v.next() {
          Some(v) => return Some((t.0.clone(), t.1.clone(), v)),
          None => {
            self.t_val = self.t.next();
            self.v = V::iter();
          }
        },
      }
    }
  }
}

#[allow(unsafe_code)]
unsafe impl<T, U, V> Indexable for (T, U, V)
where
  T: Indexable + Clone,
  U: Indexable + Clone,
  V: Indexable,
{
  const SIZE: usize = T::SIZE * U::SIZE * V::SIZE;

  const SET_SIZE: usize = set_size(T::SIZE * U::SIZE * V::SIZE);

  type Iter = ThreeTupleIter<T, U, V>;

  fn index(self) -> usize {
    (self.0, self.1).index() * V::SIZE + self.2.index()
  }

  fn iter() -> Self::Iter {
    ThreeTupleIter::new()
  }
}

#[allow(unsafe_code)]
unsafe impl<T, U, V> ReverseIndexable for (T, U, V)
where
  T: ReverseIndexable + Clone,
  U: ReverseIndexable + Clone,
  V: ReverseIndexable,
{
  fn from_index(u: usize) -> Self {
    let t_index = u / V::SIZE;
    let v_index = u % V::SIZE;
    let (t, u) = <(T, U) as ReverseIndexable>::from_index(t_index);
    (t, u, V::from_index(v_index))
  }
}

#[test]
fn test_three_tuple() {
  use crate::{IndexU8, Indexable, ReverseIndexable};
  type Triple = (IndexU8<16>, IndexU8<16>, u8);
  println!("size: {}", Triple::SIZE);
  println!("set_size: {}", Triple::SET_SIZE);
  let array_map = crate::ArrayMap::<Triple, u16, { Triple::SIZE }>::from_closure(|(l, c, r)| {
    let l = (l.get() as u16) << 12;
    let c = (c.get() as u16) << 8;
    let r = *r as u16;
    l + c + r
  });
  for (i, ((l, c, r), v)) in array_map.iter().enumerate() {
    assert_eq!(*v as usize, i);
    assert_eq!(*v >> 12, l.get() as u16);
    assert_eq!((*v << 4) >> 12, c.get() as u16);
    assert_eq!(r as u16 & *v, r as u16);
    assert_eq!((l, c, r), Triple::from_index(i), "l: {l:?}, r: {r}, i: {i}");
  }
}
