#![cfg_attr(not(test), deny(warnings, clippy::all, clippy::pedantic, clippy::cargo, missing_docs))]
#![deny(unsafe_code)]
#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![allow(clippy::module_name_repetitions, clippy::inline_always)]

//! `no_std` compatible Map and Set backed by arrays.
//! This crate will evolve as more const-generic features become available.
//!
//! Features:
//! * `derive`: Includes the [`Indexable`](array_map_derive::Indexable) derive macro
//! * `serde`: Includes serde impls for [`ArrayMap`] and [`ArraySet`]
//! * `std`: Includes [`From`](core::convert::From) implementations between [`ArrayMap<K, AtomicBool, M>`](ArrayMap) and [`ArraySet<K, S>`](ArraySet)
//!
//! This is especially useful if you have a bare enum where you want to treat each key as a field. This
//! also has the benefit that adding a new enum variant forces you to update your code.
//! ```
//! use array_map::*;
//! #[repr(u8)]
//! #[derive(Indexable)]
//! enum DetectionType {
//!   Person,
//!   Vehicle,
//!   Bicycle,
//! }
//!
//! let thresholds = ArrayMap::<DetectionType, f32, {DetectionType::count()}>::from_closure(|dt| match dt {
//!     DetectionType::Person => 0.8,
//!     DetectionType::Vehicle => 0.9,
//!     DetectionType::Bicycle => 0.7,
//!   });
//!
//! let person_threshold = thresholds[DetectionType::Person];
//!
//! ```
//!
//! This can also be used to memoize some common computations
//! (this is 2x as fast as doing the computation on aarch64)
//! ```
//! use array_map::*;
//! let u8_to_f32_cache = ArrayMap::<u8, f32, {u8::SIZE}>::from_closure(|u|f32::from(*u) / 255.0);
//! # let bytes = vec![0_u8; 1024];
//! // take some bytes and convert them to f32
//! let floats = bytes.iter().copied().map(|b|u8_to_f32_cache[b]).collect::<Vec<_>>();
//! ```

mod map;
#[cfg(feature = "serde")]
mod serde;
mod set;

#[cfg(feature = "array_map_derive")]
pub use array_map_derive::Indexable;
use core::convert::TryInto;
pub use map::*;
pub use set::*;

/// Allows mapping from a type to an index
///
/// This trait is unsafe because there are a few other requirements that need to be met:
/// * The indexes of self need to be contiguous and need to be returned in order from `iter()`
/// * Iter needs to return exactly N items (this is checked in debug mode)
#[allow(unsafe_code)]
pub unsafe trait Indexable {
  /// The number of items or variants that this type can have.
  const SIZE: usize;
  /// The number of bytes it will take to represent this type in a set.
  /// # Safety
  /// This must equal `set_size(Self::SIZE)`
  const SET_SIZE: usize;
  /// The type of Iterator that will be returned by [`Self::iter()`]
  type Iter: Iterator<Item = Self>;
  /// Maps self to usize to know which value in the underling array to use
  /// # Safety
  /// This value can never be greater than [`Self::SIZE`]
  ///
  /// All values in `0..SIZE` must be returned by some variant of self.
  fn index(self) -> usize;
  /// An Iterator over all valid values of `self`
  /// # Safety
  /// This iterator must yield exactly [`Self::SIZE`] items
  fn iter() -> Self::Iter;
}

/// Allows mapping from an index to a type
///
/// This is trait is unsafe because it needs to uphold the property that it is reflexive.
/// For example:
/// ```
/// use array_map::*;
/// let index = 42_u8.index();
/// let u = u8::from_index(index);
/// assert_eq!(u, 42);
/// ```
///
/// If a value greater than or equal to [`Self::SIZE`](Indexable::SIZE) is provided, then this function can panic.
/// This library will never pass in an invalid usize value.
#[allow(unsafe_code)]
pub unsafe trait ReverseIndexable: Indexable {
  /// Converts from a usize to `Self`
  fn from_index(u: usize) -> Self;
}

/// Wrapper around another iterator where items are changed to `Option<Iterator::Item>` and the last value emitted is None.
#[derive(Clone)]
pub struct OptionIter<T> {
  inner: Option<T>,
}

impl<I, T> Iterator for OptionIter<T>
where
  T: Iterator<Item = I>,
{
  type Item = Option<I>;

  #[allow(clippy::single_match_else, clippy::option_if_let_else)]
  fn next(&mut self) -> Option<Self::Item> {
    if let Some(inner) = &mut self.inner {
      Some(match inner.next() {
        Some(v) => Some(v),
        None => {
          self.inner = None;
          None
        }
      })
    } else {
      None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    match &self.inner {
      Some(t) => {
        let (min, max) = t.size_hint();
        (min + 1, max.map(|x| x + 1))
      }
      None => (0, Some(0)),
    }
  }

  fn count(self) -> usize
  where
    Self: Sized,
  {
    match self.inner {
      Some(t) => t.count() + 1,
      None => 0,
    }
  }
}

impl<T> ExactSizeIterator for OptionIter<T>
where
  T: ExactSizeIterator,
{
  fn len(&self) -> usize {
    self.size_hint().0
  }
}

#[allow(unsafe_code)]
unsafe impl<T: Indexable> Indexable for Option<T> {
  const SIZE: usize = T::SIZE + 1;
  const SET_SIZE: usize = set_size(T::SIZE + 1);
  type Iter = OptionIter<T::Iter>;

  fn index(self) -> usize {
    match self {
      Some(x) => x.index(),
      None => T::SIZE,
    }
  }

  fn iter() -> Self::Iter {
    OptionIter { inner: Some(T::iter()) }
  }
}

#[allow(unsafe_code)]
unsafe impl<T: ReverseIndexable> ReverseIndexable for Option<T> {
  fn from_index(u: usize) -> Self {
    if u == T::SIZE {
      None
    } else {
      Some(T::from_index(u))
    }
  }
}

#[allow(unsafe_code)]
unsafe impl Indexable for bool {
  const SIZE: usize = 2;
  const SET_SIZE: usize = set_size(2);
  type Iter = core::array::IntoIter<bool, 2>;
  #[inline(always)]
  fn index(self) -> usize {
    self as usize
  }
  #[inline(always)]
  fn iter() -> Self::Iter {
    core::array::IntoIter::new([false, true])
  }
}

#[allow(unsafe_code)]
unsafe impl ReverseIndexable for bool {
  fn from_index(u: usize) -> Self {
    if u == 0 {
      false
    } else if u == 1 {
      true
    } else {
      panic!("Invalid bool index provided {}", u)
    }
  }
}

#[allow(unsafe_code)]
unsafe impl Indexable for u8 {
  const SIZE: usize = u8::MAX as usize + 1;
  const SET_SIZE: usize = set_size(u8::MAX as usize + 1);
  type Iter = core::ops::RangeInclusive<Self>;
  #[inline(always)]
  fn index(self) -> usize {
    self as usize
  }
  #[inline(always)]
  fn iter() -> Self::Iter {
    Self::MIN..=Self::MAX
  }
}

#[allow(unsafe_code)]
unsafe impl ReverseIndexable for u8 {
  fn from_index(u: usize) -> Self {
    u.try_into().unwrap_or_else(|_| panic!("Invalid u8 index provided {}", u))
  }
}

#[allow(unsafe_code)]
unsafe impl Indexable for u16 {
  const SIZE: usize = u16::MAX as usize + 1;
  const SET_SIZE: usize = set_size(u16::MAX as usize + 1);
  type Iter = core::ops::RangeInclusive<Self>;
  #[inline(always)]
  fn index(self) -> usize {
    self as usize
  }
  #[inline(always)]
  fn iter() -> Self::Iter {
    Self::MIN..=Self::MAX
  }
}

#[allow(unsafe_code)]
unsafe impl ReverseIndexable for u16 {
  fn from_index(u: usize) -> Self {
    u.try_into().unwrap_or_else(|_| panic!("Invalid u16 index provided {}", u))
  }
}

/// This struct implements [`Indexable`] and allows all values in `0..N`
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[repr(transparent)]
#[must_use]
pub struct IndexU8<const N: u8>(u8);

impl<const N: u8> IndexU8<N> {
  /// Returns a new `IndexU8` if it is in a valid range
  #[inline]
  #[must_use]
  pub const fn new(u: u8) -> Option<Self> {
    if u < N {
      Some(Self(u))
    } else {
      None
    }
  }
  /// Returns the underlying u8
  #[inline(always)]
  #[must_use]
  pub const fn get(self) -> u8 {
    self.0
  }
}

#[allow(unsafe_code)]
unsafe impl<const N: u8> Indexable for IndexU8<N> {
  const SIZE: usize = N as usize;
  const SET_SIZE: usize = set_size(N as usize);
  type Iter = core::iter::Map<core::ops::Range<u8>, fn(u8) -> Self>;

  fn index(self) -> usize {
    self.0 as usize
  }

  fn iter() -> Self::Iter {
    (0..N).map(Self)
  }
}

#[allow(unsafe_code)]
unsafe impl<const N: u8> ReverseIndexable for IndexU8<N> {
  fn from_index(u: usize) -> Self {
    let u: u8 = u.try_into().unwrap_or_else(|_| panic!("Invalid IndexU8 index provided {}", u));
    Self::new(u).unwrap_or_else(|| panic!("Invalid IndexU8 index provided {}", u))
  }
}

/// This struct implements [`Indexable`] and allows all values in `0..N`
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[repr(transparent)]
#[must_use]
pub struct IndexU16<const N: u16>(u16);

impl<const N: u16> IndexU16<N> {
  /// Returns a new `IndexU8` if it is in a valid range
  #[inline]
  #[must_use]
  pub const fn new(u: u16) -> Option<Self> {
    if u < N {
      Some(Self(u))
    } else {
      None
    }
  }
  /// Returns the underlying u8
  #[inline(always)]
  #[must_use]
  pub const fn get(self) -> u16 {
    self.0
  }
}

#[allow(unsafe_code)]
unsafe impl<const N: u16> Indexable for IndexU16<N> {
  const SIZE: usize = N as usize;
  const SET_SIZE: usize = set_size(N as usize);
  type Iter = core::iter::Map<core::ops::Range<u16>, fn(u16) -> Self>;

  fn index(self) -> usize {
    self.0 as usize
  }

  fn iter() -> Self::Iter {
    (0..N).map(Self)
  }
}

#[allow(unsafe_code)]
unsafe impl<const N: u16> ReverseIndexable for IndexU16<N> {
  fn from_index(u: usize) -> Self {
    let u: u16 = u.try_into().unwrap_or_else(|_| panic!("Invalid IndexU8 index provided {}", u));
    Self::new(u).unwrap_or_else(|| panic!("Invalid IndexU8 index provided {}", u))
  }
}

#[cfg(test)]
mod test {

  #[derive(Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Debug, ::serde::Serialize, ::serde::Deserialize)]
  #[serde(transparent)]
  pub struct Lowercase(pub(crate) char);
  #[allow(unsafe_code)]
  unsafe impl crate::Indexable for Lowercase {
    const SIZE: usize = 26;
    const SET_SIZE: usize = crate::set_size(26);
    type Iter = core::iter::Map<core::ops::RangeInclusive<char>, fn(char) -> Lowercase>;

    fn index(self) -> usize {
      self.0 as usize - 'a' as usize
    }

    fn iter() -> Self::Iter {
      ('a'..='z').map(Lowercase)
    }
  }
}
