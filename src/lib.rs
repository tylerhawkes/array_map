#![cfg_attr(
  not(test),
  deny(warnings, clippy::all, clippy::pedantic, clippy::cargo, missing_docs, missing_crate_level_docs)
)]
#![deny(unsafe_code)]
#![cfg_attr(not(test), no_std)]
#![allow(clippy::module_name_repetitions, clippy::inline_always)]

//! `no_std` compatible Map and Set backed by arrays.
//! This crate will evolve as more const-generic features become available.
//!
//! Features:
//! * `derive`: Includes the [`Indexable`](array_map_derive::Indexable) derive macro
//! * `serde`: Includes serde impls for [`ArrayMap`] and [`ArraySet`]
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

#[allow(unsafe_code)]
unsafe impl Indexable for bool {
  const SIZE: usize = 2;
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
unsafe impl Indexable for u8 {
  const SIZE: usize = 256;
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
unsafe impl Indexable for u16 {
  const SIZE: usize = 65536;
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

#[cfg(test)]
mod test {
  use super::*;

  #[derive(Copy, Clone, Debug, PartialEq, Eq, Indexable)]
  pub enum Ten {
    Zero,
    One,
    Two,
    Three,
    Four,
    #[index(disabled)]
    Five,
    Six,
    Seven,
    Eight,
    Nine,
  }

  #[derive(Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Debug, ::serde::Serialize, ::serde::Deserialize)]
  #[serde(transparent)]
  pub struct Lowercase(pub(crate) char);
  #[allow(unsafe_code)]
  unsafe impl Indexable for Lowercase {
    const SIZE: usize = 26;
    type Iter = core::iter::Map<core::ops::RangeInclusive<char>, fn(char) -> Lowercase>;

    fn index(self) -> usize {
      self.0 as usize - 'a' as usize
    }

    fn iter() -> Self::Iter {
      ('a'..='z').map(Lowercase)
    }
  }

  #[test]
  fn test_impls() {
    test_indexable::<bool, 2>();
    test_indexable::<u8, 256>();
    test_indexable::<u16, 65536>();
    test_indexable::<Ten, { Ten::count() }>();
  }

  fn test_indexable<I: Indexable, const N: usize>() {
    let mut iter = I::iter();
    for i in 0..N {
      let x = iter.next().unwrap();
      assert_eq!(x.index(), i);
    }
    assert!(iter.next().is_none());
    assert_eq!(I::iter().count(), N);
  }

  #[test]
  #[should_panic]
  fn test_disabled() {
    Ten::Five.index();
  }
}
