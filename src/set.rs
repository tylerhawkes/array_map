use crate::{ArrayMap, Indexable};
use core::marker::PhantomData;

/// A set backed by an array. All possible keys must be known statically.
///
/// This set is O(1) for all operations
#[must_use]
pub struct ArraySet<K, const N: usize> {
  pub(crate) set: [u8; N],
  // This allows Send, Sync, and Unpin to be implemented correctly
  pub(crate) phantom: PhantomData<fn() -> K>,
}

/// Convenience function used to determine the underlying size needed for an `ArraySet`
#[must_use]
pub const fn set_size(size: usize) -> usize {
  if size.trailing_zeros() >= 3 {
    size >> 3
  } else {
    (size >> 3) + 1
  }
}

impl<K: Indexable, const N: usize> Default for ArraySet<K, N> {
  fn default() -> Self {
    Self::new()
  }
}

impl<K, const N: usize> Clone for ArraySet<K, N> {
  fn clone(&self) -> Self {
    Self {
      set: self.set,
      phantom: PhantomData,
    }
  }
}

impl<K, const N: usize> Copy for ArraySet<K, N> {}

impl<K, const N: usize> PartialEq for ArraySet<K, N> {
  fn eq(&self, other: &Self) -> bool {
    self.set.eq(&other.set)
  }
}

impl<K, const N: usize> Eq for ArraySet<K, N> {}

impl<K: core::fmt::Debug + Indexable, const N: usize> core::fmt::Debug for ArraySet<K, N> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_set().entries(self.keys()).finish()
  }
}

impl<K: Indexable, const N: usize> ArraySet<K, N> {
  /// Creates a new, empty [`ArraySet`]
  #[inline]
  pub fn new() -> Self {
    Self::empty()
  }
  #[inline(always)]
  fn query<R>(&self, t: K, f: impl FnOnce(u8, u8) -> R) -> R {
    let index = t.index();
    let byte = index >> 3;
    assert!(byte < N);
    let bit = index & 0x7;
    let mask = 1_u8 << bit;
    // # Safety
    // we have already asserted that byte < N so getting this is fine
    #[allow(unsafe_code)]
    f(unsafe { *self.set.get_unchecked(byte) }, mask)
  }
  #[inline(always)]
  fn mutate<R>(&mut self, t: K, f: impl FnOnce(&mut u8, u8) -> R) -> R {
    let index = t.index();
    let byte = index >> 3;
    assert!(byte < N);
    let bit = index & 0x7;
    let mask = 1_u8 << bit;
    // # Safety
    // we have already asserted that byte < N so getting this is fine
    #[allow(unsafe_code)]
    f(unsafe { self.set.get_unchecked_mut(byte) }, mask)
  }
  /// Determines whether a key already exists in the set
  #[inline]
  #[must_use]
  pub fn contains(&self, t: K) -> bool {
    self.query(t, |b, m| b & m != 0)
  }
  /// Inserts a key into the set and returns whether it was already contained in the set
  #[inline]
  pub fn insert(&mut self, t: K) -> bool {
    self.mutate(t, |b, m| {
      let contained = *b & m != 0;
      *b |= m;
      contained
    })
  }
  /// Removes a key from the set and returns whether it was already contained in the set
  #[inline]
  pub fn remove(&mut self, t: K) -> bool {
    self.mutate(t, |b, m| {
      let contained = *b & m != 0;
      *b &= !m;
      contained
    })
  }
  /// Returns an iterator over all values that are in the set
  #[inline]
  pub fn keys(&self) -> impl Iterator<Item = K> + '_ {
    K::iter()
      .zip(K::iter())
      .filter_map(move |(q, t)| if self.contains(q) { Some(t) } else { None })
  }
  /// Returns whether all possible keys are in the set
  #[inline]
  #[must_use]
  pub fn is_full(&self) -> bool {
    self.count_ones() as usize >= K::SIZE
  }
  #[inline(always)]
  fn set_excess_zero(&mut self) {
    if K::SIZE.trailing_zeros() < 3 {
      self.set[N - 1] &= Self::last_mask();
    }
  }
  fn last_mask() -> u8 {
    if K::SIZE.trailing_zeros() < 3 {
      let used_last = K::SIZE & 0x7;
      !0_u8 >> (8 - used_last)
    } else {
      !0
    }
  }
  /// Creates a new, empty [`ArraySet`]
  #[inline]
  pub fn empty() -> Self {
    assert_eq!(set_size(K::SIZE), N);
    debug_assert_eq!(K::SIZE, K::iter().count());
    Self {
      set: [0; N],
      phantom: PhantomData,
    }
  }
  /// Creates a new, full [`ArraySet`]
  #[inline]
  pub fn full() -> Self {
    !Self::empty()
  }
}

impl<K, const N: usize> ArraySet<K, N> {
  const fn count_ones(&self) -> u32 {
    let mut n = 0;
    let mut count = 0;
    loop {
      if n == N {
        return count;
      }
      count += self.set[n].count_ones();
      n += 1;
    }
  }
  const fn count_zeros(&self) -> u32 {
    let mut n = 0;
    let mut count = 0;
    loop {
      if n == N {
        return count;
      }
      count += self.set[n].count_zeros();
      n += 1;
    }
  }
  /// Returns whether this set contains any keys
  #[inline]
  #[must_use]
  pub const fn is_empty(&self) -> bool {
    self.count_zeros() == 0
  }
}

impl<K: Indexable, const N: usize> From<K> for ArraySet<K, N> {
  fn from(k: K) -> Self {
    let mut set = Self::new();
    set.insert(k);
    set
  }
}

impl<K: Indexable, const N: usize> core::iter::FromIterator<K> for ArraySet<K, N> {
  fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
    let mut set = Self::new();
    for k in iter {
      set.insert(k);
    }
    set
  }
}

impl<K: Indexable, const N: usize> core::ops::Not for ArraySet<K, N> {
  type Output = Self;

  #[inline]
  fn not(mut self) -> Self::Output {
    for byte in self.set.iter_mut() {
      *byte = !*byte;
    }
    self.set_excess_zero();
    self
  }
}

impl<K: Indexable, const N: usize> core::ops::Add<K> for ArraySet<K, N> {
  type Output = Self;

  fn add(mut self, k: K) -> Self::Output {
    self.insert(k);
    self
  }
}

impl<K: Indexable, const N: usize> core::ops::Sub<K> for ArraySet<K, N> {
  type Output = Self;

  fn sub(mut self, k: K) -> Self::Output {
    self.remove(k);
    self
  }
}

impl<K: Indexable, const N: usize> core::ops::BitAnd for ArraySet<K, N> {
  type Output = Self;

  #[inline]
  fn bitand(mut self, rhs: Self) -> Self::Output {
    for (l, r) in self.set.iter_mut().zip(rhs.set.iter().copied()) {
      *l &= r;
    }
    self.set_excess_zero();
    self
  }
}

impl<K: Indexable, const N: usize> core::ops::BitAndAssign for ArraySet<K, N> {
  #[inline]
  fn bitand_assign(&mut self, rhs: Self) {
    *self = *self & rhs;
  }
}

impl<K: Indexable, const N: usize> core::ops::BitOr for ArraySet<K, N> {
  type Output = Self;

  #[inline]
  fn bitor(mut self, rhs: Self) -> Self::Output {
    for (l, r) in self.set.iter_mut().zip(rhs.set.iter().copied()) {
      *l |= r;
    }
    self.set_excess_zero();
    self
  }
}

impl<K: Indexable, const N: usize> core::ops::BitOrAssign for ArraySet<K, N> {
  #[inline]
  fn bitor_assign(&mut self, rhs: Self) {
    *self = *self | rhs;
  }
}

impl<K: Indexable, const N: usize> core::ops::BitXor for ArraySet<K, N> {
  type Output = Self;

  #[inline]
  fn bitxor(mut self, rhs: Self) -> Self::Output {
    for (l, r) in self.set.iter_mut().zip(rhs.set.iter().copied()) {
      *l ^= r;
    }
    self.set_excess_zero();
    self
  }
}

impl<K: Indexable, const N: usize> core::ops::BitXorAssign for ArraySet<K, N> {
  #[inline]
  fn bitxor_assign(&mut self, rhs: Self) {
    *self = *self ^ rhs
  }
}

impl<K: Indexable, const M: usize, const S: usize> From<crate::ArrayMap<K, bool, M>> for ArraySet<K, S> {
  fn from(m: ArrayMap<K, bool, M>) -> Self {
    let mut set = ArraySet::new();
    for (k, &v) in m.iter() {
      if v {
        set.insert(k);
      }
    }
    set
  }
}

impl<K: Indexable, const M: usize, const S: usize> From<ArraySet<K, S>> for crate::ArrayMap<K, bool, M> {
  fn from(s: ArraySet<K, S>) -> Self {
    let mut map = ArrayMap::default();
    for k in s.keys() {
      map[k] = true;
    }
    map
  }
}

#[cfg(feature = "std")]
mod atomics {
  use crate::{ArrayMap, ArraySet, Indexable};
  use core::sync::atomic::{AtomicBool, Ordering};
  impl<K: Indexable, const M: usize, const S: usize> From<&crate::ArrayMap<K, AtomicBool, M>> for ArraySet<K, S> {
    fn from(m: &ArrayMap<K, AtomicBool, M>) -> Self {
      let mut set = ArraySet::new();
      for (k, v) in m.iter() {
        if v.load(Ordering::SeqCst) {
          set.insert(k);
        }
      }
      set
    }
  }

  impl<K: Indexable, const M: usize, const S: usize> From<ArraySet<K, S>> for crate::ArrayMap<K, AtomicBool, M> {
    fn from(s: ArraySet<K, S>) -> Self {
      let mut map = ArrayMap::<K, AtomicBool, M>::default();
      for k in s.keys() {
        *map[k].get_mut() = true;
      }
      map
    }
  }
}

impl<K: Indexable> ArraySet<K, 1> {
  /// Returns this set as a u8 (or anything that it can be turned into) with the appropriate bits
  /// representing each of the keys in this set.
  #[inline]
  #[must_use]
  pub fn to_int<U>(self) -> U
  where
    u8: Into<U>,
  {
    self.set[0].into()
  }

  /// Creates a new `ArraySet` from the given u8. Bits may be cleared to ensure this is in a valid state.
  pub fn from_int(u: u8) -> Self {
    let mut set = Self{ set: [u], phantom: PhantomData };
    set.set_excess_zero();
    set
  }
}

impl<K: Indexable> ArraySet<K, 2> {
  /// Returns this set as a u16 (or anything that it can be turned into) with the appropriate bits
  /// representing each of the keys in this set.
  #[inline]
  #[must_use]
  pub fn to_int<U>(self) -> U
  where
    u16: Into<U>,
  {
    (self.set[0] as u16 | ((self.set[1] as u16) << 8)).into()
  }

  /// Creates a new `ArraySet` from the given u16. Bits may be cleared to ensure this is in a valid state.
  pub fn from_int(u: u16) -> Self {
    let mut set = Self{ set: [u as u8, (u >> 8) as u8], phantom: PhantomData };
    set.set_excess_zero();
    set
  }
}

impl<K: Indexable> ArraySet<K, 3> {
  /// Returns this set as a u32 (or anything that it can be turned into) with the appropriate bits
  /// representing each of the keys in this set.
  #[inline]
  #[must_use]
  pub fn to_int<U>(self) -> U
  where
    u32: Into<U>,
  {
    let mut u = 0_u32;
    for (i, &k) in self.set.iter().enumerate() {
      u |= (k as u32) << (i as u32 * 8);
    }
    u.into()
  }

  /// Creates a new `ArraySet` from the given u32. Bits may be cleared to ensure this is in a valid state.
  pub fn from_int(u: u32) -> Self {
    let mut set = Self::empty();
    for (i, b) in set.set.iter_mut().enumerate() {
      *b = (u >> (i * 8)) as u8;
    }
    set.set_excess_zero();
    set
  }
}

impl<K: Indexable> ArraySet<K, 4> {
  /// Returns this set as a u32 (or anything that it can be turned into) with the appropriate bits
  /// representing each of the keys in this set.
  #[inline]
  #[must_use]
  pub fn to_int<U>(self) -> U
    where
      u32: Into<U>,
  {
    let mut u = 0_u32;
    for (i, &k) in self.set.iter().enumerate() {
      u |= (k as u32) << (i as u32 * 8);
    }
    u.into()
  }

  /// Creates a new `ArraySet` from the given u32. Bits may be cleared to ensure this is in a valid state.
  pub fn from_int(u: u32) -> Self {
    let mut set = Self::empty();
    for (i, b) in set.set.iter_mut().enumerate() {
      *b = (u >> (i * 8)) as u8;
    }
    set.set_excess_zero();
    set
  }
}

impl<K: Indexable> ArraySet<K, 5> {
  /// Returns this set as a u64 (or anything that it can be turned into) with the appropriate bits
  /// representing each of the keys in this set.
  #[inline]
  #[must_use]
  pub fn to_int<U>(self) -> U
    where
      u64: Into<U>,
  {
    let mut u = 0_u64;
    for (i, &k) in self.set.iter().enumerate() {
      u |= (k as u64) << (i as u64 * 8);
    }
    u.into()
  }

  /// Creates a new `ArraySet` from the given u64. Bits may be cleared to ensure this is in a valid state.
  pub fn from_int(u: u64) -> Self {
    let mut set = Self::empty();
    for (i, b) in set.set.iter_mut().enumerate() {
      *b = (u >> (i * 8)) as u8;
    }
    set.set_excess_zero();
    set
  }
}

impl<K: Indexable> ArraySet<K, 6> {
  /// Returns this set as a u64 (or anything that it can be turned into) with the appropriate bits
  /// representing each of the keys in this set.
  #[inline]
  #[must_use]
  pub fn to_int<U>(self) -> U
    where
      u64: Into<U>,
  {
    let mut u = 0_u64;
    for (i, &k) in self.set.iter().enumerate() {
      u |= (k as u64) << (i as u64 * 8);
    }
    u.into()
  }

  /// Creates a new `ArraySet` from the given u64. Bits may be cleared to ensure this is in a valid state.
  pub fn from_int(u: u64) -> Self {
    let mut set = Self::empty();
    for (i, b) in set.set.iter_mut().enumerate() {
      *b = (u >> (i * 8)) as u8;
    }
    set.set_excess_zero();
    set
  }
}

impl<K: Indexable> ArraySet<K, 7> {
  /// Returns this set as a u64 (or anything that it can be turned into) with the appropriate bits
  /// representing each of the keys in this set.
  #[inline]
  #[must_use]
  pub fn to_int<U>(self) -> U
    where
      u64: Into<U>,
  {
    let mut u = 0_u64;
    for (i, &k) in self.set.iter().enumerate() {
      u |= (k as u64) << (i as u64 * 8);
    }
    u.into()
  }

  /// Creates a new `ArraySet` from the given u64. Bits may be cleared to ensure this is in a valid state.
  pub fn from_int(u: u64) -> Self {
    let mut set = Self::empty();
    for (i, b) in set.set.iter_mut().enumerate() {
      *b = (u >> (i * 8)) as u8;
    }
    set.set_excess_zero();
    set
  }
}

impl<K: Indexable> ArraySet<K, 8> {
  /// Returns this set as a u64 (or anything that it can be turned into) with the appropriate bits
  /// representing each of the keys in this set.
  #[inline]
  #[must_use]
  pub fn to_int<U>(self) -> U
    where
      u64: Into<U>,
  {
    let mut u = 0_u64;
    for (i, &k) in self.set.iter().enumerate() {
      u |= (k as u64) << (i as u64 * 8);
    }
    u.into()
  }

  /// Creates a new `ArraySet` from the given u64. Bits may be cleared to ensure this is in a valid state.
  pub fn from_int(u: u64) -> Self {
    let mut set = Self::empty();
    for (i, b) in set.set.iter_mut().enumerate() {
      *b = (u >> (i * 8)) as u8;
    }
    set.set_excess_zero();
    set
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::test::*;
  use crate::IndexU8;

  fn send_sync_traits<T: Send + Sync + Default + Clone + Copy + PartialEq + Eq + core::fmt::Debug>() {}

  fn test_traits<I: Indexable + Copy + PartialEq + core::fmt::Debug, const N: usize>() {
    send_sync_traits::<ArraySet<I, N>>();
    let mut whole_set = ArraySet::<I, N>::new();
    for i in I::iter() {
      let mut set = ArraySet::<I, N>::new();
      assert!(!set.contains(i));
      set.insert(i);
      assert!(set.contains(i));
      let mut iter = set.keys();
      assert_eq!(iter.next().unwrap(), i);
      assert!(iter.next().is_none());
      whole_set.insert(i);
    }
    // Make sure each key is using a unique bit
    for (i, k) in I::iter().zip(whole_set.keys()) {
      assert_eq!(i, k);
    }
    assert_eq!(whole_set.keys().count(), I::SIZE);
  }

  #[test]
  fn test_impls() {
    test_traits::<bool, { crate::set_size(bool::SIZE) }>();
    test_traits::<u8, { crate::set_size(u8::SIZE) }>();
    test_traits::<Lowercase, { crate::set_size(Lowercase::SIZE) }>();
  }

  fn test_roundtrip(v: u8) {
    let mut u = 0_u64;
    for i in 0..8 {
      u |= (v as u64) << (i * 8);
    }

    assert_eq!(u as u8, <ArraySet<IndexU8<8>, {crate::set_size(8)}>>::from_int(u as u8).to_int::<u8>());
    assert_eq!(u as u16, <ArraySet<IndexU8<16>, {crate::set_size(16)}>>::from_int(u as u16).to_int::<u16>());
    assert_eq!(u as u32 >> 8, <ArraySet<IndexU8<24>, {crate::set_size(24)}>>::from_int(u as u32).to_int::<u32>());
    assert_eq!(u as u32, <ArraySet<IndexU8<32>, {crate::set_size(32)}>>::from_int(u as u32).to_int::<u32>());
    assert_eq!(u >> 24, <ArraySet<IndexU8<40>, {crate::set_size(40)}>>::from_int(u).to_int::<u64>());
    assert_eq!(u >> 16, <ArraySet<IndexU8<48>, {crate::set_size(48)}>>::from_int(u).to_int::<u64>());
    assert_eq!(u >> 8, <ArraySet<IndexU8<56>, {crate::set_size(56)}>>::from_int(u).to_int::<u64>());
    assert_eq!(u, <ArraySet<IndexU8<64>, {crate::set_size(64)}>>::from_int(u).to_int::<u64>());
  }

  #[test]
  fn test_from_to_int() {
    for i in 0..=255 {
      test_roundtrip(i);
    }
  }

}
