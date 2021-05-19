use crate::Indexable;
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

impl<K: Indexable, const N: usize> ArraySet<K, N> {
  /// Creates a new, empty [`ArraySet`]
  #[inline]
  pub fn new() -> Self {
    assert_eq!(set_size(K::SIZE), N);
    debug_assert_eq!(K::SIZE, K::iter().count());
    Self {
      set: [0; N],
      phantom: PhantomData,
    }
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
  #[inline]
  fn query<R>(&self, t: K, f: impl FnOnce(u8, u8) -> R) -> R {
    let index = t.index();
    let byte = index >> 3;
    debug_assert!(byte < N);
    let bit = index & 0x7;
    let mask = 1_u8 << bit;
    f(self.set[byte], mask)
  }
  #[inline]
  fn mutate<R>(&mut self, t: K, f: impl FnOnce(&mut u8, u8) -> R) -> R {
    let index = t.index();
    let byte = index >> 3;
    debug_assert!(byte < N);
    let bit = index & 0x7;
    let mask = 1_u8 << bit;
    f(&mut self.set[byte], mask)
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
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::test::*;

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
    test_traits::<Ten, { crate::set_size(Ten::count()) }>();
  }
}
