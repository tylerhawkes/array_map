use crate::Indexable;
use core::marker::PhantomData;
use core::mem::MaybeUninit;

/// A Map backed by an array. The keys must be known statically.
///
/// This map is O(1) for all operations.
/// All members must be initialized at creation time so there is no option
/// to insert or remove items from the map. Because of this it can be indexed into using the key type.
#[must_use]
pub struct ArrayMap<K, V, const N: usize> {
  pub(crate) array: [V; N],
  pub(crate) phantom: PhantomData<*const K>,
}

#[allow(unsafe_code)]
unsafe impl<K, V: Send, const N: usize> Send for ArrayMap<K, V, N> {}
#[allow(unsafe_code)]
unsafe impl<K, V: Sync, const N: usize> Sync for ArrayMap<K, V, N> {}
impl<K, V: Unpin, const N: usize> Unpin for ArrayMap<K, V, N> {}
#[cfg(feature = "std")]
impl<K, V: std::panic::UnwindSafe, const N: usize> std::panic::UnwindSafe for ArrayMap<K, V, N> {}

impl<K, V, const N: usize> ArrayMap<K, V, N> {
  /// Function that can be used in const contexts to get a new `ArrayMap`.
  ///
  /// Callers should ensure that each `V` in the array matches the return of `index()` from `K` to get back expected results.
  #[allow(unsafe_code)]
  pub const fn new(array: [V; N]) -> Self {
    Self {
      array,
      phantom: PhantomData,
    }
  }
}

impl<K: Indexable, V: Clone, const N: usize> ArrayMap<K, V, N> {
  /// Returns a new [`ArrayMap`] where all the values are initialized to the same value
  #[inline(always)]
  pub fn from_value(v: V) -> Self {
    Self::from_closure(|_| v.clone())
  }
}

impl<K, V: Clone, const N: usize> Clone for ArrayMap<K, V, N> {
  #[inline(always)]
  fn clone(&self) -> Self {
    Self {
      array: self.array.clone(),
      phantom: PhantomData,
    }
  }
}

impl<K, V: Copy, const N: usize> Copy for ArrayMap<K, V, N> {}

impl<K, V: PartialEq, const N: usize> PartialEq for ArrayMap<K, V, N> {
  fn eq(&self, other: &Self) -> bool {
    self.array.eq(&other.array)
  }
}

impl<K, V: Eq, const N: usize> Eq for ArrayMap<K, V, N> {}

impl<K: core::fmt::Debug + Indexable, V: core::fmt::Debug, const N: usize> core::fmt::Debug for ArrayMap<K, V, N> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    assert_eq!(N, K::SIZE);
    f.debug_map().entries(self.iter()).finish()
  }
}

impl<K: Indexable, V, const N: usize> ArrayMap<K, V, N> {
  /// Returns a new [`ArrayMap`] where all the values are initialized to the value returned from each call of the closure.
  /// # Panics
  /// Panics if [`K::iter()`](Indexable::iter()) returns anything other than `N` items or if [`K::index()`](Indexable::iter()) returns a value >= `N`
  #[allow(unsafe_code)]
  pub fn from_closure(mut f: impl FnMut(&K) -> V) -> Self {
    assert_eq!(N, K::SIZE);
    let mut array = MaybeUninit::<[V; N]>::uninit();
    debug_assert_eq!(N, K::iter().take(N + 1).count());
    debug_assert!(K::iter().take(K::SIZE + 1).enumerate().all(|(l, i)| l == i.index()));
    let mut filled = [false; N];
    // Safety: we only write to values without reading them.
    K::iter().for_each(|t| {
      let v = f(&t);
      let index = t.index();
      assert!(index < N);
      unsafe { array.as_mut_ptr().cast::<V>().add(index).write(v) };
      filled[index] = true;
    });
    assert!(
      filled.iter().all(|f| *f),
      "Not all indexes have been set. Indexable::index() for {} probably isn't unique",
      core::any::type_name::<K>()
    );
    Self {
      array: unsafe { array.assume_init() },
      phantom: PhantomData,
    }
  }

  /// Returns a new [`ArrayMap`] where all the values are initialized to the same value
  /// # Errors
  /// Returns an error the first time that the provided closure returns an error.
  /// # Panics
  /// Panics if [`K::iter()`](Indexable::iter()) returns anything other than `N` items or if [`K::index()`](Indexable::iter()) returns a value >= `N`
  #[allow(unsafe_code)]
  pub fn try_from_closure<E>(mut f: impl FnMut(&K) -> Result<V, E>) -> Result<Self, E> {
    assert_eq!(N, K::SIZE);
    let mut array = MaybeUninit::<[V; N]>::uninit();
    debug_assert_eq!(N, K::iter().take(N + 1).count());
    debug_assert!(K::iter().take(K::SIZE + 1).enumerate().all(|(l, i)| l == i.index()));
    let mut filled = [false; N];
    // Safety: we only write to values without reading them.
    for k in K::iter() {
      match f(&k) {
        Ok(v) => {
          let index = k.index();
          assert!(index < N);
          unsafe { array.as_mut_ptr().cast::<V>().add(index).write(v) };
          filled[index] = true;
        }
        Err(e) => {
          // need to drop everything that has already been initialized.
          if core::mem::needs_drop::<V>() {
            for index in filled.iter().copied().enumerate().filter_map(|(i, b)| b.then(|| i)) {
              unsafe {
                let ptr = array.as_mut_ptr().cast::<V>().add(index);
                core::ptr::drop_in_place(ptr);
              }
            }
          }
          return Err(e);
        }
      }
    }
    assert!(
      filled.iter().all(|f| *f),
      "Not all indexes have been set. Indexable::index() for {} probably isn't unique",
      core::any::type_name::<K>()
    );
    Ok(Self {
      array: unsafe { array.assume_init() },
      phantom: PhantomData,
    })
  }

  /// Collects an iterator of `(K, Result<V, E>)` into a `Result<Self, E>`
  ///
  /// The trait [`FromIterator`](core::iter::FromIterator) cannot be implemented for [`Result`](core::result::Result)
  /// like it is on a Vec because of the orphan rules.
  ///
  /// # Errors
  /// Returns an error with the first emitted error from the provided iterator
  /// # Panics
  /// Panics if any of the safety requirements in the [`Indexable`](crate::Indexable) trait are wrong
  #[allow(unsafe_code)]
  pub fn from_results<E, I: IntoIterator<Item = (K, Result<V, E>)>>(iter: I) -> Result<Self, E> {
    assert_eq!(N, K::SIZE);
    let mut array = MaybeUninit::<[V; N]>::uninit();
    debug_assert_eq!(N, K::iter().take(N + 1).count());
    debug_assert!(K::iter().take(K::SIZE + 1).enumerate().all(|(l, i)| l == i.index()));
    let mut filled = [false; N];
    for (k, r) in iter {
      match r {
        Ok(v) => {
          let index = k.index();
          assert!(index < N);
          // Safety: we only write to values without reading them.
          unsafe { array.as_mut_ptr().cast::<V>().add(index).write(v) };
          filled[index] = true;
        }
        Err(e) => {
          // need to drop everything that has already been initialized.
          if core::mem::needs_drop::<V>() {
            for index in filled.iter().copied().enumerate().filter_map(|(i, b)| b.then(|| i)) {
              unsafe {
                let ptr = array.as_mut_ptr().cast::<V>().add(index);
                core::ptr::drop_in_place(ptr);
              }
            }
          }
          return Err(e);
        }
      }
    }
    assert!(
      filled.iter().all(|f| *f),
      "Not all indexes have been set. Indexable::index() for {} probably isn't unique",
      core::any::type_name::<K>()
    );
    Ok(Self {
      // Safety: We have ensured all values are initialized.
      array: unsafe { array.assume_init() },
      phantom: PhantomData,
    })
  }

  /// Collects an iterator of `(K, Option<V>)` into an `Option<Self>`
  ///
  /// # Panics
  /// Panics if any of the safety requirements in the [`Indexable`](crate::Indexable) trait are wrong
  #[allow(unsafe_code)]
  pub fn from_options<E, I: IntoIterator<Item = (K, Option<V>)>>(iter: I) -> Option<Self> {
    assert_eq!(N, K::SIZE);
    let mut array = MaybeUninit::<[V; N]>::uninit();
    debug_assert_eq!(N, K::iter().take(N + 1).count());
    debug_assert!(K::iter().take(K::SIZE + 1).enumerate().all(|(l, i)| l == i.index()));
    let mut filled = [false; N];
    for (k, r) in iter {
      #[allow(clippy::single_match_else)]
      match r {
        Some(v) => {
          let index = k.index();
          assert!(index < N);
          // Safety: we only write to values without reading them.
          unsafe { array.as_mut_ptr().cast::<V>().add(index).write(v) };
          filled[index] = true;
        }
        None => {
          // need to drop everything that has already been initialized.
          if core::mem::needs_drop::<V>() {
            for index in filled.iter().copied().enumerate().filter_map(|(i, b)| b.then(|| i)) {
              unsafe {
                let ptr = array.as_mut_ptr().cast::<V>().add(index);
                core::ptr::drop_in_place(ptr);
              }
            }
          }
          return None;
        }
      }
    }
    assert!(
      filled.iter().all(|f| *f),
      "Not all indexes have been set. Indexable::index() for {} probably isn't unique",
      core::any::type_name::<K>()
    );
    Some(Self {
      // Safety: We have ensured all values are initialized.
      array: unsafe { array.assume_init() },
      phantom: PhantomData,
    })
  }

  /// Returns an iterator over all the items in the map. Note that the keys are owned.
  /// # Panics
  /// Panics if any of the safety requirements in the [`Indexable`](crate::Indexable) trait are wrong
  #[inline(always)]
  pub fn iter(&self) -> impl Iterator<Item = (K, &V)> + '_ {
    assert_eq!(N, K::SIZE);
    debug_assert_eq!(N, K::iter().take(N + 1).count());
    debug_assert!(K::iter().take(K::SIZE + 1).enumerate().all(|(l, i)| l == i.index()));
    K::iter().zip(self.array.iter())
  }
  /// Returns a mutable iterator over all the items in the map
  /// # Panics
  /// Panics if any of the safety requirements in the [`Indexable`](crate::Indexable) trait are wrong
  #[inline(always)]
  pub fn iter_mut(&mut self) -> impl Iterator<Item = (K, &mut V)> + '_ {
    assert_eq!(N, K::SIZE);
    debug_assert_eq!(N, K::iter().take(N + 1).count());
    debug_assert!(K::iter().take(K::SIZE + 1).enumerate().all(|(l, i)| l == i.index()));
    K::iter().zip(self.array.iter_mut())
  }
  /// Returns an iterator over all the keys in the map. Note that the keys are owned.
  #[inline(always)]
  #[allow(clippy::unused_self)]
  pub fn keys(&self) -> impl Iterator<Item = K> + '_ {
    K::iter()
  }
}

impl<K, V, const N: usize> ArrayMap<K, V, N> {
  /// Returns an iterator over all the values in the map
  #[inline]
  pub fn values(&self) -> impl Iterator<Item = &V> + '_ {
    self.array.iter()
  }
  /// Returns a mutable iterator over all the values in the map
  #[inline]
  pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> + '_ {
    self.array.iter_mut()
  }
}

impl<K: Indexable, V, const N: usize> core::iter::IntoIterator for ArrayMap<K, V, N> {
  type Item = (K, V);
  type IntoIter = core::iter::Zip<K::Iter, core::array::IntoIter<V, N>>;

  fn into_iter(self) -> Self::IntoIter {
    K::iter().zip(core::array::IntoIter::new(self.array))
  }
}

impl<K: Indexable, V, const N: usize> core::iter::FromIterator<(K, V)> for ArrayMap<K, Option<V>, N> {
  fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
    let mut this = Self::from_closure(|_| None);
    for (t, u) in iter {
      this.array[t.index()] = Some(u);
    }
    this
  }
}

impl<K: Indexable, V: Default, const N: usize> Default for ArrayMap<K, V, N> {
  #[inline(always)]
  fn default() -> Self {
    Self::from_closure(|_| V::default())
  }
}

impl<K, V, const N: usize> core::convert::AsRef<[V; N]> for ArrayMap<K, V, N> {
  #[inline(always)]
  fn as_ref(&self) -> &[V; N] {
    &self.array
  }
}

impl<K, V, const N: usize> core::convert::AsMut<[V; N]> for ArrayMap<K, V, N> {
  #[inline(always)]
  fn as_mut(&mut self) -> &mut [V; N] {
    &mut self.array
  }
}

impl<K: Indexable, V, const N: usize> core::ops::Index<K> for ArrayMap<K, V, N> {
  type Output = V;

  #[inline(always)]
  fn index(&self, index: K) -> &Self::Output {
    assert_eq!(N, K::SIZE);
    &self.array[index.index()]
  }
}

impl<K: Indexable, V, const N: usize> core::ops::IndexMut<K> for ArrayMap<K, V, N> {
  #[inline(always)]
  fn index_mut(&mut self, index: K) -> &mut Self::Output {
    assert_eq!(N, K::SIZE);
    &mut self.array[index.index()]
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::test::*;

  fn send_sync_traits<
    K: Indexable,
    V,
    T: Send
      + Sync
      + Unpin
      + Default
      + Clone
      + Copy
      + PartialEq
      + Eq
      + core::fmt::Debug
      + core::convert::AsRef<[V; N]>
      + core::convert::AsMut<[V; N]>
      + core::ops::Index<K>
      + core::ops::IndexMut<K>,
    const N: usize,
  >() {
  }

  fn test_traits<
    I: Indexable + Copy + PartialEq + core::fmt::Debug,
    V: Send + Sync + Unpin + Default + Clone + Copy + PartialEq + Eq + core::fmt::Debug,
    F: FnMut(I) -> V,
    const N: usize,
  >(
    mut f: F,
  ) {
    send_sync_traits::<I, V, ArrayMap<I, V, N>, N>();
    let mut whole_map = ArrayMap::<I, V, N>::default();
    for i in I::iter() {
      assert_eq!(whole_map[i], V::default());
      let new_value = f(i);
      whole_map[i] = new_value;
      assert_eq!(whole_map[i], new_value);
      assert!(whole_map.iter().any(|(k, v)| k == i && v == &new_value));
    }
    assert_eq!(whole_map.keys().count(), I::SIZE);
    assert_eq!(whole_map.iter().count(), I::SIZE);
    assert_eq!(whole_map.iter_mut().count(), I::SIZE);
  }

  #[test]
  fn test_impls() {
    test_traits::<bool, Option<(u8, u8)>, _, { bool::SIZE }>(|b| Some((b as u8 + 1, b as u8)));
    test_traits::<u8, u32, _, { u8::SIZE }>(|u| u as u32 + 1);
    test_traits::<Lowercase, &'static str, _, { Lowercase::SIZE }>(|_| "abc");
  }
}
