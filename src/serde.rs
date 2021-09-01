use crate::{ArrayMap, ArraySet, Indexable};
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use serde::de::{MapAccess, SeqAccess};
use serde::{Deserializer, Serializer};

impl<K: serde::Serialize + Indexable, V: serde::Serialize, const N: usize> serde::Serialize for ArrayMap<K, V, N> {
  fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
  where
    S: Serializer,
  {
    debug_assert_eq!(N, K::iter().count());
    serializer.collect_map(self.iter())
  }
}

struct ExpectingN(usize);

impl serde::de::Expected for ExpectingN {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl<'de, K: serde::Deserialize<'de> + Indexable, V: serde::Deserialize<'de>, const N: usize> serde::Deserialize<'de>
  for ArrayMap<K, V, N>
{
  fn deserialize<D>(deserializer: D) -> Result<Self, <D as Deserializer<'de>>::Error>
  where
    D: Deserializer<'de>,
  {
    struct ArrayMapVisitor<K: Indexable, V, const N: usize> {
      array: MaybeUninit<[V; N]>,
      filled: [bool; N],
      phantom: PhantomData<fn() -> K>,
    }
    impl<'v, K: serde::Deserialize<'v> + Indexable, V: serde::Deserialize<'v>, const N: usize> serde::de::Visitor<'v>
      for ArrayMapVisitor<K, V, N>
    {
      type Value = ArrayMap<K, V, N>;

      fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "A map of {} values", N)
      }

      #[allow(unsafe_code)]
      fn visit_map<A>(mut self, mut map: A) -> Result<Self::Value, <A as MapAccess<'v>>::Error>
      where
        A: MapAccess<'v>,
      {
        while let Some((k, v)) = map.next_entry::<K, V>()? {
          let index = k.index();
          assert!(index < N);
          // Safety: we can only write to uninit before trying to read them which we do here
          unsafe {
            self.array.as_mut_ptr().cast::<V>().add(index).write(v);
          }
          self.filled[index] = true;
        }
        let count = self.filled.iter().filter(|f| **f).count();
        if count != N {
          use serde::de::Error;
          return Err(<A as MapAccess<'v>>::Error::invalid_length(count, &ExpectingN(N)));
        }
        Ok(ArrayMap {
          // Safety we have guaranteed that all the slots have been filled
          array: unsafe { self.array.assume_init() },
          phantom: PhantomData,
        })
      }
    }
    debug_assert_eq!(N, K::iter().count());
    deserializer.deserialize_map(ArrayMapVisitor {
      array: MaybeUninit::uninit(),
      filled: [false; N],
      phantom: PhantomData,
    })
  }
}

#[test]
fn test_array_map_serde() {
  use crate::test::Lowercase;
  type Map = ArrayMap<Lowercase, Option<(u8, u8)>, { Lowercase::SIZE }>;
  let mut h = Map::default();
  h[Lowercase('b')] = Some((50, 80));
  h[Lowercase('c')] = Some((10, 20));
  let s = serde_json::to_string(&h).unwrap();
  let h_new = serde_json::from_str::<Map>(&s).unwrap();
  assert_eq!(h, h_new);
}

impl<K: serde::Serialize + Indexable, const N: usize> serde::Serialize for ArraySet<K, N> {
  fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
  where
    S: Serializer,
  {
    debug_assert_eq!(K::SIZE, K::iter().count());
    serializer.collect_seq(self.keys())
  }
}

impl<'de, K: serde::Deserialize<'de> + Indexable, const N: usize> serde::Deserialize<'de> for ArraySet<K, N> {
  fn deserialize<D>(deserializer: D) -> Result<Self, <D as Deserializer<'de>>::Error>
  where
    D: Deserializer<'de>,
  {
    struct ArraySetVisitor<K: Indexable, const N: usize> {
      set: ArraySet<K, N>,
    }
    impl<'v, K: serde::Deserialize<'v> + Indexable, const N: usize> serde::de::Visitor<'v> for ArraySetVisitor<K, N> {
      type Value = ArraySet<K, N>;

      fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "A sequence of values")
      }

      #[allow(unsafe_code)]
      fn visit_seq<A>(mut self, mut seq: A) -> Result<Self::Value, <A as SeqAccess<'v>>::Error>
      where
        A: SeqAccess<'v>,
      {
        while let Some(k) = seq.next_element::<K>()? {
          self.set.insert(k);
        }
        Ok(self.set)
      }
    }
    debug_assert_eq!(K::SIZE, K::iter().count());
    deserializer.deserialize_seq(ArraySetVisitor { set: ArraySet::default() })
  }
}

#[test]
fn test_array_set_serde() {
  use crate::test::Lowercase;
  type Set = ArraySet<Lowercase, { crate::set_size(Lowercase::SIZE) }>;
  let mut h = Set::default();
  h.insert(Lowercase('b'));
  h.insert(Lowercase('c'));
  let s = serde_json::to_string(&h).unwrap();
  let h_new = serde_json::from_str::<Set>(&s).unwrap();
  assert_eq!(h, h_new);
}
