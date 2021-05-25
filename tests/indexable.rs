use array_map::*;

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

#[test]
fn test_impls() {
  test_indexable::<bool, 2>();
  test_indexable::<u8, 256>();
  test_indexable::<u16, 65536>();
  test_indexable::<Ten, { Ten::count() }>();
  test_indexable::<IndexU8<20>, 20>();
  test_indexable::<IndexU16<264>, 264>();
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
