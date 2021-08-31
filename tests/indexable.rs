use array_map::*;
use array_map_derive::Indexable;

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
  test_indexable::<Option<Ten>, { Ten::count() + 1 }>();
  test_indexable::<Option<bool>, 3>();
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

#[test]
#[should_panic]
fn test_option_disabled() {
  Some(Ten::Five).index();
}

#[test]
fn test_option_none() {
  assert_eq!(Option::<Ten>::None.index(), 9);
}
