# array_map
`no_std` compatible Map and Set backed by arrays.

This crate will evolve as more const-generic features become available.

This is especially useful if you have a bare enum where you want to treat each key as a field
 ```rust
 use array_map::*;
 #[repr(u8)]
 #[derive(Indexable)]
 enum DetectionType {
   Person,
   Vehicle,
   Bicycle,
 }
 let thresholds = ArrayMap::<DetectionType, f32, {DetectionType::count()}>::from_closure(|dt| match dt {
     DetectionType::Person => 0.8,
     DetectionType::Vehicle => 0.9,
     DetectionType::Bicycle => 0.7,
   });
 let person_threshold = thresholds[DetectionType::Person];
 ```
This can also be used to memoize some common computations.
(this is 2x as fast as doing the computation on aarch64)
 ```rust
 use array_map::*;
 let u8_to_f32_cache = ArrayMap::<u8, f32, {u8::SIZE}>::from_closure(|u|f32::from(*u) / 255.0);
 let bytes = vec![0_u8; 1024];
 // take some bytes and convert them to f32
 let floats = bytes.iter().copied().map(|b|u8_to_f32_cache[b]).collect::<Vec<_>>();
 ```