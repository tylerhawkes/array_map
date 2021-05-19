extern crate proc_macro;

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Data, DeriveInput, Ident};

/// Derive macro for the [Indexable](https://docs.rs/array_map/trait.Indexable.html) trait.
///
/// This properly derives the trait and upholds all the safety invariants.
///
/// Variants can be disabled by adding `#[index(disabled)]`. If [Indexable::index()](https://docs.rs/array_map/trait.Indexable.html#tymethod.index) is ever called
/// on that variant then it will panic.
#[proc_macro_derive(Indexable, attributes(index))]
pub fn indexable(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
  let ast = syn::parse_macro_input!(input as DeriveInput);

  derive(&ast).unwrap_or_else(|err| err.to_compile_error()).into()
}

// Most of this code was lifted from https://github.com/Peternator7/strum/blob/master/strum_macros/src/lib.rs
fn derive(ast: &DeriveInput) -> syn::Result<TokenStream> {
  let name = &ast.ident;
  let gen = &ast.generics;
  let (impl_generics, ty_generics, where_clause) = gen.split_for_impl();
  let vis = &ast.vis;

  if gen.lifetimes().count() > 0 {
    return Err(syn::Error::new(
      Span::call_site(),
      "This macro doesn't support enums with lifetimes. \
             The resulting enums would be unbounded.",
    ));
  }

  let phantom_data = if gen.type_params().count() > 0 {
    let g = gen.type_params().map(|param| &param.ident);
    quote! { < ( #(#g),* ) > }
  } else {
    quote! { < () > }
  };

  match &ast.data {
    Data::Enum(v) => {
      let variants = &v.variants;
      let mut arms = Vec::new();
      let mut to_index = Vec::new();
      let mut idx = 0_usize;
      let mut disabled = false;
      for variant in variants {
        use syn::Fields::*;

        if variant.get_variant_properties()?.disabled.is_some() {
          disabled = true;
          continue;
        }

        let ident = &variant.ident;
        let params = match &variant.fields {
          Unit => quote! {},
          // Unnamed(fields) => {
          //   let defaults = ::std::iter::repeat(quote!(::core::default::Default::default())).take(fields.unnamed.len());
          //   quote! { (#(#defaults),*) }
          // }
          // Named(fields) => {
          //   let fields = fields.named.iter().map(|field| field.ident.as_ref().unwrap());
          //   quote! { {#(#fields: ::core::default::Default::default()),*} }
          // }
          _ => return Err(syn::Error::new(Span::call_site(), "This macro doesn't yet support enums with data")),
        };

        arms.push(quote! {#idx => ::core::option::Option::Some(#name::#ident #params)});
        to_index.push(quote! {#name::#ident #params => #idx});
        idx += 1;
      }

      let variant_count = arms.len();
      arms.push(quote! { _ => ::core::option::Option::None });
      if disabled {
        to_index.push(quote! { _ => panic!("Invalid variant") });
      }
      let iter_name = syn::parse_str::<Ident>(&format!("{}IndexableIter", name)).unwrap();

      Ok(quote! {
          #[allow(missing_docs)]
          #vis struct #iter_name #ty_generics {
              idx: usize,
              back_idx: usize,
              marker: ::core::marker::PhantomData #phantom_data,
          }

          #[allow(missing_docs)]
          impl #impl_generics #iter_name #ty_generics #where_clause {
              fn get(&self, idx: usize) -> Option<#name #ty_generics> {
                  match idx {
                      #(#arms),*
                  }
              }
          }

          #[allow(missing_docs)]
          impl #impl_generics #name #ty_generics #where_clause {
              pub fn iter() -> #iter_name #ty_generics {
                  #iter_name {
                      idx: 0,
                      back_idx: 0,
                      marker: ::core::marker::PhantomData,
                  }
              }
              pub const fn count() -> usize {
                  #variant_count
              }
          }

          impl #impl_generics Iterator for #iter_name #ty_generics #where_clause {
              type Item = #name #ty_generics;

              fn next(&mut self) -> Option<Self::Item> {
                  self.nth(0)
              }

              fn size_hint(&self) -> (usize, Option<usize>) {
                  let t = if self.idx + self.back_idx >= #variant_count { 0 } else { #variant_count - self.idx - self.back_idx };
                  (t, Some(t))
              }

              fn nth(&mut self, n: usize) -> Option<Self::Item> {
                  let idx = self.idx + n + 1;
                  if idx + self.back_idx > #variant_count {
                      // We went past the end of the iterator. Freeze idx at #variant_count
                      // so that it doesn't overflow if the user calls this repeatedly.
                      // See PR #76 for context.
                      self.idx = #variant_count;
                      None
                  } else {
                      self.idx = idx;
                      self.get(idx - 1)
                  }
              }
          }

          impl #impl_generics ExactSizeIterator for #iter_name #ty_generics #where_clause {
              fn len(&self) -> usize {
                  self.size_hint().0
              }
          }

          impl #impl_generics DoubleEndedIterator for #iter_name #ty_generics #where_clause {
              fn next_back(&mut self) -> Option<Self::Item> {
                  let back_idx = self.back_idx + 1;

                  if self.idx + back_idx > #variant_count {
                      // We went past the end of the iterator. Freeze back_idx at #variant_count
                      // so that it doesn't overflow if the user calls this repeatedly.
                      // See PR #76 for context.
                      self.back_idx = #variant_count;
                      None
                  } else {
                      self.back_idx = back_idx;
                      self.get(#variant_count - self.back_idx)
                  }
              }
          }

          impl #impl_generics Clone for #iter_name #ty_generics #where_clause {
              fn clone(&self) -> #iter_name #ty_generics {
                  #iter_name {
                      idx: self.idx,
                      back_idx: self.back_idx,
                      marker: self.marker.clone(),
                  }
              }
          }

          #[allow(unsafe_code)]
          unsafe impl #impl_generics Indexable for #name #ty_generics #where_clause {
              const SIZE: usize = Self::count();
              type Iter = #iter_name #ty_generics;
              fn index(self) -> usize {
                  match self {
                      #(#to_index),*
                  }
              }
              fn iter() -> Self::Iter {
                  Self::iter()
              }
          }
      })
    }
    _ => Err(syn::Error::new(
      Span::call_site(),
      "This macro only supports enums and transparent structs",
    )),
  }
}

trait HasIndexableVariantProperties {
  fn get_variant_properties(&self) -> syn::Result<IndexableVariantProperties>;
}

// #[derive(Clone, Eq, PartialEq, Debug, Default)]
#[derive(Default)]
struct IndexableVariantProperties {
  pub disabled: Option<kw::disabled>,
  pub default: Option<kw::default>,
  // ident: Option<syn::Ident>,
}

// impl IndexableVariantProperties {
//   fn ident_as_str(&self, case_style: Option<CaseStyle>) -> LitStr {
//     let ident = self.ident.as_ref().expect("identifier");
//     LitStr::new(&ident.convert_case(case_style), ident.span())
//   }
// }

impl HasIndexableVariantProperties for syn::Variant {
  fn get_variant_properties(&self) -> syn::Result<IndexableVariantProperties> {
    let mut output = IndexableVariantProperties {
      // ident: Some(self.ident.clone()),
      ..Default::default()
    };

    for meta in self.get_metadata()? {
      match meta {
        VariantMeta::Disabled(kw) => {
          if let Some(ref fst_kw) = output.disabled {
            return Err(occurrence_error(fst_kw, &kw, "disabled"));
          }
          output.disabled = Some(kw);
        }
        VariantMeta::Default(kw) => {
          if let Some(ref fst_kw) = output.default {
            return Err(occurrence_error(fst_kw, &kw, "default"));
          }
          output.default = Some(kw);
        }
      }
    }

    Ok(output)
  }
}

mod kw {
  syn::custom_keyword!(disabled);
  syn::custom_keyword!(default);
}

enum VariantMeta {
  Disabled(kw::disabled),
  Default(kw::default),
}

impl syn::parse::Parse for VariantMeta {
  fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
    let lookahead = input.lookahead1();
    if lookahead.peek(kw::disabled) {
      Ok(VariantMeta::Disabled(input.parse()?))
    } else if lookahead.peek(kw::default) {
      Ok(VariantMeta::Default(input.parse()?))
    } else {
      Err(lookahead.error())
    }
  }
}

impl syn::spanned::Spanned for VariantMeta {
  fn span(&self) -> Span {
    match self {
      VariantMeta::Disabled(kw) => kw.span,
      VariantMeta::Default(kw) => kw.span,
    }
  }
}

trait VariantExt {
  /// Get all the metadata associated with an enum variant.
  fn get_metadata(&self) -> syn::Result<Vec<VariantMeta>>;
}

impl VariantExt for syn::Variant {
  fn get_metadata(&self) -> syn::Result<Vec<VariantMeta>> {
    get_metadata_inner("index", &self.attrs)
  }
}

fn get_metadata_inner<'a, T: syn::parse::Parse + syn::spanned::Spanned>(
  ident: &str,
  it: impl IntoIterator<Item = &'a syn::Attribute>,
) -> syn::Result<Vec<T>> {
  it.into_iter()
    .filter(|attr| attr.path.is_ident(ident))
    .try_fold(Vec::new(), |mut vec, attr| {
      vec.extend(attr.parse_args_with(syn::punctuated::Punctuated::<T, syn::Token![,]>::parse_terminated)?);
      Ok(vec)
    })
}

fn occurrence_error<T: quote::ToTokens>(fst: T, snd: T, attr: &str) -> syn::Error {
  let mut e = syn::Error::new_spanned(snd, format!("Found multiple occurrences of strum({})", attr));
  e.combine(syn::Error::new_spanned(fst, "first one here"));
  e
}
