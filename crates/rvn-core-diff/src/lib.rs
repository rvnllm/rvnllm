use rvn_core_tensor::{Header, MetaDiff, Tensor, TensorDiff};
use rvn_globals::types::Value;
use std::collections::HashMap;

pub fn diff_header<'a>(
    a: &'a Header,
    b: &'a Header,
) -> Option<Vec<(&'static str, (&'a u64, &'a u64))>> {
    let mut v = Vec::new();
    if a.tensor_count != b.tensor_count {
        v.push(("tensor_count", (&a.tensor_count, &b.tensor_count)));
    }
    if a.metadata_kv_count != b.metadata_kv_count {
        v.push((
            "metadata_kv_count",
            (&a.metadata_kv_count, &b.metadata_kv_count),
        ));
    }
    (!v.is_empty()).then_some(v)
}

pub fn diff_metadata<'a>(
    a: &'a HashMap<String, Value>,
    b: &'a HashMap<String, Value>,
) -> Option<MetaDiff<'a>> {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    for (k, va) in a {
        if k.starts_with("tokenizer") {
            continue;
        } // keep the noise out
        match b.get(k) {
            None => removed.push((k.as_str(), va)),
            Some(vb) if vb != va => changed.push((k.as_str(), (va, vb))),
            _ => {}
        }
    }
    for (k, vb) in b {
        if k.starts_with("tokenizer") {
            continue;
        }
        if !a.contains_key(k) {
            added.push((k.as_str(), vb));
        }
    }
    (!added.is_empty() || !removed.is_empty() || !changed.is_empty()).then_some(MetaDiff {
        added,
        removed,
        changed,
    })
}

pub fn diff_tensors<'a>(
    a: impl Iterator<Item = (&'a str, &'a Tensor)>,
    b: impl Iterator<Item = (&'a str, &'a Tensor)>,
) -> Option<TensorDiff<'a>> {
    use std::collections::HashMap;

    let map_a: HashMap<_, _> = a.collect();
    let map_b: HashMap<_, _> = b.collect();

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    for (k, ta) in &map_a {
        match map_b.get(k) {
            None => removed.push((*k, *ta)),
            Some(tb) if ta.shape != tb.shape || ta.kind != tb.kind => {
                changed.push((*k, (*ta, *tb)))
            }
            _ => {}
        }
    }
    for (k, tb) in &map_b {
        if !map_a.contains_key(k) {
            added.push((*k, *tb));
        }
    }
    (!added.is_empty() || !removed.is_empty() || !changed.is_empty()).then_some(TensorDiff {
        added,
        removed,
        changed,
    })
}
