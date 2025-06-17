use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rvn_core_diff::{diff_header, diff_metadata, diff_tensors};

#[pyfunction]
fn info(path: &str) -> PyResult<PyDataFrame> {
    let gguf = rvn_core_parser::load_model(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load model: {e:?}"))
    })?;

    // Extract comprehensive tensor information
    let tensor_names: Vec<&str> = gguf.tensors.keys().map(String::as_str).collect();
    let tensor_count = tensor_names.len();

    let df = DataFrame::new(vec![
        Series::new("tensor".into(), tensor_names).into(),
        Series::new(
            "shape".into(),
            gguf.tensors
                .values()
                .map(|t| format!("{:?}", t.shape))
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "dtype".into(),
            gguf.tensors
                .values()
                .map(|t| t.kind.to_string())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "n_dims".into(),
            gguf.tensors
                .values()
                .map(|t| t.shape.len() as u32)
                .collect::<Vec<_>>(),
        )
        .into(),
        // Fix the product calculation by dereferencing
        Series::new(
            "n_elements".into(),
            gguf.tensors
                .values()
                .map(|t| t.shape.iter().copied().product::<u64>())
                .collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "size_bytes".into(),
            gguf.tensors
                .values()
                .map(|t| {
                    let elem_count = t.shape.iter().copied().product::<u64>();
                    let elem_size = match t.kind.to_string().as_str() {
                        "F32" | "I32" | "U32" => 4,
                        "F64" | "I64" | "U64" => 8,
                        "F16" | "BF16" | "I16" | "U16" => 2,
                        "I8" | "U8" => 1,
                        "Q4_0" | "Q4_1" => elem_count / 2, // Rough estimate for quantized
                        "Q8_0" => elem_count,
                        _ => 4, // Default fallback
                    };
                    elem_count * elem_size
                })
                .collect::<Vec<_>>(),
        )
        .into(),
    ])
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create DataFrame: {e:?}"
        ))
    })?;

    let enriched_df = df
        .lazy()
        .with_columns([
            lit(tensor_count as u32).alias("total_tensors"),
            lit(path).alias("model_path"),
        ])
        .collect()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to add metadata: {e:?}"
            ))
        })?;

    Ok(PyDataFrame(enriched_df))
}

#[pyfunction]
fn diff(path_a: &str, path_b: &str) -> PyResult<PyDataFrame> {
    let (a, b) = (
        rvn_core_parser::load_model(path_a).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load model A: {e:?}"
            ))
        })?,
        rvn_core_parser::load_model(path_b).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load model B: {e:?}"
            ))
        })?,
    );

    let tensor_diff = diff_tensors(
        a.tensors.iter().map(|(n, t)| (n.as_str(), t)),
        b.tensors.iter().map(|(n, t)| (n.as_str(), t)),
    );

    let header_diff = diff_header(&a.header, &b.header);
    let meta_diff = diff_metadata(&a.metadata, &b.metadata);

    let mut all_changes = Vec::new();
    let mut change_types = Vec::new();
    let mut lhs_shapes = Vec::new();
    let mut rhs_shapes = Vec::new();
    let mut lhs_dtypes = Vec::new();
    let mut rhs_dtypes = Vec::new();
    let mut dtype_changed_flags = Vec::new();
    let mut shape_changed_flags = Vec::new();

    if let Some(tdiff) = tensor_diff {
        // Process changed tensors
        for (name, (ta, tb)) in tdiff.changed.iter() {
            all_changes.push(*name);
            change_types.push("tensor_changed");
            lhs_shapes.push(format!("{:?}", ta.shape));
            rhs_shapes.push(format!("{:?}", tb.shape));
            lhs_dtypes.push(ta.kind.to_string());
            rhs_dtypes.push(tb.kind.to_string());
            dtype_changed_flags.push(ta.kind != tb.kind);
            shape_changed_flags.push(ta.shape != tb.shape);
        }

        // Process added tensors
        for (name, tensor) in tdiff.added.iter() {
            all_changes.push(*name);
            change_types.push("tensor_added");
            lhs_shapes.push("None".to_string());
            rhs_shapes.push(format!("{:?}", tensor.shape));
            lhs_dtypes.push("None".to_string());
            rhs_dtypes.push(tensor.kind.to_string());
            dtype_changed_flags.push(true);
            shape_changed_flags.push(true);
        }

        // Process removed tensors
        for (name, tensor) in tdiff.removed.iter() {
            all_changes.push(*name);
            change_types.push("tensor_removed");
            lhs_shapes.push(format!("{:?}", tensor.shape));
            rhs_shapes.push("None".to_string());
            lhs_dtypes.push(tensor.kind.to_string());
            rhs_dtypes.push("None".to_string());
            dtype_changed_flags.push(true);
            shape_changed_flags.push(true);
        }
    }

    // Add header differences
    if let Some(hdiff) = header_diff {
        for (field_name, (lhs_val, rhs_val)) in hdiff.iter() {
            all_changes.push(field_name);
            change_types.push("header_changed");
            lhs_shapes.push(lhs_val.to_string());
            rhs_shapes.push(rhs_val.to_string());
            lhs_dtypes.push("u64".to_string());
            rhs_dtypes.push("u64".to_string());
            dtype_changed_flags.push(false);
            shape_changed_flags.push(false);
        }
    }

    // Add metadata differences
    if let Some(mdiff) = meta_diff {
        // Changed metadata
        for (key, (lhs_val, rhs_val)) in mdiff.changed.iter() {
            all_changes.push(key);
            change_types.push("metadata_changed");
            lhs_shapes.push(format!("{:?}", lhs_val));
            rhs_shapes.push(format!("{:?}", rhs_val));
            lhs_dtypes.push("metadata".to_string());
            rhs_dtypes.push("metadata".to_string());
            dtype_changed_flags.push(true);
            shape_changed_flags.push(false);
        }

        // Added metadata
        for (key, val) in mdiff.added.iter() {
            all_changes.push(key);
            change_types.push("metadata_added");
            lhs_shapes.push("None".to_string());
            rhs_shapes.push(format!("{:?}", val));
            lhs_dtypes.push("None".to_string());
            rhs_dtypes.push("metadata".to_string());
            dtype_changed_flags.push(true);
            shape_changed_flags.push(false);
        }

        // Removed metadata
        for (key, val) in mdiff.removed.iter() {
            all_changes.push(key);
            change_types.push("metadata_removed");
            lhs_shapes.push(format!("{:?}", val));
            rhs_shapes.push("None".to_string());
            lhs_dtypes.push("metadata".to_string());
            rhs_dtypes.push("None".to_string());
            dtype_changed_flags.push(true);
            shape_changed_flags.push(false);
        }
    }

    // Count changes for metadata before moving the vector
    let n_added = change_types.iter().filter(|&ct| *ct == "added").count() as u32;
    let n_removed = change_types.iter().filter(|&ct| *ct == "removed").count() as u32;
    let n_changed = change_types.iter().filter(|&ct| *ct == "changed").count() as u32;

    let df = DataFrame::new(vec![
        Series::new("tensor".into(), all_changes).into(),
        Series::new("change_type".into(), change_types).into(),
        Series::new("lhs_shape".into(), lhs_shapes).into(),
        Series::new("rhs_shape".into(), rhs_shapes).into(),
        Series::new("lhs_dtype".into(), lhs_dtypes).into(),
        Series::new("rhs_dtype".into(), rhs_dtypes).into(),
        Series::new("dtype_changed".into(), dtype_changed_flags).into(),
        Series::new("shape_changed".into(), shape_changed_flags).into(),
    ])
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create diff DataFrame: {e:?}"
        ))
    })?;

    // Add summary statistics
    let enriched_df = df
        .lazy()
        .with_columns([
            lit(n_added).alias("n_added"),
            lit(n_removed).alias("n_removed"),
            lit(n_changed).alias("n_changed"),
            lit(path_a).alias("lhs_path"),
            lit(path_b).alias("rhs_path"),
        ])
        .collect()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to add metadata: {e:?}"
            ))
        })?;

    Ok(PyDataFrame(enriched_df))
}

#[pymodule]
fn rvnllm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(info, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    Ok(())
}
