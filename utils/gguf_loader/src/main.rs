mod types;

use ctensor::tensor_view::{TensorView, Tensor, TensorDType};
use std::fs::File;
use std::io::{Cursor, Read};
use memmap2::Mmap;
use byteorder::{LittleEndian, ReadBytesExt};
use anyhow::Result;
use types::{MetadataValueType, GGMLType};
use serde_json::Value;
use anyhow::anyhow;
use std::collections::HashMap;

#[derive(Debug)]
struct GgufHeader {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}


/*
 * ILoad GGUF metadata + tensor table -> Done 
 * Mmap tensor data section into memory	-> Done 
 * Build TensorViews (slices over mmaped memory)	 Next (easy)
Setup model architecture skeleton	 (define layers + connect)
Load weights into model (using TensorViews)	 (no copies if possible)
Prepare input tokenizer	 (GGUF metadata gives tokenizer info)
Encode prompt → input tensor (tokenizer stage)
Run inference pass (forward) through model (actual compute graph execution)
Sampling step (choose next token) (argmax or temperature sampling)
Repeat until generation stops (full loop)
*/

// 1	TensorView struct (zero-copy memory view)
// 2	Basic Ops (MatMul f32, Add, RMSNorm)
// 3	Attention Block (Query/Key/Value, Scores, Attention Output)
// 4	FeedForward Block (MLP + activation)
// 5	Wire Blocks sequentially (block.0 → block.1 → block.N)
// 6	Tokenizer basic loading (minimal just to feed prompt)
// 7	Forward inference pass
// 8	Sampling next token
// 9	Full text generation loop
// 10	Benchmarks


fn main() -> anyhow::Result<()> {
    // Step 1: Open and mmap the file
    let file = File::open("../models/llama-2-13b-ensemble-v5.Q6_K.gguf")?;
    let mmap = unsafe {
        Mmap::map(&file)?
    };
    let mut cursor = Cursor::new(&mmap[..]);
    
    // Step 2: Parse GGUF header
    let header = GgufHeader {
        magic: cursor.read_u32::<LittleEndian>()?,
        version: cursor.read_u32::<LittleEndian>()?,
        tensor_count: cursor.read_u64::<LittleEndian>()?,
        metadata_kv_count: cursor.read_u64::<LittleEndian>()?,
    };

    println!("[DEBUG] GGUF Header: {:#?}", header);

    // Step 3: Sanity check the magic number
    const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
    if header.magic != GGUF_MAGIC {
        panic!("Not a valid GGUF file!");
    }

    //println!("[DEBUG] GGUF metadata length: {:#?}", header.metadata_kv_count);
    // Step 4: Read metadata key-value pairs
    for _i in 0..header.metadata_kv_count {
        let key = read_string(&mut cursor)?;
        let val_type: MetadataValueType = cursor.read_u32::<LittleEndian>()?.try_into()?;
    //    let val = read_metadata_val(&mut cursor);
        let val = match val_type {
            MetadataValueType::Uint8 => Value::from(cursor.read_u8()?),
            MetadataValueType::Int8 => Value::from(cursor.read_i8()?),
            MetadataValueType::Uint16 => Value::from(cursor.read_u16::<LittleEndian>()?),
            MetadataValueType::Int16 => Value::from(cursor.read_i16::<LittleEndian>()?),
            MetadataValueType::Uint32 => Value::from(cursor.read_u32::<LittleEndian>()?),
            MetadataValueType::Int32 => Value::from(cursor.read_i32::<LittleEndian>()?),
            MetadataValueType::Float32 => Value::from(cursor.read_f32::<LittleEndian>()?),
            MetadataValueType::Bool => Value::from(cursor.read_u8()? == 1),
            MetadataValueType::String => Value::from(read_string(&mut cursor)?),
            MetadataValueType::Array => {
                // Recursive handling!
                let inner_elem_type: MetadataValueType = cursor.read_u32::<LittleEndian>()?.try_into()?;
                let inner_array_len = cursor.read_u64::<LittleEndian>()?;
              //  println!("[DEBUG] inner_array_len: {:#?}", inner_array_len);
                let mut inner_arr = Vec::new();
                for _ in 0..inner_array_len {
                    // Recursive call: match on inner_elem_type again
                    let inner_item = match inner_elem_type {
                        MetadataValueType::Uint8 => Value::from(cursor.read_u8()?),
                        MetadataValueType::Int8 => Value::from(cursor.read_i8()?),
                        MetadataValueType::Uint16 => Value::from(cursor.read_u16::<LittleEndian>()?),
                        MetadataValueType::Int16 => Value::from(cursor.read_i16::<LittleEndian>()?),
                        MetadataValueType::Uint32 => Value::from(cursor.read_u32::<LittleEndian>()?),
                        MetadataValueType::Int32 => Value::from(cursor.read_i32::<LittleEndian>()?),
                        MetadataValueType::Float32 => Value::from(cursor.read_f32::<LittleEndian>()?),
                        MetadataValueType::Bool => Value::from(cursor.read_u8()? != 0),
                        MetadataValueType::String => Value::from(read_string(&mut cursor)?),
                        MetadataValueType::Uint64 => Value::from(cursor.read_u64::<LittleEndian>()?),
                        MetadataValueType::Int64 => Value::from(cursor.read_i64::<LittleEndian>()?),
                        MetadataValueType::Float64 => Value::from(cursor.read_f64::<LittleEndian>()?),
                        _ => return Err(anyhow!("[ERROR] unsupported nested array element")),
                    };
                    
                    inner_arr.push(inner_item);
                }
                Value::Array(inner_arr)
            },
            MetadataValueType::Uint64 => Value::from(cursor.read_u64::<LittleEndian>()?),
            MetadataValueType::Int64 => Value::from(cursor.read_i64::<LittleEndian>()?),
            MetadataValueType::Float64 => Value::from(cursor.read_f64::<LittleEndian>()?),

            //_ => return Err(anyhow!("[ERROR] unsupported element")),
        };
        
        //println!("[DEBUG] kv [{:#?}] index: {} vtype {:#?} key={:#?}", val_type, _i, key, val);
    }

        //let mut tensors = Vec::new();
    let mut tensor_map: HashMap<String, Tensor> = HashMap::new();        
    for _i in 0..header.tensor_count {
        let name = read_string(&mut cursor)?;
        //println!("[DEBUG] tensor name: {:#?}", name);
        let dims = cursor.read_u32::<LittleEndian>()?;
        let mut shape = [1; 4];
        for i in 0..dims {
            shape[i as usize] = cursor.read_u64::<LittleEndian>()?;
        }
        let kind = cursor.read_u32::<LittleEndian>()?;
        let offset = cursor.read_u64::<LittleEndian>()?;
        let block_size = match kind {
            _ if kind < 2 => 1,
            _ if kind < 10 => 32,
            _ => 256,
        };
        let ggml_type_kind: GGMLType = kind.try_into()?;
        let type_size = match ggml_type_kind {
                GGMLType::F32 => 4,
                GGMLType::F16 => 2,
                GGMLType::Q4_0 => 2 + block_size / 2,
                GGMLType::Q4_1 => 2 + 2 + block_size / 2,
                GGMLType::Q5_0 => 2 + 4 + block_size / 2,
                GGMLType::Q5_1 => 2 + 2 + 4 + block_size / 2,
                GGMLType::Q8_0 => 2 + block_size,
                GGMLType::Q8_1 => 4 + 4 + block_size,
                GGMLType::Q2K => block_size / 16 + block_size / 4 + 2 + 2,
                GGMLType::Q3K => block_size / 8 + block_size / 4 + 12 + 2,
                GGMLType::Q4K => 2 + 2 + 12 + block_size / 2,
                GGMLType::Q5K => 2 + 2 + 12 + block_size / 8 + block_size / 2,
                GGMLType::Q6K => block_size / 2 + block_size / 4 + block_size / 16 + 2,
                GGMLType::Q8K => todo!(),
                GGMLType::I8 => todo!(),
                GGMLType::I16 => todo!(),
                GGMLType::I32 => todo!(),
                GGMLType::Count => todo!(),
            };

            let parameters = shape[0] * shape[1] * shape[2] * shape[3];
            let size = parameters * type_size / block_size;

            let tensor = Tensor {
                name,
                kind,
                offset,
                size,
                shape: shape.to_vec(),
            }; 
            tensor_map.insert(tensor.name.clone(), tensor);

            //println!("[DEBUG] tensor name: {:#?}\t\t, offset: {}", name.to_string(), offset);
        //#[cfg(feature = "debug")]
        //{
        //debug!(
        //    "kv [{}] vtype {:?} key={:#?}, value={}",
        //        _i, val_type, key, val);
        //}
        
    }

    println!("[DEBUG] tensor hash: {:#?}", tensor_map);
    
    // 1. get the tensor metadata
    let tensor = tensor_map.get("blk.0.attn_norm.weight").unwrap();

    // 2. Create a live (brething) view
    let view = tensor.view(&mmap)?;

    // 3. Try to slice it as f32 values
    let f32_data = view.as_f32_slice()?;

    // 4. Print a few values -> NO COPY, NO MOVE, first bits of zero copy tensor architecture
    println!("First 10 weights: {:?}", &f32_data[..10]);


    Ok(())
}

//fn read_string(cursor: &mut Cursor<&[u8]>) -> anyhow::Result<String> {
//    let len = cursor.read_u64::<LittleEndian>()?;
//    let mut buf = vec![0u8; len as usize];
//    cursor.read_exact(&mut buf)?;
//    Ok(String::from_utf8_lossy(&buf).to_string())
//}
//

fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let len = cursor.read_u64::<LittleEndian>()?;  // ✅ 4 bytes, NOT 8 bytes
    let mut buf = vec![0u8; len as usize];
    cursor.read_exact(&mut buf)?;
    let s = String::from_utf8(buf)?;
    //println!("[DEBUG] string: {:#?}, {:#?}", len, s);
    Ok(s)
}


fn read_metadata_val(cursor: &mut Cursor<&[u8]>) -> anyhow::Result<()> {
    let val_type = cursor.read_u8()?;
    //match val_type {

    //}

    Ok(())
}
